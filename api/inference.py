import datetime
from typing import Any, Dict
import numpy as np
from sklearn.preprocessing import StandardScaler
import zuko
import torch
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def ge_model_path(disaster_type):
    return f"../runs/best_flow_{disaster_type}.pt"

def _tensor_to_py(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, (list, tuple)):
        return [_tensor_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _tensor_to_py(v) for k, v in x.items()}
    return x


async def predict_dist(request) -> Dict[str, Any]:
    # expected feature names (same order used by the web form)
    fields = [
        'BEGIN_LAT','BEGIN_LON','DURATION_HOURS','WIND_SPEED','DATE','PRECIPITATION',
        'TMIN','TMAX','ELEVATION','SLOPE',
        'COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN',
        'RIVER_DISTANCE','SEA_DISTANCE'
    ]

    # Accept either a dict-like payload or a Request
    if isinstance(request, dict):
        payload = request
    else:
        # Try to parse form first (the UI posts a FormData), otherwise JSON
        try:
            form = await request.form()
            source = 'form' if form and len(form) > 0 else 'json'
        except Exception:
            form = None
            source = 'json'

        payload = {}
        if source == 'form' and form is not None:
            for f in fields + ['EVENT_TYPE']:
                v = form.get(f)
                payload[f] = v
        else:
            try:
                payload = await request.json()
            except Exception:
                payload = {}

    # EVENT_TYPE selects the model
    disaster = payload.get('EVENT_TYPE') or payload.get('EVENT') or payload.get('event_type') or 'flood'

    # Validate and convert numeric fields
    data = {}
    for f in fields:
        v = payload.get(f)
        if v is None:
            data[f] = None
        else:
            if f == 'DATE':
                data[f] = v
            else:
                try:
                    data[f] = float(v)
                except Exception:
                    data[f] = v

    cov_keys = ['COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN']
    missing = [k for k in cov_keys if data.get(k) is None]
    if missing:
        return {"error": f"Missing coverage fields: {missing}"}

    try:
        cov_values = [float(data[k]) for k in cov_keys]
    except Exception:
        return {"error": "Coverage fields must be numeric"}

    cov_sum = sum(cov_values)
    if abs(cov_sum - 1.0) > 1e-6:
        return {"error": "Coverage fractions must sum to 1", "cov_sum": cov_sum}

    # Build feature vector
    feat = []
    for f in fields:
        if f == 'DATE':
            raw = data.get('DATE')
            if raw is None:
                return {"error": "DATE is required"}
            # try ISO date YYYY-MM-DD
            try:
                dt = datetime.datetime.fromisoformat(raw)
                day = float(dt.timetuple().tm_yday)
                # day fraction in [0,1)
                day_frac = (day - 1.0) / 365.0
                sin_day = float(np.sin(2 * np.pi * day_frac))
                cos_day = float(np.cos(2 * np.pi * day_frac))
                # normalized year: 2000 -> 0, 2050 -> 1 (adjust scale if needed)
                year_norm = float((dt.year - 2000) / 50.0)
                # append TIME_DAY_SIN, TIME_DAY_COS, TIME_YEAR_NORM
                feat.extend([sin_day, cos_day, year_norm])
            except Exception:
                # fallback: if numeric provided, treat as day-of-year
                try:
                    day = float(raw)
                    day_frac = (day - 1.0) / 365.0
                    sin_day = float(np.sin(2 * np.pi * day_frac))
                    cos_day = float(np.cos(2 * np.pi * day_frac))
                    year_norm = 0.0
                    feat.extend([sin_day, cos_day, year_norm])
                except Exception:
                    return {"error": f"Unable to parse DATE: {raw}"}
        else:
            v = data.get(f)
            try:
                feat.append(float(v))
            except Exception:
                return {"error": f"Field {f} must be numeric (got {v})"}

    # normalization 
    scaler = StandardScaler()
    x = scaler.fit_transform(np.array(feat).reshape(1, -1))
    x = torch.tensor(x, dtype=torch.float32)

    model_path = ge_model_path(disaster)
    loaded = zuko.flows.NSF(
        features=2,
        context=21,
        transforms=8,
        hidden_features=[256, 256, 256],
        bins=16,
    )
    try:
        loaded.load_state_dict(torch.load(model_path, weights_only=True))
    except Exception as e:
        return {"error": f"Failed to load model at {model_path}: {e}"}

    # If the saved object is the model itself, use it. Otherwise try to find a model
    model = None
    if hasattr(loaded, 'eval') and callable(getattr(loaded, 'eval')):
        model = loaded
    else:
        raise ValueError(f"Loaded object does not appear to be a model: {loaded}")
    
    model.eval()
    try:
        with torch.no_grad():
            # sample n times to get empirical distribution
            n = 1000
            samples = []
            for _ in range(n):
                out = model(x)
                s = out.sample()  # shape (1,2)
                # convert to numpy 1D
                samples.append(s[0].cpu().numpy())
            samples = np.array(samples)  # shape (n,2)
    except Exception as e:
        return {"error": f"Model forward pass failed: {e}"}

    # inverse-transform samples if targets were trained with log1p
    try:
        samples_orig = np.expm1(samples)
    except Exception:
        print("Warning: inverse transform failed, returning raw samples")
        samples_orig = samples

    # compute mean and covariance of sampled points in original scale
    mean = samples_orig.mean(axis=0)
    cov = np.cov(samples_orig, rowvar=False)

    # compute 95% confidence ellipse parameters for 2D (chi-square quantile ~5.991)
    chi2_val = 5.991
    vals, vecs = np.linalg.eigh(cov)
    # sort eigenvalues descending
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    # angle in degrees for ellipse rotation
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # width and height (2*sqrt(eigval * chi2_val))
    width, height = 2 * np.sqrt(vals * chi2_val)

    # create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(samples_orig[:, 0], samples_orig[:, 1], s=8, alpha=0.25)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor='red', facecolor='none', lw=1)
    ax.add_patch(ellipse)
    ax.set_xlabel('damages')
    ax.set_ylabel('casualties')
    ax.set_title('Predictive samples and 95% mass ellipse')
    # set axis limits so both axes use the same data span
    x_min, x_max = samples_orig[:, 0].min(), samples_orig[:, 0].max()
    y_min, y_max = samples_orig[:, 1].min(), samples_orig[:, 1].max()
    span_x = x_max - x_min if x_max > x_min else 1.0
    span_y = y_max - y_min if y_max > y_min else 1.0
    span = max(span_x, span_y)
    # add padding ~10%
    pad_factor = 0.10
    half = 0.5 * span * (1.0 + pad_factor)
    ax.set_xlim(mean[0] - half, mean[0] + half)
    ax.set_ylim(mean[1] - half, mean[1] + half)
    ax.set_aspect('equal', 'box')

    # save to PNG buffer and return base64 data URI for embedding in HTML/JS
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    # optional debug output on disk
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    data_uri = f"data:image/png;base64,{img_b64}"

    return {
        'plot': data_uri,
        'mean': mean.tolist(),
        'cov': cov.tolist(),
        'n_samples': int(samples_orig.shape[0])
    }
    # return {"prediction": _tensor_to_py(out), "input": data}


    
# async def predict_one(request) -> Dict[str, Any]:
#     # expected feature names (same order used by the web form)
#     fields = [
#         'BEGIN_LAT','BEGIN_LON','DURATION_HOURS','WIND_SPEED','DATE','PRECIPITATION',
#         'TMIN','TMAX','ELEVATION','SLOPE',
#         'COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN',
#         'RIVER_DISTANCE','SEA_DISTANCE'
#     ]

#     # Accept either a dict-like payload or a Request
#     if isinstance(request, dict):
#         payload = request
#     else:
#         # Try to parse form first (the UI posts a FormData), otherwise JSON
#         try:
#             form = await request.form()
#             source = 'form' if form and len(form) > 0 else 'json'
#         except Exception:
#             form = None
#             source = 'json'

#         payload = {}
#         if source == 'form' and form is not None:
#             for f in fields + ['EVENT_TYPE']:
#                 v = form.get(f)
#                 payload[f] = v
#         else:
#             try:
#                 payload = await request.json()
#             except Exception:
#                 payload = {}

#     # EVENT_TYPE selects the model
#     disaster = payload.get('EVENT_TYPE') or payload.get('EVENT') or payload.get('event_type') or 'flood'

#     # Validate and convert numeric fields
#     data = {}
#     for f in fields:
#         v = payload.get(f)
#         if v is None:
#             data[f] = None
#         else:
#             if f == 'DATE':
#                 data[f] = v
#             else:
#                 try:
#                     data[f] = float(v)
#                 except Exception:
#                     data[f] = v

#     cov_keys = ['COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN']
#     missing = [k for k in cov_keys if data.get(k) is None]
#     if missing:
#         return {"error": f"Missing coverage fields: {missing}"}

#     try:
#         cov_values = [float(data[k]) for k in cov_keys]
#     except Exception:
#         return {"error": "Coverage fields must be numeric"}

#     cov_sum = sum(cov_values)
#     if abs(cov_sum - 1.0) > 1e-6:
#         return {"error": "Coverage fractions must sum to 1", "cov_sum": cov_sum}

#     # Build feature vector
#     feat = []
#     for f in fields:
#         if f == 'DATE':
#             raw = data.get('DATE')
#             if raw is None:
#                 return {"error": "DATE is required"}
#             # try ISO date YYYY-MM-DD
#             try:
#                 dt = datetime.datetime.fromisoformat(raw)
#                 day = float(dt.timetuple().tm_yday)
#                 # day fraction in [0,1)
#                 day_frac = (day - 1.0) / 365.0
#                 sin_day = float(np.sin(2 * np.pi * day_frac))
#                 cos_day = float(np.cos(2 * np.pi * day_frac))
#                 # normalized year: 2000 -> 0, 2050 -> 1 (adjust scale if needed)
#                 year_norm = float((dt.year - 2000) / 50.0)
#                 # append TIME_DAY_SIN, TIME_DAY_COS, TIME_YEAR_NORM
#                 feat.extend([sin_day, cos_day, year_norm])
#             except Exception:
#                 # fallback: if numeric provided, treat as day-of-year
#                 try:
#                     day = float(raw)
#                     day_frac = (day - 1.0) / 365.0
#                     sin_day = float(np.sin(2 * np.pi * day_frac))
#                     cos_day = float(np.cos(2 * np.pi * day_frac))
#                     year_norm = 0.0
#                     feat.extend([sin_day, cos_day, year_norm])
#                 except Exception:
#                     return {"error": f"Unable to parse DATE: {raw}"}
#         else:
#             v = data.get(f)
#             try:
#                 feat.append(float(v))
#             except Exception:
#                 return {"error": f"Field {f} must be numeric (got {v})"}

#     # normalization 
#     scaler = StandardScaler()
#     x = scaler.fit_transform(np.array(feat).reshape(1, -1))
#     x = torch.tensor(x, dtype=torch.float32)

#     model_path = ge_model_path(disaster)
#     loaded = zuko.flows.NSF(
#         features=2,
#         context=21,
#         transforms=8,
#         hidden_features=[256, 256, 256],
#         bins=16,
#     )
#     try:
#         loaded.load_state_dict(torch.load(model_path, weights_only=True))
#     except Exception as e:
#         return {"error": f"Failed to load model at {model_path}: {e}"}

#     # If the saved object is the model itself, use it. Otherwise try to find a model
#     model = None
#     if hasattr(loaded, 'eval') and callable(getattr(loaded, 'eval')):
#         model = loaded
#     else:
#         raise ValueError(f"Loaded object does not appear to be a model: {loaded}")
    
#     model.eval()
#     try:
#         with torch.no_grad():
#             # sample n times and average
#             n = 100
#             samples = []
#             for _ in range(n):
#                 out = model(x)
#                 samples.append(out.sample())
#             out = torch.stack(samples).mean(dim=0)
#     except Exception as e:
#         return {"error": f"Model forward pass failed: {e}"}
#     # inverse-transform if targets were trained with log1p
#     try:
#         out_inv = torch.expm1(out)
#     except Exception:
#         out_inv = out

#     return {'damages': out_inv[0,0].item(), 'casualties': out_inv[0,1].item()}
