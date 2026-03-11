from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import inference

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@app.post("/*")
async def predict(request: Request):
	print('aaaaaaaaaaaaaaaaaaaaa')
	return JSONResponse({"message": "This is a placeholder response. Replace with actual prediction logic."})

@app.post("/predict")
async def predict(request: Request):
	"""Parse form values for the requested inputs and validate covariate sum.

	Accepts form-encoded fields (see template inputs) or a JSON body.
	Returns 400 if covariate fractions do not sum to 1.
	"""
	# Expected field names
	fields = [
		'BEGIN_LAT','BEGIN_LON','DURATION_HOURS','WIND_SPEED','DATE','PRECIPITATION',
		'TMIN','TMAX','ELEVATION','SLOPE',
		'COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN',
		'RIVER_DISTANCE','SEA_DISTANCE'
	]

	data = {}

	# Prefer form data
	try:
		form = await request.form()
		source = 'form'
	except Exception:
		form = None
		source = 'json'

	if source == 'form' and form is not None and len(form) > 0:
		for f in fields:
			v = form.get(f)
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
	else:
		try:
			payload = await request.json()
		except Exception:
			payload = {}
		for f in fields:
			v = payload.get(f)
			data[f] = v

	cov_keys = ['COV_BARREN','COV_CULTIVATED','COV_VEGETATION','COV_FOREST','COV_WATER','COV_SNOW_ICE','COV_URBAN']
	# Check all cov fields present and numeric
	missing = [k for k in cov_keys if data.get(k) is None]
	if missing:
		return JSONResponse({"error": f"Missing coverage fields: {missing}"}, status_code=400)

	try:
		cov_values = [float(data[k]) for k in cov_keys]
	except Exception:
		return JSONResponse({"error": "Coverage fields must be numeric"}, status_code=400)

	cov_sum = sum(cov_values)
	if abs(cov_sum - 1.0) > 1e-6:
		return JSONResponse({"error": "Coverage fractions must sum to 1", "cov_sum": cov_sum}, status_code=400)

	# Placeholder: replace with real model inference
	# return JSONResponse({"prediction": "dummy", "input": data})
	output = await inference.predict_one(data)
	return JSONResponse({"prediction": output, "input": data})	