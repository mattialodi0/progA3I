import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_disaster_dataset(data_dir):
    try:
        files = [
            data_dir / 'StormEvents_details-ftp_v1.0_d2016_c20250818.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2017_c20250520.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2018_c20250520.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2019_c20250520.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2020_c20251118.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2021_c20250520.csv',
            data_dir / 'StormEvents_details-ftp_v1.0_d2022_c20250721.csv',
        ]
        df_list = []
        for file in files:
            df = pd.read_csv(file)
            df_list.append(df)

        
        df = pd.concat(df_list, ignore_index=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None
    
    return df


def damage_to_numeric(damage_str):
    if pd.isna(damage_str):
        return 0
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    if damage_str[-1] in multipliers:
        try:
            value = float(damage_str[:-1])
            return value * multipliers[damage_str[-1]]
        except ValueError:
            return 0
    try:
        return float(damage_str)
    except ValueError:
        return 0  

disaster_events_group_map = {
    'Thunderstorm Wind': 'Wind',
    'Hail': 'Hail',
    'Marine Hail': 'Hail',
    'Flood': 'Flood',
    'Flash Flood': 'Flood',
    'Coastal Flood': 'Flood',
    'Lakeshore Flood': 'Flood',
    'Winter Weather': 'Freeze',
    'Extreme Cold/Wind Chill': 'Freeze',
    'Frost/Freeze': 'Freeze',
    'Cold/Wind Chill': 'Freeze',
    'Freezing Fog': 'Freeze',
    'Heat': 'Heat',
    'Excessive Heat': 'Heat',
    'Heavy Snow': 'Snow',
    'Blizzard': 'Snow',
    'Lake-Effect Snow': 'Snow',
    'Sleet': 'Snow',
    'Winter Storm': 'Storm',
    'Marine Thunderstorm Wind': 'Storm',
    'Tornado': 'Storm',
    'Tropical Storm': 'Storm',
    'Dust Storm': 'Storm',
    'Funnel Cloud': 'Storm',
    'Ice Storm': 'Storm',
    'Waterspout': 'Storm',
    'Marine Tropical Storm': 'Storm',
    'Hurricane (Typhoon)': 'Storm',
    'Marine Hurricane/Typhoon': 'Storm',
    'High Wind':  'Wind',
    'Strong Wind':  'Wind',
    'Marine High Wind':  'Wind',
    'Marine Strong Wind':  'Wind',
    'Dense Fog':  'Fog',
    'Marine Dense Fog':  'Fog'
}

def get_gaussian_lattice(device='cpu', n_rings=10, n_points=10):
    """
    Generates a fixed 101-point lattice for 2D flow visualization.
    
    Structure:
    - 1 Center point (0,0)
    - n Rings of k points each
    - Radii based on Gaussian probability intervals
    - Alternating phase shift (0, 2pi/k) between rings
    
    Returns:
        z (Tensor): [1+n*k, 2] inputs for the flow inverse.
        colors (Array): [1+n*k, 3] RGB values for plotting the output.
    """

    import torch
    import matplotlib.colors as mcolors

    points = []
    colors_hsv = []

    # --- 1. The Center (Point 0) ---
    points.append([0.0, 0.0])
    # Color: White (Saturation=0, Value=1)
    colors_hsv.append([0.0, 0.0, 1.0]) 

    # --- 2. The 10 Rings ---
    phase_shift = np.pi / n_points  # Alternating shift
    
    # Calculate radii corresponding to equidistant Gaussian probabilities
    # We slice probability space from 51% to 95%
    # This ensures we cover the 'meat' of the distribution and the tails
    probs = torch.linspace(0.50, 0.95, n_rings)
    dist = torch.distributions.Normal(0, 1)
    radii = dist.icdf(probs) # Transform prob -> gaussian radius (sigma)
    max_r = radii[-1].item()

    for i, r in enumerate(radii):
        r = r.item()
        
        # Alternating phase: Even rings start at 0, Odd rings start at pi/20
        # (i starts at 0, so ring 1 is index 0)
        current_phase = phase_shift if (i % 2 != 0) else 0.0
        
        # Generate 10 angles for this ring
        angles = torch.linspace(0, 2*np.pi, n_points + 1)[:-1] + current_phase
        
        # Polar -> Cartesian
        x = r * torch.cos(angles)
        y = r * torch.sin(angles)
        
        # Append points
        ring_points = torch.stack([x, y], dim=1)
        points.append(ring_points)
        
        # --- Color Logic (Polar -> HSV) ---
        # Hue = Angle (normalized 0-1)
        # Saturation = Radius (normalized 0-1)
        # Value = 1.0 (Max brightness)
        
        # Normalize angles to [0, 1] for Hue
        # We use modulo to strictly keep it in range
        hues = (angles % (2*np.pi)) / (2*np.pi)
        
        # Saturation increases with radius (White center -> Vivid edge)
        saturation = r / (max_r * 1.1) # Divide by slightly more than max to avoid clipping
        saturations = torch.full_like(hues, saturation)
        
        values = torch.ones_like(hues) # Full brightness
        
        # Stack HSV for this ring
        ring_hsv = torch.stack([hues, saturations, values], dim=1)
        colors_hsv.append(ring_hsv)

    # --- 3. Assemble Final Tensors ---
    # Concatenate all lists into single tensors
    z_tensor = torch.cat([torch.tensor(points[:1]), torch.cat(points[1:])], dim=0)
    c_tensor_hsv = torch.cat([torch.tensor(colors_hsv[:1]), torch.cat(colors_hsv[1:])], dim=0)
    
    # SAFETY: Clamp HSV values to [0, 1] to prevent matplotlib errors
    c_tensor_hsv = torch.clamp(c_tensor_hsv, 0, 1)
    colors_rgb = mcolors.hsv_to_rgb(c_tensor_hsv.cpu().numpy())
    
    return z_tensor.float().to(device), colors_rgb