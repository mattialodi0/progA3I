import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy.stats import qmc
from pathlib import Path

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

# Configuration
figsize=(9, 3)

map_state_abbrev = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO",
    "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC", "PUERTO RICO": "PR"
}

def load_datasets(files):
    """
    Loads multiple CSV files into a single pandas DataFrame.
    Handles both a single file path or a list of file paths.
    """
    # Handle single file or list of files
    if isinstance(files, (str, Path)):
        files = [files]
    
    df_list = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            raise
    # Concatenate the datasets
    df = pd.concat(df_list, ignore_index=True)
    return df


def state_df_rolling_ws(df, state, windows_size=7):
    """
    Extract and prepare the dataset for a single state with the additional feature:
    rain_window -> precipitation over a fixed windows_size
    """
    
    data = df[df['state'] == state].sort_values('date').copy()

    if data.empty:
        raise ValueError(f"No match for the state: {state}")

    # Feature B: Pioggia nella finestra temporale (Rolling Sum)
    data['rain_window'] = data['precipitation'].rolling(window=windows_size).sum()
    
    # Remove the NaN created by the rolling windows
    data = data.dropna(subset=['rain_window'])

    return data


class FloodCostModel:
    """Cost model for flood detection with temporal tolerance."""
    
    def __init__(self, c_alarm, c_missed, tolerance=1):
        self.c_alarm = c_alarm
        self.c_missed = c_missed
        self.tolerance = tolerance

    def cost(self, y_scores, y_true, thr):
        # Obtain errors using temporal tolerance
        fp, fn = get_errors(y_scores, y_true, thr, self.tolerance)
        
        # Compute the cost
        return self.c_alarm * len(fp) + self.c_missed * len(fn)


def opt_threshold(y_scores, y_true, th_range, cost_model):
    """Find optimal threshold that minimizes cost."""
    costs = [cost_model.cost(y_scores, y_true, th) for th in th_range]
    best_th = th_range[np.argmin(costs)]
    best_cost = np.min(costs)
    return best_th, best_cost


def get_errors(y_scores, y_true, thr, tolerance=1):
    """Get false positives and false negatives with temporal tolerance."""
    # Predictions: where score < threshold
    pred = np.where(y_scores < thr)[0]
    # True events: where y_true == 1
    anomalies = np.where(y_true == 1)[0]
    
    # Initialize false positives and false negatives
    fp = set(pred)
    fn = set(anomalies)
    
    # Apply temporal tolerance
    for lag in range(-tolerance, tolerance + 1):
        fp = fp - set(anomalies + lag)
        fn = fn - set(pred + lag)
    
    return fp, fn


def plot_state_series(state, df):
    "Plot the preciptation as a timeseries with the scatter plot of the flood events"
    subset = df[df['state'] == state].sort_values('date')
    
    if subset.empty:
        print(f"No data available for the state: {state}")
        return 0

    plt.figure(figsize=(15, 6))
    
    plt.bar(subset['date'], subset['precipitation'], color='skyblue', label='Precipitation', width=2.0)
    
    floods_subset = subset[subset['is_flood'] == 1]
    
    plt.scatter(floods_subset['date'], floods_subset['precipitation'], 
                color='red', s=50, zorder=5, label='Flood event', marker='x')
    
    plt.title(f"Time Series precipitation and flood - state: {state}")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (in))")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
