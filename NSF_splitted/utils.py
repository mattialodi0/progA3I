import pandas as pd
import matplotlib.pyplot as plt
import torch

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


# 
def run_inference_and_plot(flow, test_loader, y_scaler, target_cols, device):
    """
    Run inference for the test set and plot results for damages and casualties
     Parameters
    ----------
    flow        : trained zuko NSF
    test_loader : iterable over the test set
    y_scaler    : y_scaler to plot target feature in its standard input's form
    target_cols : target columns to predict and plot
    device      : device on which we want to work

    Returns
    n plots     : n = len(target_cols)
    """

    flow.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # Get the distribution and sample from it
            dist = flow(x_batch)
            y_hat = dist.sample().squeeze(1)  # shape: [batch, 2]
            y_true.append(y_batch.cpu())
            y_pred.append(y_hat.cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    # Inverse scale
    y_true = y_scaler.inverse_transform(y_true)
    y_pred = y_scaler.inverse_transform(y_pred)
    # Plot for each target
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.5)
    plt.xlabel(f"True {target_cols[0]}")
    plt.ylabel(f"Predicted {target_cols[0]}")
    plt.title(f"True vs Predicted: {target_cols[0]}")
    plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], [y_true[:, 0].min(), y_true[:, 0].max()], 'r--')
    plt.grid()
    plt.subplot(1,2,2)
    plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.5)
    plt.xlabel(f"True {target_cols[1]}")
    plt.ylabel(f"Predicted {target_cols[1]}")
    plt.title(f"True vs Predicted: {target_cols[1]}")
    plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], [y_true[:, 1].min(), y_true[:, 1].max()], 'r--')
    plt.grid()
    plt.show()