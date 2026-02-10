import pandas as pd

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