import pandas as pd
import numpy as np

def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def add_advanced_features(df):
    df = df.copy()
    
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df = df.sort_values(by=['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
        
        hours = df['trans_date_trans_time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
    
    prev_lat = df.groupby('cc_num')['lat'].shift(1)
    prev_long = df.groupby('cc_num')['long'].shift(1)
    prev_time = df.groupby('cc_num')['unix_time'].shift(1)
    
    dist_between_trans = haversine_distance_vectorized(df['lat'], df['long'], prev_lat, prev_long).fillna(0)
    time_diff_hours = (df['unix_time'] - prev_time) / 3600.0
    time_diff_hours = time_diff_hours.fillna(0)
    
    velocity_kmh = np.where(time_diff_hours > 0, dist_between_trans / time_diff_hours, 0)
    df['velocity_kmh'] = np.clip(velocity_kmh, 0, 1500)
    df['time_since_last_trans_sec'] = (df['unix_time'] - prev_time).fillna(999999)
    
    df['dist_home_to_merch_km'] = haversine_distance_vectorized(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    if 'amt_mad' in df.columns:
        card_mean_amt = df.groupby('cc_num')['amt_mad'].transform('mean')
        df['amt_ratio_to_card_avg'] = df['amt_mad'] / (card_mean_amt + 1e-5)
        df['amt_diff_from_card_avg'] = df['amt_mad'] - card_mean_amt
    
    df['trans_count_card'] = df.groupby('cc_num')['cc_num'].transform('count')
    
    return df

if __name__ == '__main__':
    sample_df = pd.read_csv('data/fraudTrain.csv', nrows=1000)
    sample_df['amt_mad'] = sample_df['amt'] * 10.0
    df_feat = add_advanced_features(sample_df)
    print("Variables générées :", [c for c in df_feat.columns if c in ['hour_sin', 'hour_cos', 'velocity_kmh', 'dist_home_to_merch_km', 'amt_ratio_to_card_avg', 'amt_diff_from_card_avg']])
