import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import add_advanced_features

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def preprocess_data(df, is_train=True, category_encoder=None, use_advanced_features=True):
    df = df.copy()
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    USD_TO_MAD = 10.0
    df['amt_mad'] = df['amt'] * USD_TO_MAD
    df['amt_mad_log'] = np.log1p(df['amt_mad'])
    
    if use_advanced_features:
        df = add_advanced_features(df)
    
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    if is_train:
        category_counts = df['category'].value_counts(normalize=True).to_dict()
    else:
        category_counts = category_encoder
        
    df['category_freq'] = df['category'].map(category_counts).fillna(0)
    
    category_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
    df = pd.concat([df, category_dummies], axis=1)
    
    features_to_drop = [
        'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
        'job', 'dob', 'trans_num', 'unix_time', 'lat', 'long', 'merch_lat', 'merch_long'
    ]
    
    target = None
    if 'is_fraud' in df.columns:
        target = df['is_fraud']
        features_to_drop.append('is_fraud')
        
    X = df.drop(columns=[c for c in features_to_drop if c in df.columns], errors='ignore')
    
    if is_train:
        return X, target, category_counts
    else:
        return X, target

if __name__ == '__main__':
    sample_df = pd.read_csv('data/fraudTrain.csv', nrows=1000)
    X, y, cat_enc = preprocess_data(sample_df, is_train=True)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Features préparées:", X.columns.tolist()[:10])
