import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcule la distance géodésique Haversine (en kilomètres) entre la position du client et celle du commerçant.
    """
    R = 6371.0 # Rayon de la Terre en kilomètres
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

from src.feature_engineering import add_advanced_features

def preprocess_data(df, is_train=True, category_encoder=None, use_advanced_features=True):
    """
    Nettoie et transforme le dataframe brut pour la modélisation.
    """
    df = df.copy()
    
    # 1. Feature Temporelles & Âge
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    # Âge du client au moment de la transaction
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    
    # 2. Feature Géographique : Distance Client - Commerçant
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    
    # 3. Transformations du Montant en Dirhams Marocains (1 USD = 10 DH)
    USD_TO_MAD = 10.0
    df['amt_mad'] = df['amt'] * USD_TO_MAD
    df['amt_mad_log'] = np.log1p(df['amt_mad'])
    
    # 4. Ajout des Variables Avancées (Vitesse, Ratios, Agrégations)
    if use_advanced_features:
        df = add_advanced_features(df)
    
    # 4. Encodage du Genre
    df['gender_encoded'] = (df['gender'] == 'M').astype(int)
    
    # 5. Encodage Fréquence pour la Catégorie
    if is_train:
        category_counts = df['category'].value_counts(normalize=True).to_dict()
    else:
        category_counts = category_encoder
        
    df['category_freq'] = df['category'].map(category_counts).fillna(0)
    
    # One-Hot Encoding pour les catégories principales
    category_dummies = pd.get_dummies(df['category'], prefix='cat', drop_first=True)
    df = pd.concat([df, category_dummies], axis=1)
    
    # 6. Colonnes à Conserver pour le Modèle
    features_to_drop = [
        'Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
        'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
        'job', 'dob', 'trans_num', 'unix_time', 'lat', 'long', 'merch_lat', 'merch_long'
    ]
    
    # Conserver target si présente
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
    print("Test du module preprocessing...")
    sample_df = pd.read_csv('data/fraudTrain.csv', nrows=1000)
    X, y, cat_enc = preprocess_data(sample_df, is_train=True)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Features préparées:", X.columns.tolist()[:10])
