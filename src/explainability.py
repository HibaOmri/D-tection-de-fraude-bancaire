import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_data

def generate_shap_plots(model_path='models/best_model_sota.joblib', train_path='data/fraudTrain.csv', output_dir='reports/figures', n_samples=500):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        model_path = 'models/rf_advanced_week2.joblib'
        
    model = joblib.load(model_path)
    df_train_full = pd.read_csv(train_path)
    
    cat_encoder_path = 'models/category_encoder.joblib'
    cat_encoder = joblib.load(cat_encoder_path) if os.path.exists(cat_encoder_path) else None
    
    X_full, y_full, _ = preprocess_data(df_train_full, is_train=True, use_advanced_features=True)
    _, X_val, _, y_val = train_test_split(X_full, y_full, test_size=0.2, stratify=y_full, random_state=42)
    
    fraud_indices = y_val[y_val == 1].index[:n_samples//2]
    legit_indices = y_val[y_val == 0].index[:n_samples//2]
    sample_indices = fraud_indices.union(legit_indices)
    
    X_sample = X_val.loc[sample_indices]
    y_sample = y_val.loc[sample_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_class1.values, X_sample, show=False)
    plt.title("Importance globale des variables (SHAP Summary Plot)", fontsize=14, pad=15)
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    fraud_idx_in_sample = np.where(y_sample.values == 1)[0]
    if len(fraud_idx_in_sample) > 0:
        example_idx = fraud_idx_in_sample[0]
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_vals_class1[example_idx], show=False)
        plt.title(f"Explication détaillée d'une alerte Fraude (SHAP Waterfall)", fontsize=13, pad=15)
        plt.tight_layout()
        waterfall_path = os.path.join(output_dir, 'shap_waterfall_fraud_example.png')
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    generate_shap_plots()
