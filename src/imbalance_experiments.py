import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
from src.preprocessing import preprocess_data

def run_advanced_experiments(train_path='data/fraudTrain.csv', test_path='data/fraudTest.csv', models_dir='models'):
  
    os.makedirs(models_dir, exist_ok=True)
    print("=== CHARGEMENT ET PRÉTRAITEMENT AVEC FEATURE ENGINEERING (SEMAINE 2) ===")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train, y_train, cat_encoder = preprocess_data(df_train, is_train=True, use_advanced_features=True)
    X_test, y_test = preprocess_data(df_test, is_train=False, category_encoder=cat_encoder, use_advanced_features=True)
    
    # Alignement des colonnes
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
    
    print(f"Nouvelle forme X_train: {X_train.shape}, X_test: {X_test.shape}")
    print("Liste des nouvelles variables clés :", [c for c in X_train.columns if c in ['velocity_kmh', 'amt_ratio_to_card_avg', 'time_since_last_trans_sec', 'trans_count_card']])
    
    # Modèle Random Forest Avancé
    print("\n[Entraînement] Random Forest avec Feature Engineering Avancé...")
    rf_advanced = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', n_jobs=-1, random_state=42)
    rf_advanced.fit(X_train, y_train)
    
    # Prédictions
    y_pred_proba = rf_advanced.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Recherche du Seuil Optimal
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
    prec_opt = precision_score(y_test, y_pred_opt, zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
    
    print("\n==================== RÉSULTATS SEMAINE 2 (Random Forest Avancé) ====================")
    print(f"ROC-AUC                    : {roc_auc:.4f}")
    print(f"PR-AUC (Métrique Prioritaire): {pr_auc:.4f} (Vs Baseline Semaine 1: 0.8398)")
    print(f"Seuil Optimal F1-Score     : {best_threshold:.4f}")
    print(f"Précision au Seuil Optimal  : {prec_opt:.4f}")
    print(f"Rappel (Recall)            : {rec_opt:.4f}")
    print(f"F1-Score Maximum           : {f1_opt:.4f}")
    print(f"Matrice de Confusion :\n{confusion_matrix(y_test, y_pred_opt)}")
    print("====================================================================================\n")
    
    # Sauvegarde du modèle avancé
    joblib.dump(rf_advanced, os.path.join(models_dir, 'rf_advanced_week2.joblib'))
    return pr_auc

if __name__ == '__main__':
    run_advanced_experiments()
