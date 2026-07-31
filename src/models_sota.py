import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import preprocess_data

def evaluate_with_cost_matrix(model, X_val, y_val, amt_mad_val, model_name="Modèle", cost_fp=150.0):
    start_time = time.time()
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    inference_time = time.time() - start_time
    
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    
    total_fraud_loss_baseline = amt_mad_val[y_val == 1].sum()
    
    best_savings = -float('inf')
    best_threshold = 0.5
    best_prec, best_rec, best_f1 = 0, 0, 0
    best_cm = None
    best_fn_cost, best_fp_cost, best_loss = 0, 0, 0
    
    threshold_grid = np.linspace(0.01, 0.99, 99)
    for th in threshold_grid:
        y_pred = (y_pred_proba >= th).astype(int)
        
        fn_mask = (y_val == 1) & (y_pred == 0)
        fn_cost = amt_mad_val[fn_mask].sum()
        
        fp_count = ((y_val == 0) & (y_pred == 1)).sum()
        fp_cost = fp_count * cost_fp
        
        total_loss = fn_cost + fp_cost
        savings = total_fraud_loss_baseline - total_loss
        
        if savings > best_savings:
            best_savings = savings
            best_threshold = th
            best_fn_cost = fn_cost
            best_fp_cost = fp_cost
            best_loss = total_loss
            best_prec = precision_score(y_val, y_pred, zero_division=0)
            best_rec = recall_score(y_val, y_pred, zero_division=0)
            best_f1 = f1_score(y_val, y_pred, zero_division=0)
            best_cm = confusion_matrix(y_val, y_pred)
            
    savings_pct = (best_savings / (total_fraud_loss_baseline + 1e-10)) * 100
    fp_count_best = int(best_fp_cost / cost_fp)
    
    print(f"\n==================== {model_name} ====================")
    print(f"ROC-AUC                    : {roc_auc:.4f}")
    print(f"PR-AUC                     : {pr_auc:.4f}")
    print(f"Seuil Optimal Financier     : {best_threshold:.4f}")
    print(f"Précision @ Seuil Optimal   : {best_prec:.4f}")
    print(f"Rappel (Recall)             : {best_rec:.4f}")
    print(f"F1-Score                    : {best_f1:.4f}")
    print(f"Matrice de Confusion :\n{best_cm}")
    print(f"Perte Fraude Sans Modèle   : {total_fraud_loss_baseline:,.2f} DH")
    print(f"Coût des Fraudes Manquées  : {best_fn_cost:,.2f} DH")
    print(f"Coût des Fausses Alertes   : {best_fp_cost:,.2f} DH ({fp_count_best} alertes * {cost_fp} DH)")
    print(f"Perte Totale Avec Modèle   : {best_loss:,.2f} DH")
    print(f"ÉCONOMIES NETTES BANQUE    : {best_savings:,.2f} DH (Gains: {savings_pct:.2f}%)")
    print(f"Temps d'inférence          : {inference_time:.3f} sec")
    print("========================================================\n")
    
    return {
        'model_name': model_name,
        'model_obj': model,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'precision': best_prec,
        'recall': best_rec,
        'f1_score': best_f1,
        'total_loss_mad': best_loss,
        'savings_mad': best_savings,
        'savings_pct': savings_pct
    }

def train_and_compare_sota_models(train_path='data/fraudTrain.csv', models_dir='models', val_size=0.2):
    os.makedirs(models_dir, exist_ok=True)
    df_train_full = pd.read_csv(train_path)
    
    if 'amt_mad' not in df_train_full.columns:
        df_train_full['amt_mad'] = df_train_full['amt'] * 10.0
    
    X_full, y_full, cat_encoder = preprocess_data(df_train_full, is_train=True, use_advanced_features=True)
    joblib.dump(cat_encoder, os.path.join(models_dir, 'category_encoder.joblib'))
    
    amt_mad_full = df_train_full['amt_mad'].values
    
    X_train, X_val, y_train, y_val, amt_train, amt_val = train_test_split(
        X_full, y_full, amt_mad_full, test_size=val_size, stratify=y_full, random_state=42
    )
    
    ratio_neg_pos = (y_train == 0).sum() / (y_train == 1).sum()
    results = []
    
    rf = RandomForestClassifier(
        n_estimators=120, max_depth=16, class_weight='balanced',
        max_features='sqrt', n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)
    res_rf = evaluate_with_cost_matrix(rf, X_val, y_val, amt_val, model_name="Random Forest")
    results.append(res_rf)
    joblib.dump(rf, os.path.join(models_dir, 'rf_advanced_week2.joblib'))
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=ratio_neg_pos,
        n_jobs=-1, random_state=42, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    res_xgb = evaluate_with_cost_matrix(xgb_model, X_val, y_val, amt_val, model_name="XGBoost")
    results.append(res_xgb)
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_sota.joblib'))
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=7, num_leaves=31, learning_rate=0.08,
        scale_pos_weight=15.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    res_lgb = evaluate_with_cost_matrix(lgb_model, X_val, y_val, amt_val, model_name="LightGBM")
    results.append(res_lgb)
    joblib.dump(lgb_model, os.path.join(models_dir, 'lightgbm_sota.joblib'))
    
    cat_model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.08,
        scale_pos_weight=ratio_neg_pos, verbose=0, random_seed=42
    )
    cat_model.fit(X_train, y_train)
    res_cat = evaluate_with_cost_matrix(cat_model, X_val, y_val, amt_val, model_name="CatBoost")
    results.append(res_cat)
    joblib.dump(cat_model, os.path.join(models_dir, 'catboost_sota.joblib'))
    
    df_res = pd.DataFrame([{
        'Modèle': r['model_name'],
        'PR-AUC': r['pr_auc'],
        'ROC-AUC': r['roc_auc'],
        'F1-Score': r['f1_score'],
        'Seuil Financier Optimal': r['best_threshold'],
        'Économies (MAD)': r['savings_mad'],
        'Gains (%)': r['savings_pct']
    } for r in results])
    
    print("\n==================== TABLEAU COMPARATIF FINAL ====================")
    print(df_res.to_string(index=False))
    print("==================================================================\n")
    
    best_model_info = max(results, key=lambda x: x['savings_mad'])
    joblib.dump(best_model_info['model_obj'], os.path.join(models_dir, 'best_model_sota.joblib'))
    
    return results

if __name__ == '__main__':
    train_and_compare_sota_models()
