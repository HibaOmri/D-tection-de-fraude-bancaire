import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, precision_score, recall_score, f1_score
)
from src.preprocessing import preprocess_data

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    prec_def = precision_score(y_test, y_pred_default, zero_division=0)
    rec_def = recall_score(y_test, y_pred_default, zero_division=0)
    f1_def = f1_score(y_test, y_pred_default, zero_division=0)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
    prec_opt = precision_score(y_test, y_pred_opt, zero_division=0)
    rec_opt = recall_score(y_test, y_pred_opt, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
    
    print(f"\n==================== {model_name} ====================")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Precision: {prec_def:.4f} | Recall: {rec_def:.4f} | F1-Score: {f1_def:.4f}")
    print(f"Confusion Matrix (0.5):\n{confusion_matrix(y_test, y_pred_default)}")
    print(f"Precision: {prec_opt:.4f} | Recall: {rec_opt:.4f} | F1-Score: {f1_opt:.4f}")
    print(f"Confusion Matrix (Optimal):\n{confusion_matrix(y_test, y_pred_opt)}")
    print("========================================================\n")
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'precision_opt': prec_opt,
        'recall_opt': rec_opt,
        'f1_opt': f1_opt
    }

def train_and_evaluate_baselines(train_path='data/fraudTrain.csv', models_dir='models', val_size=0.2):
    os.makedirs(models_dir, exist_ok=True)
    df_train_full = pd.read_csv(train_path)
    
    X_full, y_full, cat_encoder = preprocess_data(df_train_full, is_train=True, use_advanced_features=True)
    joblib.dump(cat_encoder, os.path.join(models_dir, 'category_encoder.joblib'))
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=val_size, stratify=y_full, random_state=42
    )
    
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    res_lr = evaluate_model(lr, X_val, y_val, model_name="Logistic Regression (Baseline)")
    joblib.dump(lr, os.path.join(models_dir, 'baseline_logistic_regression.joblib'))
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    res_rf = evaluate_model(rf, X_val, y_val, model_name="Random Forest (Baseline)")
    joblib.dump(rf, os.path.join(models_dir, 'baseline_random_forest.joblib'))
    
    return [res_lr, res_rf]

if __name__ == '__main__':
    train_and_evaluate_baselines()
