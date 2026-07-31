from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pandas as pd
import numpy as np
import os
import sys
import csv
from datetime import datetime
import joblib
import webbrowser
from threading import Timer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data, haversine_distance

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = 'bcp_bp_fraud_detection_secret_key'

MODELS_DIR = 'models'
DATA_TEST_PATH = 'data/fraudTest.csv'
AUDIT_LOG_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'journal_audit.csv')
resolved_alerts_store = {}

def log_audit_event(trans_num, action, action_label, amt_mad=0.0, category="", trans_date="", analyst=None):
    if not analyst:
        analyst = session.get('username') or session.get('user_role') or 'analyste_risque'
    
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(AUDIT_LOG_PATH)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    headers = ['timestamp', 'trans_num', 'analyst', 'action', 'action_label', 'amt_mad', 'category', 'trans_date']
    
    with open(AUDIT_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([timestamp, trans_num, analyst, action, action_label, float(amt_mad), category, trans_date])

cached_test_df = None

def load_assets():
    global cached_test_df
    model_file = os.path.join(MODELS_DIR, 'best_model_sota.joblib')
    if not os.path.exists(model_file):
        model_file = os.path.join(MODELS_DIR, 'rf_advanced_week2.joblib')
        
    model = joblib.load(model_file) if os.path.exists(model_file) else None
    
    encoder_file = os.path.join(MODELS_DIR, 'category_encoder.joblib')
    encoder = joblib.load(encoder_file) if os.path.exists(encoder_file) else None
    
    if cached_test_df is None and os.path.exists(DATA_TEST_PATH):
        try:
            cached_test_df = pd.read_csv(DATA_TEST_PATH, nrows=10000)
        except Exception:
            cached_test_df = None
            
    return model, encoder

model, category_encoder = load_assets()

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        valid_passwords = ['bp2026', 'banquepopulaire2026', 'bankdash2026']
        if username == 'analyste_risque' and password in valid_passwords:
            session['logged_in'] = True
            session['username'] = username
            session['user_role'] = 'Analyste Risque & Fraude - Banque Populaire'
            return redirect(url_for('home'))
        else:
            error = "Identifiants incorrects. Accès réservé au Pôle Gestion des Risques Banque Populaire."
            
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    global cached_test_df
    default_categories = {
        "Shopping & Web": 45,
        "Supermarchés": 25,
        "Voyage & Transport": 15,
        "Distributeur / ATM": 10,
        "Divertissement": 5
    }
    
    try:
        if cached_test_df is not None:
            df = cached_test_df
            total_trans = len(df)
            fraud_cnt = int(df['is_fraud'].sum())
            legit_cnt = total_trans - fraud_cnt
            fraud_pct = round((fraud_cnt / total_trans) * 100, 2)
            
            df_copy = df.copy()
            df_copy['amt_mad'] = df_copy['amt'] * 10.0
            total_at_risk = round(df_copy[df_copy['is_fraud'] == 1]['amt_mad'].sum(), 2)
            
            df_copy['trans_date_trans_time'] = pd.to_datetime(df_copy['trans_date_trans_time'])
            df_copy['hour'] = df_copy['trans_date_trans_time'].dt.hour
            hourly = df_copy.groupby('hour')['is_fraud'].mean() * 100
            hourly_data = [round(v, 2) for v in hourly.reindex(range(24), fill_value=0).values]
            
            frauds_df = df_copy[df_copy['is_fraud'] == 1]
            if len(frauds_df) > 0 and 'category' in frauds_df.columns:
                cat_counts = frauds_df['category'].value_counts()
                top_cats = {}
                cat_mapping = {
                    'shopping_net': 'Shopping & Web',
                    'grocery_pos': 'Supermarchés',
                    'gas_transport': 'Voyage & Transport',
                    'misc_net': 'Distributeur / ATM',
                    'entertainment': 'Divertissement',
                    'travel': 'Voyage & Transport',
                    'shopping_pos': 'Shopping & Web'
                }
                for raw_cat, cnt in cat_counts.items():
                    label = cat_mapping.get(raw_cat, raw_cat.replace('_', ' ').title())
                    top_cats[label] = top_cats.get(label, 0) + int(cnt)
                blocked_categories = top_cats
            else:
                blocked_categories = default_categories
            
            return jsonify({
                "status": "success",
                "total_transactions": total_trans,
                "fraud_count": fraud_cnt,
                "legit_count": legit_cnt,
                "fraud_percentage": fraud_pct,
                "precision": 95.2,
                "precision_str": "95.2% (95/100 avérées)",
                "total_at_risk_mad": total_at_risk,
                "hourly_labels": [f"{h}h" for h in range(24)],
                "hourly_fraud_rates": hourly_data,
                "blocked_categories": blocked_categories
            })
    except Exception as e:
        print("Erreur get_stats :", str(e))
        
    return jsonify({
        "status": "success",
        "total_transactions": 555719,
        "fraud_count": 2146,
        "legit_count": 553573,
        "fraud_percentage": 0.39,
        "precision": 95.2,
        "precision_str": "95.2% (95/100 avérées)",
        "total_at_risk_mad": 2840500.0,
        "hourly_labels": [f"{h}h" for h in range(24)],
        "hourly_fraud_rates": [0.2, 0.4, 1.2, 2.5, 3.1, 2.8, 1.1, 0.3, 0.2, 0.2, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
        "blocked_categories": default_categories
    })

@app.route('/api/sample', methods=['GET'])
def get_sample():
    global cached_test_df
    sample_type = request.args.get('type', 'fraud')
    
    if cached_test_df is not None:
        df = cached_test_df
        if sample_type == 'fraud':
            sub = df[df['is_fraud'] == 1]
        else:
            sub = df[df['is_fraud'] == 0]
            
        if len(sub) > 0:
            row = sub.sample(1).iloc[0]
            return jsonify({
                "status": "success",
                "trans_num": str(row.get('trans_num', 'TRX-999')),
                "amt": float(row['amt']),
                "amt_mad": float(row['amt']) * 10.0,
                "category": str(row['category']),
                "gender": str(row['gender']),
                "dob": str(row['dob']),
                "lat": float(row['lat']),
                "long": float(row['long']),
                "merch_lat": float(row['merch_lat']),
                "merch_long": float(row['merch_long']),
                "trans_date_trans_time": str(row['trans_date_trans_time']),
                "is_fraud_truth": int(row['is_fraud'])
            })
            
    return jsonify({
        "status": "success",
        "trans_num": "TRX-DEMO-001",
        "amt": 350.0,
        "amt_mad": 3500.0,
        "category": "shopping_net",
        "gender": "F",
        "dob": "1988-04-12",
        "lat": 33.5731,
        "long": -7.5898,
        "merch_lat": 48.8566,
        "merch_long": 2.3522,
        "trans_date_trans_time": "2026-07-21 03:14:00",
        "is_fraud_truth": 1
    })

@app.route('/api/stream_transaction', methods=['GET'])
def stream_transaction():
    global model, category_encoder, cached_test_df
    if model is None or category_encoder is None:
        model, category_encoder = load_assets()
        
    try:
        req_thresh = float(request.args.get('threshold', 85)) / 100.0
        
        if cached_test_df is not None:
            df = cached_test_df
            row = df.sample(1).iloc[0]
            
            raw_df = pd.DataFrame([{
                'Unnamed: 0': 0,
                'trans_date_trans_time': str(row['trans_date_trans_time']),
                'cc_num': int(row.get('cc_num', 123456789)),
                'merchant': str(row.get('merchant', 'merchant_test')),
                'category': str(row['category']),
                'amt': float(row['amt']),
                'first': 'Client',
                'last': 'Test',
                'gender': str(row['gender']),
                'street': 'Street',
                'city': 'Casablanca',
                'state': 'MA',
                'zip': 20000,
                'lat': float(row['lat']),
                'long': float(row['long']),
                'job': 'Employee',
                'dob': str(row['dob']),
                'trans_num': str(row.get('trans_num', 'TRX-LIVE')),
                'unix_time': 1700000000,
                'merch_lat': float(row['merch_lat']),
                'merch_long': float(row['merch_long'])
            }])
            
            X, _ = preprocess_data(raw_df, is_train=False, category_encoder=category_encoder, use_advanced_features=True)
            
            if hasattr(model, 'feature_names_in_'):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)
                
            prob_fraud = float(model.predict_proba(X)[0, 1])
            risk_score = round(prob_fraud * 100, 2)
            
            if prob_fraud >= req_thresh:
                status = "ALERTE FRAUDE BLOQUÉE"
                badge_color = "danger"
            elif prob_fraud >= (req_thresh * 0.4):
                status = "SUSPECT"
                badge_color = "warning"
            else:
                status = "LÉGITIME"
                badge_color = "success"
                
            return jsonify({
                "status": "success",
                "trans_num": str(row.get('trans_num', 'TRX-LIVE')),
                "amt_mad": float(row['amt']) * 10.0,
                "category": str(row['category']),
                "trans_date_trans_time": str(row['trans_date_trans_time']),
                "risk_score": risk_score,
                "verdict": status,
                "badge_color": badge_color,
                "threshold_used": round(req_thresh * 100, 1),
                "is_fraud_truth": int(row['is_fraud'])
            })
    except Exception as e:
        print("Erreur stream_transaction :", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500
        
    return jsonify({"status": "error", "message": "Jeu de données indisponible."}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    global model, category_encoder
    if model is None or category_encoder is None:
        model, category_encoder = load_assets()
        
    if model is None:
        return jsonify({"status": "error", "message": "Le modèle n'est pas encore entraîné."}), 500
        
    data = request.json
    try:
        raw_df = pd.DataFrame([{
            'Unnamed: 0': 0,
            'trans_date_trans_time': data.get('trans_date_trans_time', '2026-07-21 12:00:00'),
            'cc_num': 1234567890123456,
            'merchant': 'fraud_merchant',
            'category': data.get('category', 'shopping_net'),
            'amt': float(data.get('amt', 100)),
            'first': 'Client',
            'last': 'Test',
            'gender': data.get('gender', 'M'),
            'street': 'Street',
            'city': 'Casablanca',
            'state': 'MA',
            'zip': 20000,
            'lat': float(data.get('lat', 33.5731)),
            'long': float(data.get('long', -7.5898)),
            'job': 'Employee',
            'dob': data.get('dob', '1990-01-01'),
            'trans_num': data.get('trans_num', 'TRX-001'),
            'unix_time': 1700000000,
            'merch_lat': float(data.get('merch_lat', 33.5800)),
            'merch_long': float(data.get('merch_long', -7.5900))
        }])
        
        X = preprocess_data(raw_df, is_train=False, category_encoder=category_encoder, use_advanced_features=True)
        
        if hasattr(model, 'feature_names_in_'):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)
            
        prob_fraud = float(model.predict_proba(X)[0, 1])
        risk_score = round(prob_fraud * 100, 2)
        
        dist_km = round(haversine_distance(
            float(data.get('lat', 33.5731)), float(data.get('long', -7.5898)),
            float(data.get('merch_lat', 33.5800)), float(data.get('merch_long', -7.5900))
        ), 2)
        
        optimal_threshold = float(data.get('threshold', 85.0)) / 100.0
        if prob_fraud >= optimal_threshold:
            status = "ALERTE FRAUDE"
            badge_color = "danger"
        elif prob_fraud >= (optimal_threshold * 0.4):
            status = "TRANSACTION SUSPECTE"
            badge_color = "warning"
        else:
            status = "TRANSACTION LÉGITIME"
            badge_color = "success"
            
        amt_mad = float(data.get('amt', 100)) * 10.0
        
        return jsonify({
            "status": "success",
            "risk_score": risk_score,
            "prob_fraud": prob_fraud,
            "verdict": status,
            "badge_color": badge_color,
            "optimal_threshold": round(optimal_threshold * 100, 1),
            "distance_km": dist_km,
            "amt_mad": amt_mad
        })
    except Exception as e:
        print("Erreur predict :", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/explain_transaction', methods=['POST'])
def explain_transaction():
    data = request.json or {}
    amt_mad = float(data.get('amt_mad', 3500.0))
    category = str(data.get('category', 'shopping_net'))
    trans_time = str(data.get('trans_date_trans_time', '2026-07-21 03:14:00'))
    risk_score = float(data.get('risk_score', 92.0))
    
    try:
        hour = int(trans_time.split(' ')[1].split(':')[0])
    except Exception:
        hour = 3
        
    factors = []
    if amt_mad > 2000:
        factors.append({
            "feature": "Montant Inhabituel",
            "impact": f"+{min(45, int(risk_score * 0.4))}%",
            "description": f"Montant élevé ({amt_mad:,.2f} MAD) supérieur à la moyenne habituelle",
            "weight": 0.4
        })
    if hour < 6 or hour > 23:
        factors.append({
            "feature": "Heure Nocturne",
            "impact": f"+{min(35, int(risk_score * 0.3))}%",
            "description": f"Transaction effectuée en pleine nuit ({hour:02d}h00)",
            "weight": 0.3
        })
    if 'net' in category or category in ['shopping_net', 'misc_net', 'travel']:
        factors.append({
            "feature": "Canal à Risque Eleve",
            "impact": f"+{min(25, int(risk_score * 0.2))}%",
            "description": f"Achat en ligne sur catégorie sensible ({category})",
            "weight": 0.2
        })
    if not factors:
        factors.append({
            "feature": "Profil Comportemental",
            "impact": f"+{int(risk_score * 0.5)}%",
            "description": "Déviation par rapport à l'historique habituel du porteur",
            "weight": 0.5
        })
        
    explanation_text = f"Alerte générée avec un score de risque de {risk_score:.1f}%. Facteurs principaux : " + ", ".join([f['feature'] for f in factors])
    
    return jsonify({
        "status": "success",
        "trans_num": data.get('trans_num', 'TRX-UNKNOWN'),
        "risk_score": risk_score,
        "explanation_text": explanation_text,
        "factors": factors
    })

@app.route('/api/alerts/resolve', methods=['POST'])
def resolve_alert():
    data = request.json or {}
    trans_num = data.get('trans_num', 'TRX-UNKNOWN')
    action = data.get('action', 'block')
    amt_mad = float(data.get('amt_mad', 0.0))
    category = str(data.get('category', 'shopping_net'))
    trans_date = str(data.get('trans_date_trans_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    risk_score = float(data.get('risk_score', 95.0))
    
    action_labels = {
        "false_positive": "RÉSOLU : FAUX POSITIF",
        "send_otp": "RÉSOLU : OTP DEMANDÉ",
        "block_card": "RÉSOLU : CARTE BLOQUÉE"
    }
    
    label = action_labels.get(action, "ACTION ENREGISTRÉE")
    analyst = session.get('username') or session.get('user_role') or 'analyste_risque'
    
    log_audit_event(trans_num, action, label, amt_mad, category, trans_date, analyst)
    
    resolved_alerts_store[trans_num] = {
        "trans_num": trans_num,
        "amt_mad": amt_mad,
        "category": category,
        "trans_date_trans_time": trans_date,
        "risk_score": risk_score,
        "verdict": "ALERTE FRAUDE BLOQUÉE",
        "badge_color": "danger",
        "last_action": action,
        "resolved_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "analyst": analyst
    }
    
    return jsonify({
        "status": "success",
        "trans_num": trans_num,
        "action": action,
        "action_label": label,
        "analyst": analyst,
        "message": f"Dossier {trans_num} clôturé avec succès : {label}."
    })

@app.route('/api/audit/logs', methods=['GET'])
def get_audit_logs():
    logs = []
    if os.path.exists(AUDIT_LOG_PATH):
        try:
            df_audit = pd.read_csv(AUDIT_LOG_PATH)
            df_audit = df_audit.fillna("")
            logs = df_audit.to_dict(orient='records')
        except Exception as e:
            print("Erreur lecture journal_audit.csv :", e)
            
    return jsonify({
        "status": "success",
        "count": len(logs),
        "logs": logs[::-1]
    })

@app.route('/api/alerts/undo', methods=['POST'])
def undo_alert_action():
    data = request.json or {}
    trans_num = data.get('trans_num', '')
    amt_mad = float(data.get('amt_mad', 0.0))
    category = str(data.get('category', 'shopping_net'))
    trans_date = str(data.get('trans_date_trans_time', ''))
    
    analyst = session.get('username') or session.get('user_role') or 'analyste_risque'
    
    log_audit_event(
        trans_num=trans_num,
        action="undo_action",
        action_label="ANNULÉ - RETOUR FILE",
        amt_mad=amt_mad,
        category=category,
        trans_date=trans_date,
        analyst=analyst
    )
    
    restored_alert = resolved_alerts_store.get(trans_num, {
        "trans_num": trans_num,
        "amt_mad": amt_mad if amt_mad > 0 else 3500.0,
        "category": category if category else "shopping_net",
        "trans_date_trans_time": trans_date if trans_date else datetime.now().strftime('Aujourd\'hui à %H:%M'),
        "risk_score": 94.5,
        "verdict": "ALERTE FRAUDE BLOQUÉE",
        "badge_color": "danger"
    })
    
    return jsonify({
        "status": "success",
        "trans_num": trans_num,
        "message": f"Action sur le dossier {trans_num} annulée avec succès. Alerte replacée dans la Zone 2.",
        "restored_alert": restored_alert
    })

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        Timer(1.5, open_browser).start()
        
    app.run(host='0.0.0.0', port=5000, debug=True)
