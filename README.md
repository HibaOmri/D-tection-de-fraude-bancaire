# BP-RiskGuard : Détection de Fraude Bancaire par Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20API-orange.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20CatBoost-green.svg)
![SHAP](https://img.shields.io/badge/XAI-SHAP%20Explainability-purple.svg)
![Banque Populaire](https://img.shields.io/badge/Banque%20Populaire-Risk%20Analytics-red.svg)

**BP-RiskGuard** est une plateforme end-to-end de détection de fraude sur cartes bancaires, développée dans le cadre d'un stage au Pôle Gestion des Risques de la **Banque Populaire du Maroc**.

L'application intègre l'ensemble de la chaîne ML : de l'ingestion de données déséquilibrées à l'optimisation par matrice de coût financier (en MAD), l'explicabilité des alertes via SHAP et un dashboard d'opérations en temps réel.

---

## Fonctionnalités Clés

- **Analyse Exploratoire (EDA)** : Visualisation du déséquilibre de classe, distribution log des montants (MAD) et analyse temporelle/catégorielle des fraudes.
- **Feature Engineering Avancé** :
  - Vitesse géodésique de transaction (`velocity_kmh`) entre deux achats consécutifs par carte.
  - Distances Haversine client-commerçant et domicile-commerçant.
  - Encodage cyclique trigonométrique (`hour_sin`, `hour_cos`).
  - Ratios et écarts par rapport au profil historique du porteur de carte.
- **Modélisation SOTA** : Comparatif entre **Random Forest**, **XGBoost**, **LightGBM** et **CatBoost**.
- **Matrice de Coûts Financiers (MAD)** : Recherche automatique du seuil de décision optimal minimisant la perte totale ($\text{FN}_{\text{montant}} + \text{FP} \times 150\text{ DH}$).
- **Explicabilité SHAP (XAI)** : SHAP Summary Plot pour l'importance globale et Waterfall Plot pour l'explication locale par transaction.
- **Dashboard Web & API Flask** : Interface temps réel pour analystes risques avec simulateur de flux, réglage dynamique de sensibilité, clôture de dossiers (Faux Positif, OTP, Blocage carte) et journal d'audit rétroactif.

---

## Structure du Projet

```
detection_fraude/
├── data/                       # Données & Journal d'audit CSV (journal_audit.csv)
├── models/                     # Modèles ML sérialisés
├── notebooks/                  # Notebooks Jupyter d'expérimentation
├── reports/                    # Graphiques EDA & SHAP générés (figures/)
├── src/                        # Code Source Principal
│   ├── app.py                  # API Backend Flask & Serveur d'opérations
│   ├── baseline_model.py       # Modèles de référence (Régression Logistique, RF)
│   ├── eda.py                  # Génération des analyses graphiques
│   ├── explainability.py       # Calcul des SHAP values et graphiques XAI
│   ├── feature_engineering.py  # Calculs géodésiques, vitesse et ratios
│   ├── imbalance_experiments.py# Expérimentations déséquilibre
│   ├── models_sota.py          # Entraînement SOTA & Optimisation Financière
│   └── preprocessing.py        # Nettoyage et encodages des données
├── templates/                  # Interfaces HTML (index.html, login.html)
├── static/                     # CSS, JS et ressources de style
├── .gitignore                  # Exclusion des fichiers volumineux (>100 Mo)
└── README.md                   # Documentation du projet
```

---

## Installation & Démarrage

### 1. Prérequis & Installation
```bash
# Cloner le dépôt
git clone https://github.com/HibaOmri/D-tection-de-fraude-bancaire.git
cd D-tection-de-fraude-bancaire

# Installer les dépendances
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap flask matplotlib seaborn joblib
```

### 2. Lancer l'application Web
```bash
python src/app.py
```
Accédez au tableau de bord via votre navigateur sur `http://127.0.0.1:5000/`.

---

## Auteur
Développé dans le cadre d'un stage au Pôle Gestion des Risques de la **Banque Populaire du Maroc**.
