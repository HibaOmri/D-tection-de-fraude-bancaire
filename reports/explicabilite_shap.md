# Explicabilité du Modèle (Explainable AI - XAI avec SHAP)

Ce document fournit l'interprétation détaillée du modèle final **XGBoost** à partir des valeurs **SHAP (SHapley Additive exPlanations)** pour la rédaction du rapport de stage et la préparation des diapositives de soutenance.

---

## 1. Pourquoi l'Explicabilité (XAI) est-elle Indispensable en Banque ?

Dans le secteur bancaire et financier, un modèle de Machine Learning ne peut pas être une simple "boîte noire". Pour chaque transaction bloquée, la banque doit être en mesure :
1. D'expliquer au porteur de carte la raison du refus de paiement (transparence client).
2. De justifier la conformité réglementaire de ses algorithmes devant les régulateurs bancaires.
3. De permettre aux analystes du pôle Fraude d'analyser rapidement la légitimité d'une alerte.

---

## 2. Analyse Globale des Caractéristiques (SHAP Summary Plot)

Le graphique **SHAP Summary Plot** identifie les variables les plus discriminantes et leur sens d'impact sur le risque de fraude :

![SHAP Summary Plot](file:///c:/Users/Hiba/Desktop/4EME_HIBA/detection_fraude/reports/figures/shap_summary_plot.png)

### Les 4 Variables Clés Déterminantes :

1. **`velocity_kmh` (Vitesse géodésique de transaction en km/h)** :
   * **Impact** : C'est le facteur n°1 de détection. Une valeur élevée (point rouge) pousse fortement la probabilité vers le domaine de la fraude.
   * **Interprétation Métier** : Une transaction réalisée dans une ville distante trop rapidement après un achat précédent signale une usurpation physique ou un clonage de carte.

2. **`amt_mad` & `amt_diff_from_card_avg` (Montant en Dirhams)** :
   * **Impact** : Plus le montant de l'achat s'écarte à la hausse de la moyenne habituelle du porteur de carte, plus la contribution SHAP au risque de fraude est importante.
   * **Interprétation Métier** : Les fraudeurs cherchent souvent à maximiser leur gain avant que la carte ne soit bloquée.

3. **`dist_home_to_merch_km` (Distance Domicile - Commerçant)** :
   * **Impact** : Les transactions effectuées loin de la zone géographique de résidence du client augmentent l'indice de risque.

4. **`hour_sin` / `hour_cos` (Encodage Temporel)** :
   * **Impact** : Les transactions réalisées pendant la nuit (entre 0h et 4h du matin) affichent une pondération de risque accrue.

---

## 3. Explication Locale d'une Alerte Fraude (SHAP Waterfall Plot)

Le graphique **SHAP Waterfall Plot** illustre la décomposition exacte d'une prédiction pour une transaction frauduleuse spécifique :

![SHAP Waterfall Plot](file:///c:/Users/Hiba/Desktop/4EME_HIBA/detection_fraude/reports/figures/shap_waterfall_fraud_example.png)

### Lecture du Graphique pour la Soutenance :
* **Valeur de référence $E[f(X)]$** : Risque moyen de base (~0.58% de fraude globale).
* **Contributions positives (en Rouge)** :
  * `velocity_kmh = 420.5 km/h` ajoute **+0.45** au score de risque.
  * `amt_mad = 3,500.00 DH` ajoute **+0.28** au score de risque.
  * `dist_home_to_merch_km = 142.8 km` ajoute **+0.12** au score de risque.
* **Score Final $f(X)$** : **0.94** (soit 94% de probabilité de fraude), déclenchant le blocage automatique de la transaction par le système d'inférence bancaire.
