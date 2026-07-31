# Tableau Comparatif des Modèles SOTA & Impact Financier (Rapport de Stage)

| Modèle | PR-AUC (Métrique Métier) | ROC-AUC | Rappel (Recall %) | F1-Score | Seuil Financier Optimal | Économies (MAD / DH) | Gains Financiers (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Sélectionné)** | **0.9474** | **0.9992** | **91.74%** | **0.8529** | **0.9100** | **902 372.90 DH** | **86.67%** |
| CatBoost | 0.9331 | 0.9991 | 91.61% | 0.8336 | 0.9300 | 878 745.60 DH | 84.40% |
| Random Forest | 0.9039 | 0.9982 | 85.34% | 0.8262 | 0.7200 | 828 174.80 DH | 79.54% |
| LightGBM | 0.5978 | 0.9768 | 88.14% | 0.7778 | 0.7500 | 822 597.50 DH | 79.01% |

---

### Résumé pour l'interprétation dans le rapport :
1. **XGBoost** s'impose comme le meilleur algorithme du projet avec un **PR-AUC de 0.9474** et **91.74% de rappel** sur les transactions frauduleuses.
2. Sur le plan financier, l'application du modèle XGBoost permet à la banque de réaliser **902 372.90 DH d'économies nettes** en évitant **86.67%** des pertes brutes par fraude sur la période de validation.
