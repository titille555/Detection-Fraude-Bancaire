#  Détection de Fraude Bancaire — Arbre de Décision C4.5

> *Peut-on apprendre à un algorithme à repérer un fraudeur... quand le fraudeur fait tout pour ressembler à un client normal ?*  
> C'est la question que ce projet tente de répondre.

---

##  Contexte

Ce projet s'inscrit dans le cadre d'un TP de Data Mining suivant la méthodologie **CRISP-DM**.  
L'objectif : construire un modèle de machine learning capable de **prédire automatiquement si une transaction bancaire est frauduleuse**, à partir d'un dataset de 51 000 transactions réelles.

| Élément | Détail |
|---|---|
|  Dataset | `Fraud.csv` — 51 000 transactions, 12 variables |
|  Variable cible | `Fraudulent` (0 = légitime, 1 = fraude) |
|  Algorithme principal | `DecisionTreeClassifier` (criterion=`entropy` → C4.5) |
|  Métriques | AUC-ROC, F1-score, Recall |

---

##  Structure du projet

```
 fraud-detection/
├──  notebook.ipynb       # Notebook principal (toutes les étapes)
├──  Fraud.csv            # Dataset
└──  README.md            # Ce fichier
```

---

##  Étapes réalisées

### 1️ EDA — Analyse Exploratoire

Avant de modéliser quoi que ce soit, il faut **comprendre ses données**.  
On a cherché à répondre à : *Qu'est-ce qui distingue une transaction frauduleuse d'une transaction normale ?*

- **Analyse des valeurs manquantes** — détection des NaN et des valeurs parasites (`"Invalid Method"`)
- **Distribution de la cible** — le dataset est très déséquilibré : seulement ~5% de fraudes. Ce point est crucial pour le choix des métriques.
- **Variables numériques** — histogrammes et boxplots stratifiés par classe → les moyennes sont quasi identiques entre fraudes et transactions légitimes. Pas de séparabilité évidente.
- **Variables catégorielles** — taux de fraude par modalité → Chicago, les achats en ligne et le mobile présentent un léger sur-risque.
- **Matrice de corrélation** — aucune corrélation linéaire significative → justifie l'usage d'un modèle non-linéaire.

>  **Ce qu'on a appris :** les fraudeurs imitent presque parfaitement les clients légitimes. Le signal est faible, fragmenté, non-linéaire.

---

### 2️ Feature Engineering

Puisque les variables brutes ne suffisent pas, on en a **créé de nouvelles** à partir du contexte métier :

| Nouvelle variable | Logique |
|---|---|
| `Is_High_Amount` | Transaction au-dessus du 75e percentile → montant suspect |
| `Is_High_Frequency` | Trop de transactions en 24h → comportement inhabituel |
| `Is_New_Account` | Compte de moins de 12 mois → profil à risque |
| `Has_Fraud_History` | A déjà fraudé → signal fort |
| `Is_Night_Transaction` | Transaction entre 0h et 6h → moment suspect |

On a aussi :
- Supprimé `Transaction_ID` et `User_ID` (identifiants sans valeur prédictive)
- Remplacé `"Invalid Method"` par `NaN` puis imputé
- Imputé les NaN (médiane pour le numérique, mode pour le catégoriel)
- Encodé les variables textuelles avec `LabelEncoder`

---

### 3️ Modélisation & Comparaison

Trois modèles ont été entraînés et comparés via la **courbe ROC** :

| Modèle | Pourquoi ce choix |
|---|---|
|  Decision Tree C4.5 | Modèle principal du TP — interprétable, produit des règles lisibles |
|  Régression Logistique | Baseline simple et rapide à interpréter |
|  Random Forest | Ensemble puissant pour comparer la performance |

Le **GridSearch avec validation croisée (5 folds)** a été utilisé pour optimiser les hyperparamètres du Decision Tree (`max_depth`, `min_samples_leaf`, `min_samples_split`).

---

##  Résultats

| Modèle | AUC-ROC |
|---|---|
| Decision Tree (tuned) | ~0.51 |
| Régression Logistique | ~0.51 |
| Random Forest | ~0.51 |

Tous les modèles plafonnent autour de **0.50** — ce qui équivaut à prédire au hasard.

> **Ce résultat n'est pas un échec de la méthodologie, c'est une conclusion en soi.**  
> Il démontre que le signal de fraude n'est tout simplement pas présent dans les variables disponibles. Les fraudeurs imitent parfaitement les comportements légitimes — les algorithmes, aussi puissants soient-ils, ne peuvent pas extraire ce qui n'existe pas dans les données.

---

##  Ce qu'on retient

- La **qualité des données** prime toujours sur la puissance du modèle
- Un AUC-ROC de 0.50 n'est pas une erreur de code — c'est un diagnostic
- Sur un dataset déséquilibré, **l'Accuracy est trompeuse** → toujours regarder le Recall et l'AUC
- Pour aller plus loin : données biométriques, adresses IP, ou techniques de sur-échantillonnage (SMOTE)

---

##  Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-lightblue)

```
pandas · numpy · matplotlib · seaborn
scikit-learn (DecisionTreeClassifier, RandomForestClassifier,
              LogisticRegression, GridSearchCV, LabelEncoder)
```

---

##  Lancer le notebook

```bash
# Cloner le repo
git clone https://github.com/ton-username/fraud-detection.git
cd fraud-detection

# Installer les dépendances
pip install pandas numpy matplotlib seaborn scikit-learn

# Ouvrir le notebook
jupyter notebook notebook.ipynb
```

> Le notebook peut aussi être exécuté directement sur **Google Colab** sans installation locale.

---

*Projet réalisé dans le cadre d'un TP de Data Mining — Méthodologie CRISP-DM*
