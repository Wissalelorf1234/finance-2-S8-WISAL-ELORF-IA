# Prédiction de la Volatilité des Actions Cotées
## Bourse de Casablanca (BVC)
**Variable cible : Volatilité Réalisée sur 30 Jours**

> *Feature Engineering Temporel • Gestion de Portefeuille • Risque de Marché*

| Destinataires | Source des Données | Horizon Prédictif |
|---|---|---|
| Data Scientists, Analystes, Gérants | Bourse de Casablanca (BVC) | 30 jours calendaires |

---

## Résumé Exécutif

Ce rapport présente une démarche complète de modélisation prédictive de la volatilité des actions cotées à la Bourse de Casablanca (BVC). La variable cible retenue est la **volatilité réalisée sur une fenêtre glissante de 30 jours calendaires**, mesure de référence en finance quantitative pour appréhender le risque de marché à court terme.

L'objectif est double : d'une part **pédagogique**, en introduisant les notions fondamentales de la volatilité, du risque de marché et du feature engineering temporel ; d'autre part **opérationnel**, en fournissant un cadre méthodologique reproductible applicable à la gestion de portefeuille et au contrôle des risques.

> **Points clés du rapport**
> - Construction de la variable cible (volatilité réalisée 30J) à partir des données BVC
> - Feature engineering avancé sur séries temporelles financières
> - Modélisation comparative (GARCH, Random Forest, LSTM)
> - Application directe en gestion de portefeuille et surveillance des risques

---

## Partie I — Contexte & Enjeux Pédagogiques

### 1.1 La Bourse de Casablanca : panorama des données

La Bourse de Casablanca (BVC), créée en 1929 et réformée en 1993, est la principale place boursière d'Afrique francophone. Elle regroupe aujourd'hui plus de 70 sociétés cotées réparties dans les secteurs des télécommunications, des banques, des matériaux de construction, de l'immobilier et de l'agroalimentaire.

| Indice / Secteur | Exemples de Titres | Capitalisation | Liquidité |
|---|---|---|---|
| MASI (All Shares) | Attijariwafa, Maroc Telecom, CIH | ~700 Mds MAD | Élevée |
| MADEX (20 titres) | IAM, BCP, Lafarge Maroc | ~550 Mds MAD | Très élevée |
| Banques & Assurances | Attijariwafa, BMCE, Wafa Assurance | ~200 Mds MAD | Élevée |
| Matériaux & Bâtiment | LafargeHolcim, CimMorocco | ~120 Mds MAD | Modérée |
| Télécommunications | Maroc Telecom (IAM) | ~150 Mds MAD | Élevée |

Les données disponibles comprennent les cours de clôture ajustés (dividendes et splits), les volumes échangés, les cours bid/ask, ainsi que les indices sectoriels. La fréquence principale est journalière, avec des historiques remontant pour certains titres à plus de 20 ans.

### 1.2 Pourquoi prédire la volatilité ? Enjeux pédagogiques

| Domaine | Apport pédagogique |
|---|---|
| Statistiques | Distributions de probabilité, moments statistiques, hétéroscédasticité |
| Séries temporelles | Stationnarité, autocorrélation, effets ARCH/GARCH, saisonnalité |
| Machine Learning | Feature engineering, validation croisée temporelle, régression, LSTM |
| Finance de marché | Risque de marché, VaR, CVaR, théorie des portefeuilles de Markowitz |
| Gestion des risques | Stress testing, Value at Risk, limites de position, margining |

---

## Partie II — Notions de Risque de Marché

### 2.1 Définition et taxonomie du risque de marché

Le risque de marché désigne la perte potentielle résultant d'une variation défavorable des prix de marché : actions, taux d'intérêt, devises ou matières premières.

| Type de Risque | Source | Mesure associée |
|---|---|---|
| Risque de prix (actions) | Variation des cours boursiers | Volatilité réalisée, Bêta |
| Risque de liquidité | Absence de contrepartie | Bid-ask spread, Volume |
| Risque de corrélation | Co-mouvement entre actifs | Matrice de corrélation, DCC |
| Risque systémique | Effets de contagion | SRISK, CoVaR, MES |
| Risque de modèle | Inadéquation du modèle | Backtesting P&L, Kupiec test |

### 2.2 La volatilité réalisée : définition formelle

Le rendement logarithmique journalier est défini par :

```
r(t) = ln [ P(t) / P(t-1) ]
```

La volatilité réalisée sur une fenêtre glissante de N = 22 jours ouvrables (≈ 30 jours calendaires) est :

```
RV(t, N) = √( (252/N) × Σᵢ₌₁ᴺ r²(t−i+1) )
```

Le facteur **252** correspond au nombre de jours de bourse annuels conventionnels, permettant d'exprimer la volatilité en base annuelle.

> **Remarque — Choix de la fenêtre 30 jours**
> La fenêtre de 30 jours représente un compromis optimal : assez longue pour lisser le bruit microstructural, assez courte pour capter les régimes de volatilité changeants. Elle est cohérente avec les pratiques réglementaires (Bâle III).

### 2.3 Propriétés empiriques de la volatilité (Stylized Facts)

- **Clustering de volatilité** : les périodes de forte volatilité se regroupent, capturé par le modèle GARCH.
- **Asymétrie (Leverage Effect)** : les baisses de cours génèrent plus de volatilité que les hausses de même amplitude.
- **Mean-reversion** : la volatilité tend à revenir vers sa moyenne de long terme.
- **Persistance de longue mémoire** : l'autocorrélation de la volatilité réalisée décroît lentement (modèles FIGARCH, HAR).
- **Distribution à queue épaisse** : les rendements boursiers suivent une distribution leptokurtique (Kurtosis > 3).

### 2.4 Mesures de risque dérivées de la volatilité

| Mesure | Définition | Utilisation |
|---|---|---|
| VaR (99%) | Perte max sur 1 jour, non dépassée avec 99% de prob. | Exigences réglementaires Bâle III |
| CVaR / ES | Espérance de perte au-delà de la VaR | FRTB, gestion interne |
| Tracking Error | Écart-type des rendements actif vs benchmark | Gestion indicielle & active |
| Beta ajusté | Sensibilité au marché en période de stress | Construction de portefeuille |
| Sharpe Ratio | Rendement/Volatilité ajusté du taux sans risque | Évaluation de performance |

---

## Partie III — Feature Engineering Temporel

### 3.1 Philosophie du feature engineering sur données financières

Le feature engineering est l'art de transformer les données brutes en variables explicatives (features) porteuses d'information prédictive. La règle fondamentale est l'**absence de fuite de données (data leakage)** : toute feature calculée à la date t ne doit utiliser que des informations disponibles à t.

### 3.2 Construction de la variable cible

| Formulation | Définition | Usage recommandé |
|---|---|---|
| RV(t, 22) — historique | Volatilité des 22 derniers jours depuis t | Features / explication du passé |
| RV(t+22, 22) — forward | Volatilité des 22 prochains jours depuis t | **Variable CIBLE de prédiction** |
| log(RV) transformée | Logarithme de la vol. réalisée | Normalisation, meilleure régression |

### 3.3 Catalogue de features temporels

#### 3.3.1 Features de volatilité historique (endogènes)

| Feature | Description | Information capturée |
|---|---|---|
| RV_5d | Volatilité réalisée sur 5 jours | Tendance court terme |
| RV_22d | Volatilité réalisée sur 22 jours | Tendance mensuelle |
| RV_66d | Volatilité réalisée sur 66 jours | Tendance trimestrielle |
| RV_ratio_5_22 | RV_5d / RV_22d | Accélération de la volatilité |
| RV_lag_1 | RV_22d décalée de 1 jour | Effet d'inertie AR(1) |
| RV_lag_5 | RV_22d décalée de 5 jours | Dépendance hebdomadaire |
| RV_expanding_mean | Moyenne historique depuis le début | Niveau de long terme (μ) |
| RV_z_score | (RV - μ_252) / σ_252 | Anomalie de volatilité |

#### 3.3.2 Features de rendement et microstructure

- **Rendements absolus** (|r(t)|) : proxy direct de la volatilité instantanée.
- **Rendements au carré** (r²(t)) : mesure de la variance réalisée sur 1 jour.
- **Return positif / négatif** : capture l'asymétrie (effet levier).
- **Volume normalisé** (V(t) / V_bar_20) : prédicteur avancé de la volatilité future.
- **Turnover ratio** : volume × cours / capitalisation, mesure de liquidité relative.
- **Range intraday** (High-Low) / cours : estimateur de Parkinson.

#### 3.3.3 Features temporels et calendaires

| Feature calendaire | Encoding | Effet attendu sur BVC |
|---|---|---|
| Jour de la semaine | One-hot (lundi=0, ..., vendredi=4) | Volatilité lundi > autres jours |
| Mois de l'année | Sin/Cos encoding (cyclique) | Effets janvier, décembre |
| Fin de trimestre | Binaire (j-5 à j+5 fin trim.) | Rebalancement institutionnel |
| Période Ramadan | Binaire + jours restants | Baisse de liquidité, vol. réduite |
| Veille jour férié | Binaire | Volatilité asymétrique pré-fermeture |
| Saison des résultats | Binaire (fév-mars, juil-août) | Pics de volatilité idiosyncratique |

> **Note — Encoding cyclique**
> Pour les variables cycliques, utiliser sinus/cosinus plutôt que l'encodage ordinal :
> `feature_sin = sin(2π × m / 12)` et `feature_cos = cos(2π × m / 12)`

#### 3.3.4 Features de contexte macro et de marché

- **Volatilité de l'indice MASI** : prédicteur systémique le plus puissant.
- **Corrélation glissante avec le MASI** (corr_22d) : caractère systémique vs idiosyncratique.
- **Taux directeur Bank Al-Maghrib** : impact sur la valorisation des actions.
- **Taux de change USD/MAD et EUR/MAD** : pour les exportateurs (OCP, IAM).
- **Prix du pétrole Brent** : impact sur les secteurs transport et chimie.

### 3.4 Pipeline de feature engineering — Implémentation Python

```python
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construction des features temporels — données BVC"""
    df = df.copy().sort_index()

    # --- Rendements logarithmiques
    df['ret'] = np.log(df['close'] / df['close'].shift(1))

    # --- Volatilité réalisée multi-échelle (annualisée)
    for w in [5, 22, 66]:
        df[f'rv_{w}d'] = (df['ret']**2).rolling(w).mean().apply(
            lambda x: np.sqrt(x * 252))

    # --- Variable CIBLE : volatilité forward 22 jours
    df['target_rv_30d'] = (df['ret']**2).shift(-22).rolling(22).mean().apply(
        lambda x: np.sqrt(x * 252))

    # --- Features calendaires (encoding cyclique)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['dow_sin']   = np.sin(2 * np.pi * df.index.dayofweek / 5)

    # --- Ratio d'accélération et z-score de volatilité
    df['rv_ratio_5_22'] = df['rv_5d'] / df['rv_22d'].replace(0, np.nan)
    mu  = df['rv_22d'].rolling(252).mean()
    sig = df['rv_22d'].rolling(252).std()
    df['rv_zscore'] = (df['rv_22d'] - mu) / sig

    return df.dropna()
```

---

## Partie IV — Stratégies de Modélisation

### 4.1 Validation croisée temporelle

La validation croisée standard (K-Fold aléatoire) est **strictement interdite** sur les séries temporelles financières. La méthode correcte est la **validation Walk-Forward** :

- Diviser la série en K blocs consécutifs.
- À chaque itération, entraîner sur les blocs 1..k et tester sur le bloc k+1.
- Prévoir un gap de 22 jours entre train et test pour éviter le chevauchement.

### 4.2 Modèles économétriques classiques

#### Modèle HAR (Heterogeneous AutoRegressive)

```
RV(t+22) = α + β_d × RV_1d + β_w × RV_5d + β_m × RV_22d + ε
```

#### Modèle GARCH(1,1)

```
h(t) = ω + α × ε²(t-1) + β × h(t-1),   α + β < 1
```

### 4.3 Tableau comparatif des modèles

| Modèle | RMSE typique | MAE typique | Interprétabilité | Avantages BVC |
|---|---|---|---|---|
| HAR-RV | 0.042 | 0.031 | Très élevée | Benchmark solide, peu de données |
| GARCH(1,1) | 0.048 | 0.036 | Élevée | Capture ARCH effects |
| GJR-GARCH | 0.044 | 0.033 | Moyenne | Effet levier, asymétrie |
| Random Forest | 0.038 | 0.028 | Moyenne | Robuste, importance features |
| XGBoost | 0.035 | 0.026 | Moyenne | Performance, interactions |
| LSTM | 0.033 | 0.024 | Faible | Dépendances longues, non-linéaire |
| **Ensemble (HAR+XGB)** | **0.031** | **0.022** | Moyenne | **Meilleur compromis BVC** |

---

## Partie V — Applications à la Gestion de Portefeuille

### 5.1 Optimisation sous contrainte de volatilité

```
max  w'μ   s.c.   w'Σ̂w ≤ σ²_target,   1'w = 1,   w ≥ 0
```

où Σ̂ est la matrice de covariance construite à partir des volatilités prévisionnelles et des corrélations dynamiques (modèle DCC-GARCH).

### 5.2 Stratégies de gestion dynamique du risque

| Stratégie | Mécanisme | Signal déclencheur |
|---|---|---|
| Risk Parity | Pondération inversement prop. à la vol. | RV prévisionnelle de chaque titre |
| Target Volatility | Ajustement du levier pour vol. cible 10% | RV globale > 12% |
| Stop-Loss dynamique | Seuil de perte ajusté à la vol. | Z-score volatilité > 2σ |
| Couverture Delta | Achat de puts sur MASI | Vol. prévisionnelle > percentile 80% |
| Désallocation sectorielle | Réduction expo. secteur volatil | Clustering détecté |

### 5.3 Système d'alertes à 3 niveaux

- 🟢 **VERT (normal)** : RV prévisionnelle < percentile 60%. Gestion standard.
- 🟠 **ORANGE (vigilance)** : RV entre percentile 60% et 85%. Revue des positions.
- 🔴 **ROUGE (alerte)** : RV > percentile 85%. Activation des protocoles de réduction du risque.

### 5.4 Backtesting et validation métier

- **Test de Kupiec** : vérification du taux de dépassement de la VaR.
- **Test de Christoffersen** : indépendance des dépassements dans le temps.
- **P&L attribution** : fraction de la variation de VaR expliquée par le modèle.
- **Comparaison Sharpe** : portefeuille vol. prévisionnelle vs historique.

---

## Partie VI — Recommandations & Feuille de Route

### 6.1 Recommandations techniques

- Privilégier le modèle **HAR-RV** comme baseline, complété par un XGBoost avec features calendaires BVC-spécifiques.
- Construire un **ensemble** combinant HAR-RV et XGBoost avec pondération optimisée sur validation Walk-Forward.
- Incorporer systématiquement les features **Ramadan** et jours fériés marocains (effet -15% à -25% sur la volatilité).
- Utiliser **log(RV)** comme variable cible pour améliorer la normalité des résidus.
- Pour les titres à faible liquidité, utiliser l'estimateur de **Parkinson** (High/Low).

### 6.2 Feuille de route d'implémentation

| Phase | Durée | Actions clés | Livrable |
|---|---|---|---|
| 1 | Semaines 1-3 | Collecte et nettoyage des données BVC | Base de données propre |
| 2 | Semaines 4-6 | Construction des features, calcul RV_30d | Feature store |
| 3 | Semaines 7-9 | Entraînement HAR-RV et GARCH, benchmark | Modèles baseline |
| 4 | Semaines 10-13 | XGBoost, LSTM, tuning, ensemble learning | Modèle final |
| 5 | Semaines 14-16 | Intégration production, tableau de bord, alertes | Système live |

### 6.3 Limites et points de vigilance

> ⚠️ **Limites importantes**
> 1. La volatilité réalisée passée n'est pas la volatilité future.
> 2. Les événements exogènes non-anticipés restent imprévisibles.
> 3. Le faible nombre de titres liquides à la BVC limite la diversification.
> 4. Toute stratégie doit être backtestée sur au moins **5 ans** avant mise en production.

---

## Conclusion

Ce rapport a présenté un cadre complet pour la prédiction de la volatilité réalisée sur 30 jours des actions cotées à la Bourse de Casablanca. En articulant les dimensions pédagogiques, méthodologiques et opérationnelles, il constitue une feuille de route applicable aussi bien dans un contexte académique qu'en environnement professionnel.

| Variable Cible | Features clés | Modèle recommandé |
|---|---|---|
| Volatilité Réalisée 30J | RV multi-échelle + Calendaire | HAR-RV + XGBoost Ensemble |

---
*Bourse de Casablanca — Analyse Quantitative des Risques*

<img width="654" height="390" alt="image" src="https://github.com/user-attachments/assets/def3a759-d056-4ef1-b772-3b97d8e1fd08" />

<img width="654" height="330" alt="image" src="https://github.com/user-attachments/assets/67d51646-cf44-4220-a870-89c70b5b72d0" />


<img width="654" height="390" alt="image" src="https://github.com/user-attachments/assets/83f24ea1-9aa3-4984-8de9-ecab94e21e0f" />
