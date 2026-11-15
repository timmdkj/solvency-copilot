# Solvency Metrics Copilot

Prototype illustratif : un assistant qui **explique les variations de RWA**, **repère les contreparties atypiques**, **répond sur les métriques** et **explique les termes**, avec la preuve à côté de chaque chiffre.

> **Données entièrement synthétiques. Modèle volontairement simple : RWA = exposition × pondération selon la notation.** Ce projet ne reproduit aucun calcul réglementaire officiel (CRR3, Bâle III, FRTB). Il montre une méthode d'analyse, de contrôle et d'explication autour de ces métriques.

## Le problème traité

À chaque clôture mensuelle, une équipe de métriques de solvabilité doit expliquer pourquoi les RWA ont varié, vérifier qu'aucune donnée n'est aberrante, et répondre à des questions récurrentes sur les chiffres. Ce travail est souvent manuel et fait dans l'urgence. Ce prototype montre comment l'automatiser **sans jamais laisser un modèle de langage inventer un chiffre**.

## Ce que fait l'application

| Fonction | Contenu |
|---|---|
| **Explication des variations** | Décomposition du RWA entre deux clôtures mensuelles (M-1 = 30/09/2026, M = 31/10/2026) en 5 effets : périmètre, nouvelles opérations, sorties, variation d'encours, changement de notation, plus un résiduel. Contrôle de réconciliation. Principaux contributeurs avec explication. |
| **Vue type Pilier 3 (CR8)** | Les effets sont regroupés comme dans un tableau de flux de RWA publié : Taille des actifs, Qualité des actifs, Acquisitions et cessions, Autres. |
| **Anomalies sur le ratio de levier** | Détection des variations mensuelles aberrantes sur 24 clôtures (z-score robuste). Pour la variation du mois, indique si elle est dans la norme, son **origine par périmètre et par contrepartie** (en points de base) et si une analyse existe déjà. |
| **Anomalies sur les contreparties** | Isolation Forest sur les variations depuis M-1, avec une raison lisible par alerte. |
| **Questions sur les métriques** | Un assistant choisit un outil (variation, anomalies, levier, vues SQL prédéfinies, définitions) et rédige la réponse. Hors périmètre, il refuse. |
| **Définitions** | Petit glossaire (effet de périmètre, notation, résiduel, EAD, ratio de levier, Pilier 3…). |
| **Mémoire des analyses** | Chaque réponse indique si une analyse a déjà été réalisée (historique fictif dans `data/history.json`). |

## Correspondance avec le Pilier 3

| Effet du prototype | Catégorie type Pilier 3 (CR8) |
|---|---|
| Nouvelles opérations, sorties, variation d'encours | Taille des actifs |
| Changement de notation | Qualité des actifs |
| Effet de périmètre | Acquisitions et cessions |
| Résiduel (ajustement manuel) | Autres |

Le change n'est volontairement pas modélisé. Les noms de lignes exacts du tableau officiel n'ont pas pu être revérifiés ligne à ligne : la correspondance est indicative.

## Principes de conception

1. **Le calcul est déterministe** (Python, SQL). Le modèle de langage ne calcule rien.
2. Le modèle **choisit un outil et rédige** à partir des chiffres retournés.
3. **Chaque nombre cité est recontrôlé** contre les résultats des outils ; sinon, un avertissement s'affiche.
4. Les requêtes SQL sont **prédéfinies, en liste blanche et en lecture seule**. La requête exécutée est toujours affichée.
5. Sans outil approprié, l'assistant **refuse** plutôt que de répondre de mémoire.
6. Sans clé API, un **mode démo** utilise les mêmes outils et les mêmes chiffres, avec des gabarits de texte.

## Lancer en local

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 34 tests
streamlit run app.py
```

Sans configuration, l'application tourne en **mode démo**. Pour utiliser Mistral, créer le fichier **`.streamlit/secrets.toml`** (le fichier `.example` n'est qu'un modèle, il n'est jamais lu) :

```toml
MISTRAL_API_KEY = "ta-clé"
MODEL = "mistral-small-latest"
```

Ce fichier est ignoré par git. Le badge de la barre latérale indique « Mistral » ou « Mode démo ».

## Déployer (Streamlit Community Cloud)

1. Pousser le dossier sur GitHub (`secrets.toml` est déjà dans `.gitignore`).
2. Sur share.streamlit.io, créer une app pointant vers `app.py`.
3. Dans *Settings → Secrets*, coller les deux lignes ci-dessus avec la vraie clé.
4. Recommandé : fixer un plafond de dépense sur le compte API, et définir `APP_PASSWORD` pour protéger l'accès.

L'application limite à 10 messages par session le nombre d'appels au modèle ; au-delà, elle passe en mode démo.

## Structure

```
app.py                  interface Streamlit (Assistant, Variation du RWA, Alertes)
solvency/data.py        données synthétiques + événements injectés (vérité terrain)
solvency/engine.py      RWA illustratif, explication des variations, anomalies, SQL
solvency/glossary.py    glossaire des termes du prototype
solvency/llm.py         outils, appel Mistral, contrôle des chiffres, mode démo
data/history.json       historique fictif d'analyses
tests/                  réconciliation, événements retrouvés, anomalies, SQL, couche LLM
```

## Données et événements injectés

~1 200 contreparties sur deux clôtures mensuelles, deux périmètres (existant et acquis), 24 clôtures de séries. Le **ratio de levier est relié aux contreparties** : mesure d'exposition = 2,1 × EAD en périmètre (coefficient fictif) et Tier 1 = 15 % du RWA de M-1, supposé constant entre les deux clôtures. Une nouvelle opération fait donc monter le RWA et baisser le levier. Pondérations illustratives par notation (AA 15 %, A 30 %, BBB 55 %, BB 90 %, B 140 %, CCC 220 %).

**Ordres de grandeur** (calés « plus ou moins » sur une grande banque de financement et d'investissement, sans la représenter) : Tier 1 / RWA ≈ 14-15 %, RWA / mesure d'exposition ≈ 25 %, ratio de levier ≈ 3,5-3,7 %. Les pondérations, la répartition des notations et les tailles sont inventées.

Événements connus injectés, pour vérifier que le moteur les retrouve :

- une grosse nouvelle opération (Énergie, 600 M€) et quarante petites ; quarante sorties ;
- l'entrée dans le périmètre d'un portefeuille acquis (~5 % des contreparties) ;
- quatre dégradations et trois relèvements de notation d'un cran ;
- un ajustement manuel de 40 M€ (retrouvé comme résiduel) ;
- anomalies sur les contreparties : EAD multipliée par 4, souverain passé de AA à CCC ;
- anomalie sur le ratio de levier : pic de fin de trimestre au 31/03/2026 (déjà analysé dans l'historique fictif) ; la variation du mois (≈ −13 pb) est expliquée par le périmètre et les contreparties, mais reste dans la norme statistique.

Les tests (34) vérifient notamment que l'effet de périmètre égale le RWA du périmètre entrant, que le résiduel égale l'ajustement injecté, que les anomalies injectées sont retrouvées, que le levier est cohérent avec les expositions, et que les explications se réconcilient.

## Hypothèses et limites

- **Pondération par notation** : table illustrative, sans traitement par classe d'exposition, sans modèles internes (PD, LGD, maturité), sans devises. Un seul risque (crédit). Le ratio de levier est simplifié : mesure d'exposition proportionnelle à l'EAD crédit, sans dérivés ni opérations de financement modélisés. Pour le levier, l'allocation des points de base entre effets est proportionnelle à la variation d'exposition.
- **Ordre de décomposition** : l'effet croisé entre encours et notation dépend de l'ordre (encours d'abord, notation ensuite). L'ordre est choisi, documenté et fixe ; le résiduel ne contient ici que l'ajustement manuel.
- **Fréquence** : les publications Pilier 3 sont trimestrielles ; une production interne mensuelle est supposée ici (non vérifiée). Dans la réalité, le résiduel (« Autres ») peut être bien plus important et nécessite une explication narrative.
- **Anomalies** : signaux à vérifier, pas des erreurs confirmées. Le modèle se concentre sur les variations depuis M-1 ; les mouvements communs à tout un secteur sont neutralisés.
- **Isolation Forest** : paramétré avec `max_samples=1.0`, car les variations non nulles sont rares (sinon la plupart des arbres ne les voient pas).
- **Mode démo** : routage par mots-clés, plus rigide que le modèle de langage.
- Le contrôle des chiffres ignore les petits entiers et ne vérifie pas le sens des phrases, seulement les nombres.
- Les noms de modèles et l'API Mistral peuvent évoluer : `MODEL` est un simple réglage.

## Adapter à de vrais besoins

Remplacer `data.py` par un accès aux sources réelles (mapping des champs, droits d'accès, validation métier), étendre les périmètres (CCR, marché, grandes expositions, levier), brancher l'historique d'analyses sur les outils existants, et faire valider les définitions par l'équipe avant tout usage.

## Démo en 5 minutes

1. « Pourquoi le RWA a-t-il varié ce mois-ci ? » : cascade, vue type Pilier 3, contributeurs, réconciliation.
2. « Y a-t-il des contreparties atypiques ? » : alertes avec raisons.
3. « Quel est le ratio de levier ? » : variation du mois dans la norme, origine par périmètre et contrepartie, pic de mars déjà analysé.
4. « C'est quoi l'effet de périmètre ? » : définition du glossaire.
5. « Quelle est la VaR de la desk Actions ? » : refus propre.
6. Montrer la requête SQL et le contrôle des chiffres dans « Preuves ».
