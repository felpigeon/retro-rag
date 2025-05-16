# Retro RAG

## Description

Retro RAG est une application multi-service de Retrieval-Augmented Generation (RAG) dédiée aux articles de magazines sur les jeux rétro. Cette application permet d'interroger une base de connaissances d'articles, d'en extraire des informations pertinentes et de générer des réponses contextualisées à des questions.

![demo](./images/demo.gif)

## Architecture

Le projet est divisé en plusieurs services indépendants :

- **rag-app** : Application web en Java exposant des API REST pour l'interaction utilisateur
- **rag-backend** : Backend Python gérant l'extraction d'entités, les embeddings et le stockage des données
- **gpu-service** : Service Python dédié aux opérations nécessitant un GPU (reclassement, résumé extractif, détection d'hallucinations)
- **lab** : Environnement d'expérimentation contenant des notebooks qui démontrent les améliorations de performance

## Démarrage

### Prérequis

- Docker et Docker Compose
- GPU avec CUDA (recommandé pour les performances optimales)
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) pour l'exécution GPU
- Clé API Google AI

### Configuration

1. Copiez le fichier d'exemple d'environnement :

```bash
cp .env.example .env
```

2. Ouvrez le fichier `.env` et ajoutez votre clé API Google AI :

```
GOOGLE_API_KEY=votre_clé_api_ici
```

### Installation

Clonez le dépôt et lancez l'application avec Docker Compose :

```bash
git clone https://github.com/username/retro-rag.git
cd retro-rag
# Configuration de l'environnement (voir section précédente)
docker compose up
```

Une fois les dockers en marche, les articles peuvent etre ingérés avec le notebook `lab/ingest_articles.ipynb`

## Services disponibles

Une fois l'application démarrée, les services suivants sont disponibles :

| Service | URL | Description |
|---------|-----|-------------|
| Application Web | http://localhost:8080 | Interface utilisateur et API REST Java |
| Backend API | http://localhost:5000 | API du backend Python |
| Documentation Backend | http://localhost:5000/apidocs/ | Documentation de l'API du backend |
| GPU Service | http://localhost:5001 | Service pour les opérations GPU |
| Documentation GPU Service | http://localhost:5001/apidocs/ | Documentation de l'API du service GPU |

## Fonctionnalités principales

- **Interrogation intelligente** : Posez des questions sur les jeux rétro et obtenez des réponses précises basées sur des articles pertinents
- **Ingestion de documents** : Ajoutez facilement de nouveaux articles pour enrichir la base de connaissances
- **Détection d'entités** : Extraction automatique des jeux, consoles et développeurs mentionnés dans les articles
- **Résumé extractif** : Résumés intelligents des articles longs adaptés aux requêtes spécifiques qui remplace le *chunking* des documents.
- **Détection d'hallucinations** : Vérification des réponses générées pour garantir leur fiabilité

## Structure du projet

```
retro-rag/
├── rag-app/               # Application web Java
├── rag-backend/           # Backend Python pour le RAG
├── gpu-service/           # Service Python pour les opérations GPU
├── lab/                   # Notebooks d'expérimentation et évaluation
├── docker-compose.yml     # Configuration Docker Compose
└── README.md              # Ce fichier
```

## Composants détaillés

Pour plus de détails sur chaque composant, consultez les README spécifiques :

- [rag-app](./rag-app/README.md) - Application web Java
- [rag-backend](./rag-backend/README.md) - Backend Python
- [gpu-service](./gpu-service/README.md) - Service GPU Python
- [lab](./lab/README.md) - Expérimentations et évaluations

## Laboratoire d'expériences

Le dossier `lab` contient des notebooks et des résultats d'expériences démontrant les améliorations apportées au système RAG, notamment :

- Comparaison de différentes méthodes de recherche (sémantique, BM25, hybride)
- Impact du filtrage par entités
- Évaluation du reclassement (reranking) des documents
- Tests avec différents jeux de données synthétiques

