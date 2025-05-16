# RAG Backend

## Description

Backend pour le système RAG (Retrieval-Augmented Generation). Il fournit des services pour l'ingestion de documents et un pipeline de questions-réponses.

## Fonctionnalités

- **Ingestion** : Extraction d'entités et de métadonnées, génération d'embeddings et stockage dans Qdrant. Supporte l'ingestion d'un document avec `/ingest` et l'ingestion par batch avec `/ingest_batch`
- **Pipeline QA** : Répond aux questions en utilisant les documents pertinents.

## Prérequis

- Python 3.12 ou supérieur
- Base de données DuckDB
- Serveur Qdrant

## Tests

Pour exécuter les tests unitaires :
```bash
uv run pytest
```

## Documentation

La documentation est générée automatiquement avec Sphynx.

```
cd docs
uv run make html
```

## Documentation API

La documentation interactive de l'API est disponible à l'adresse suivante :
```
http://localhost:5000/apidocs/
