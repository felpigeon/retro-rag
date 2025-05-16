# GPU Service

## Description

Les services nécessitant un GPU sont rassemblés dans ce Docker. Il comprend :

- **Reclassement** : Réorganise les documents en fonction de leur pertinence par rapport à une requête.
- **Résumé** : Génère un résumé (extractif) d'un document conditionné sur une requête. Ceci remplace le *chunking*.
- **Détection d'hallucinations** : Identifie les hallucinations dans une réponse donnée en fonction d'une requête et d'un contexte (optimsé pour l'anglais).

## Prérequis

- Python 3.12 ou supérieur
- GPU avec CUDA activé pour accélérer les calculs
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) pour exécuter les conteneurs Docker avec accès au GPU
- Le gestionnaire de dépendances **uv** .

## Endpoints API

- **Reclassement** : `POST /rerank`
  - Corps de la requête :
    ```json
    {
      "query": "votre_requête",
      "documents": [
        {"text": "texte_du_document_1"},
        {"text": "texte_du_document_2"}
      ]
    }
    ```

- **Résumé** : `POST /summarize`
  - Corps de la requête :
    ```json
    {
      "query": "votre_requête",
      "document": "texte_du_document",
      "length": 200
    }
    ```

- **Détection d'hallucinations** : `POST /detect_hallucination`
  - Corps de la requête :
    ```json
    {
      "query": "votre_requête",
      "context": "texte_du_contexte",
      "response": "texte_de_la_réponse"
    }
    ```

## Documentation API

La documentation interactive de l'API est disponible à l'adresse suivante :
```
http://localhost:5001/apidocs/
```
