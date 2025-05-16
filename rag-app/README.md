# Application RAG

Cette application est un service de Retrieval Augmented Generation (RAG) développé en Java utilisant Gradle comme système de build.

## Fonctionnalités

L'application expose les endpoints REST suivants:

- `/ask` : Permet de poser une question au système et d'obtenir une réponse basée sur les documents indexés.
- `/ingest` : Permet d'ingérer un document individuel dans le système.
- `/ingest_batch` : Permet d'ingérer plusieurs documents en une seule requête.

## Prérequis

- Java
- Gradle

## Compilation

Pour compiler l'application:

```bash
./gradlew build
```

## Exécution

Pour exécuter l'application:

```bash
./gradlew bootRun
```

L'application sera accessible à l'adresse `http://localhost:8080`.

## Tests

### Exécution des tests unitaires

Pour exécuter tous les tests unitaires:

```bash
./gradlew test
```

## Documentation

### Génération de la documentation

Pour générer la documentation Javadoc:

```bash
./gradlew javadoc
```

La documentation générée sera disponible dans le répertoire `build/docs/javadoc/`.

## Utilisation

### Exemple de requête à l'API

#### Poser une question

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est la procédure pour..."}'
```

#### Ingérer un document

```bash
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Document Title"}'
```

#### Ingérer plusieurs documents

```bash
curl -X POST http://localhost:8080/ingest_batch \
  -H "Content-Type: application/json" \
  -d '[{"text": "Document 1"}, {"text": "Document 2"}]'
```

## Structure du projet

```
rag-app/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/example/rag/
│   │   │       ├── controllers/
│   │   │       ├── services/
│   │   │       ├── models/
│   │   │       └── Application.java
│   │   └── resources/
│   └── test/
│       ├── java/
│       └── resources/
├── build.gradle
├── settings.gradle
└── README.md
```
