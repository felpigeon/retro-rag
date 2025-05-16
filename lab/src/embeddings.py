import os
from google import genai

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))


def generate_embeddings(documents):
    embeddings = []

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]

        result = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
        )

        embeddings += [e.values for e in result.embeddings]

    return embeddings
