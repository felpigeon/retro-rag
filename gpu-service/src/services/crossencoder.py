from sentence_transformers import CrossEncoder


crossencoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device='cuda')