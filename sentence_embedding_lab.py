from sentence_transformers import SentenceTransformer
# Load the pre-trained sentence embedding model (all-MiniLM-L6-v2)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Example sentences to embed
sentences = [
    "Machine learning is changing the world.",
    "Artificial intelligence and machine learning revolutionize industries.",
    "The cat sits on the mat.",
    "A kitten is sitting on a rug."
]

# Compute embeddings
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"Embedding shape: {embeddings.shape}")

import numpy as np

# Cosine similarity between sentence 0 and 1 (which are about ML/AI)
cos_sim_0_1 = np.dot(embeddings[0], embeddings[1])
# Cosine similarity between sentence 2 and 3 (both about cat on mat)
cos_sim_2_3 = np.dot(embeddings[2], embeddings[3])
print("Cosine sim between 0 and 1:", cos_sim_0_1)
print("Cosine sim between 2 and 3:", cos_sim_2_3)

from transformers import AutoTokenizer, AutoModel
import torch

distilbert_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(distilbert_name)
model_distilbert = AutoModel.from_pretrained(distilbert_name)

sentence = "Machine learning is changing the world."
inputs = tokenizer(sentence, return_tensors='pt')
token_ids = inputs['input_ids']
mask = inputs['attention_mask']

with torch.no_grad():
    outputs = model_distilbert(token_ids, attention_mask=mask)
    token_embeddings = outputs.last_hidden_state

expanded_mask = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
summed = torch.sum(token_embeddings * expanded_mask, dim=1)
counts = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
sentence_embedding = (summed / counts).squeeze()

print("DistilBERT embedding vector:", sentence_embedding[:5], "...")
print("Length:", sentence_embedding.shape[0])

import faiss

corpus = [
    "Machine learning is changing the world of technology and society.",
    "Artificial intelligence and machine learning are revolutionizing many industries.",
    "Cats and dogs are common household pets.",
    "A kitten was found sitting on a carpet."
]

corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
d = corpus_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(corpus_embeddings)
print("FAISS index size:", index.ntotal)

query = "How is AI impacting different sectors?"
query_embedding = model.encode([query], normalize_embeddings=True)
k = 2
distances, indices = index.search(query_embedding, k)

print("Nearest neighbors indices:", indices)
print("Nearest neighbors scores:", distances)
for idx in indices[0]:
    print("Retrieved doc:", corpus[idx])

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

all_embeddings = np.vstack([embeddings, corpus_embeddings])
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(all_embeddings)

plt.figure(figsize=(6,5))
plt.scatter(emb_2d[:len(sentences), 0], emb_2d[:len(sentences), 1], c='blue', label='Query sentences')
plt.scatter(emb_2d[len(sentences):, 0], emb_2d[len(sentences):, 1], c='red', marker='^', label='Corpus docs')
for i, txt in enumerate(sentences + corpus):
    plt.annotate(f"{i}", (emb_2d[i,0]+0.1, emb_2d[i,1]+0.1))
plt.legend()
plt.title("PCA projection of sentence embeddings")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()