# FAISS Similarity Search with USE (Jupyter Notebook Project)

This project demonstrates semantic similarity search using:

- **FAISS** (Facebook AI Similarity Search) for fast nearest-neighbor indexing and retrieval
- **Universal Sentence Encoder (USE)** for generating dense sentence embeddings

The workflow is implemented as a **Jupyter Notebook** so you can run each step interactively.

## Project Goal

Build a simple semantic search system that:

1. Encodes text data into vector embeddings with USE
2. Indexes vectors using FAISS
3. Accepts a query sentence
4. Returns the most semantically similar sentences/documents

## Recommended Notebook Structure

Use a notebook (for example, `faiss_use_similarity_search.ipynb`) with these sections:

1. **Install Dependencies**
2. **Import Libraries**
3. **Load or Define Text Corpus**
4. **Generate USE Embeddings**
5. **Create FAISS Index**
6. **Run Similarity Search Queries**
7. **Display Top-K Results**

## Dependencies

Install the required packages in a notebook cell:

```bash
pip install faiss-cpu tensorflow tensorflow-hub numpy pandas
```

## Example Implementation (Notebook Cells)

### 1) Imports

```python
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import faiss
```

### 2) Sample Corpus

```python
corpus = [
    "FAISS is a library for efficient similarity search.",
    "Universal Sentence Encoder creates semantic embeddings.",
    "Jupyter Notebook is great for interactive machine learning experiments.",
    "Vector databases help retrieve similar information quickly.",
    "Deep learning models can represent text as dense vectors."
]
```

### 3) Load USE Model and Create Embeddings

```python
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embeddings = model(corpus).numpy().astype("float32")
```

### 4) Build FAISS Index

```python
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

### 5) Query and Retrieve Top-K Similar Sentences

```python
query = "How can I search similar sentences efficiently?"
query_embedding = model([query]).numpy().astype("float32")

k = 3
distances, indices = index.search(query_embedding, k)

for rank, idx in enumerate(indices[0], start=1):
    print(f"{rank}. {corpus[idx]} (distance={distances[0][rank-1]:.4f})")
```

## Expected Outcome

You should get the top-k corpus entries that are closest to the query in embedding space, enabling practical semantic search.

## Notes

- `IndexFlatL2` is an exact L2 index. For larger datasets, explore IVF/HNSW/PQ FAISS indexes.
- You can replace the sample corpus with data from CSV files, databases, or APIs.
- If you use GPU, replace `faiss-cpu` with a GPU-compatible FAISS installation.
