# Medical QA RAG System

This project implements a question-answering system using medical QA pairs. It uses sentence-transformer embeddings for retrieval and T5/FLAN-T5 models for generation. The final model supports both retrieval-based and generation-based predictions, with evaluation using Cosine Similarity and ROUGE metrics.

## Dataset

The dataset contains question-answer pairs related to medical topics. It is cleaned and split into train and test sets (80/20).

## Components

### 1. Retrieval (RAG Retriever)
- Uses `MiniLM` from Sentence Transformers to embed answers.
- Retrieves top-k similar answers to a question using cosine similarity.

### 2. Generation (RAG Generator)
- Retrieves top-3 answers using semantic search.
- Uses `google/flan-t5-base` (or `t5-base`) to generate a new answer conditioned on the question and retrieved context.

## Evaluation

### Retriever
- Evaluated using **Top-1 Accuracy**.
- Compares if the correct answer is the top-1 retrieved.

### Generator
- Evaluated using **ROUGE-1**, **ROUGE-2**, and **ROUGE-L** (Precision, Recall, F1).
- Reference answers are from the test set.

## Usage

Instantiate the class with a cleaned dataframe:
```python
nlp = qa_model(dataframe=df)
```

Run predictions:
```python
nlp.predict(solution_type=1, question="What causes Glaucoma?")  # Retriever
nlp.predict(solution_type=2, question="What causes Glaucoma?")  # Generator
```

Run evaluations:
```python
nlp.evaluate_rag_retriever()
nlp.evaluate_rag_generator()
```

## Notes
- Embeddings are computed only on the training set answers.
- Generator uses context concatenation of top-k retrieved answers.
- Model runs on CPU by default

## Dependencies
- `transformers`
- `sentence-transformers`
- `scikit-learn`
- `rouge`
