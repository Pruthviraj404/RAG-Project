# Retrieval-Augmented Generation (RAG) System for Machine Learning Documents

##  Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system on Machine Learning documents.  
The objective is to analyze how chunking strategies and prompt design affect retrieval quality and generated responses.

The system extracts text from PDF documents, generates embeddings, stores them in a vector database, retrieves relevant context, and uses a language model to generate answers.

---

##  Objectives
- Implement a baseline RAG pipeline
- Compare fixed-size and sentence-based chunking strategies
- Evaluate retrieval relevance and answer quality
- Analyze real-world extraction challenges
- Reflect on system limitations and improvements

---

## System Architecture
1. **PDF Text Extraction**
2. **Text Cleaning**
3. **Chunking Strategy**
   - Fixed-size chunking
   - Sentence-based chunking
4. **Embedding Generation**
   - Sentence Transformers (MiniLM)
5. **Vector Storage**
   - ChromaDB
6. **Retrieval**
   - Top-k similarity search
7. **Answer Generation**
   - Local LLM (Ollama with Mistral/Gallama)

---

##  Setup Instructions

### Prerequisites
- Python 3.9+
- Jupyter Notebook
- VS Code (recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
Local LLM Setup
This project uses Ollama with the Mistral model:

bash
ollama pull mistral
Run Notebook
Open Jupyter Notebook and execute all cells in order:

Part 1: Data collection & analysis

Part 2: Baseline RAG

Part 3: Experiments (chunking, prompting)

Part 4: Handling challenges

Part 5: Final reflection

 Expected Output
Extracted text chunks from PDFs

Embeddings stored in ChromaDB

Retrieval results for 10 evaluation questions

Comparison tables for chunking and prompting strategies

Final structured answers from the improved pipeline

 Deliverables
notebooks/ – Jupyter notebook with code and analysis

experiment_results.md – Summary of experiments

README.md – Setup and usage instructions

requirements.txt – Dependencies list

data/ – Sample ML PDFs (if shareable)

 Experiments Conducted
1️ Baseline Evaluation
Fixed-size chunking

Basic prompt

Manual scoring (Retrieval relevance & Answer quality)

2️ Chunking Strategy Comparison
Fixed-size vs Sentence-based chunking

Quantitative scoring

Sentence-based chunking preserved context better

3️ Prompt Engineering
Basic vs Structured prompting

Improved readability and formatting

 Key Findings
Both chunking strategies performed reliably for short, definition-based queries.

Sentence-based chunking produced clearer, more contextual answers.

Structured prompts significantly improved answer clarity.

PDF extraction noise (headers, equations, terminology) impacted retrieval quality.

 Limitations
Formatting loss during PDF extraction (equations, tables).

Dense technical terminology sometimes confused embeddings.

Manual evaluation (no automated metrics like precision@k).

Small dataset size, mostly definition-based queries.

 Cost & Monitoring
No API costs: Ollama runs locally.

If deployed with a paid API (e.g., GPT‑3.5), estimated cost for 100 users × 10 queries/day ≈ $30/month.

In production, monitoring would include:

Retrieval relevance

Answer accuracy

Latency

Failure modes (e.g., “I don’t know” fallback too often)

 Reflection
This project provided practical insights into RAG pipeline design, retrieval trade-offs, and prompt engineering.
While chunking strategy had limited impact for simple conceptual queries, the study highlights how dataset structure influences retrieval effectiveness.

Theory vs Practice:  
In theory, RAG pipelines seem straightforward, but in practice I faced challenges with noisy text, overlapping content, and environment setup (Ollama installation, PATH variables).
I learned that preprocessing and chunking decisions matter more than expected. Compared to academic ML projects, this assignment felt closer to real-world engineering: dealing with messy data, debugging tools, and balancing trade-offs between accuracy, cost, and reproducibility.

Author
Pratik Malunjkar
