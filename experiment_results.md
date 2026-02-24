# Experiment Results – RAG Study Assistant

##  Baseline Evaluation (Fixed-size Chunk + Basic Prompt)

| Question | Retrieval (1–5) | Answer Quality (1–5) | Observations |
|----------|-----------------|----------------------|--------------|
| What is machine learning? | 4 | 4 | Clear definition, slightly brief |
| Who coined the term machine learning? | 3 | 3 | Name retrieved, missing context |
| What are the categories of machine learning? | 3 | 4 | Correct categories, phrasing limited |
| What is supervised learning? | 3 | 3 | Clear definition, lacked examples |
| What is unsupervised learning? | 3 | 3 | Definition correct, missing clustering examples |
| What is reinforcement learning? | 3 | 4 | Good definition, missing trial/error detail |
| What is overfitting? | 3 | 4 | Accurate, slightly brief |
| What is the role of data preprocessing? | 3 | 3 | Verbose, lacked structured explanation |
| What is the difference between AI and ML? | 2 | 2 | Confused context, fallback “I don’t know” |
| What are applications of machine learning? | 3 | 3 | Few examples, missing breadth |

---

##  Chunking Strategy Comparison

### Average Scores Summary

| Strategy        | Avg Retrieval | Avg Quality | Notes |
|-----------------|---------------|-------------|-------|
| Fixed-size      | 4.0           | 4.1         | Concise but fragmented |
| Sentence-based  | 5.0           | 4.9         | Clearer, more contextual |

**Conclusion:** Sentence-based chunking preserved context better, especially for conceptual queries. Fixed-size chunking was simpler but often cut sentences mid-way.

---

##  Prompting Techniques Comparison

| Question | Basic Prompt | Improved Prompt | Observations |
|----------|--------------|-----------------|--------------|
| What is supervised learning? | Definition only | Definition + examples, structured | Improved clarity |
| What is reinforcement learning? | Verbose, less structured | Clear, bullet points | Easier to study |
| Who coined the term machine learning? | Correct name | Correct name + extra detail | Extra detail unnecessary |

**Conclusion:** Improved prompting produced clearer, study-friendly answers. Best for conceptual queries; factual queries sometimes added noise.

---

## ⚠️ Real-World Challenges

1. **Headers/Footers Noise** – Removed with regex cleaning.  
2. **Equations Losing Formatting** – Preserved partially, but LaTeX support needed.  
3. **Dense Technical Terminology** – Sentence-based chunking helped, but domain-specific embeddings would improve accuracy.

---

## 🧠 Reflection Summary

- **What worked:** Sentence-based chunking, structured prompts, local Ollama setup.  
- **What didn’t:** Handling equations/tables, factual queries with improved prompts.  
- **Future improvements:** Domain-specific embeddings, LaTeX-aware parsing, better table extraction.  
- **Theory vs Practice:** Tutorials made RAG look simple, but real-world data cleaning and environment setup were the hardest parts. Learned that preprocessing is as important as algorithms.  
- **Cost:** Free with Ollama; cloud APIs would cost ~$30/month for 100 users × 10 queries/day.  
- **Monitoring:** Retrieval relevance, answer accuracy, latency, fallback rates.


