# 🏥 Clinical NLP RAG Pipeline
### Cotiviti Intern Assessment — Topic 1: Clinical Natural Language Technology

**Author:** Arshad Naguru | MS Artificial Intelligence, Rochester Institute of Technology  
**Assessment:** Cotiviti Healthcare Informatics Intern | June 2026

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cotiviti-clinical-nlp.streamlit.app/)

> Paste any clinical note → extract biomedical entities → retrieve relevant context → generate a structured SOAP note with ICD-10 codes

---

## 📋 Pipeline Architecture

```
Clinical Note / Patient Query
          ↓
Fine-tuned Biomedical NER          ← arshadnaguru/biomedical-ner-ncbi (F1: 0.808)
          ↓
BGE-Large Embeddings               ← BAAI/bge-large-en-v1.5 (1024-dim)
          ↓
FAISS Vector Index                 ← Cosine similarity, CPU
          ↓
Top-k Semantic Retrieval           ← Hit Rate@5: 1.000 | MRR: 0.987
          ↓
Llama-3.1-8B via Groq              ← RAG-grounded clinical generation
          ↓
SOAP Note + ICD-10 Codes + Entity Table
```

---

## 📊 Evaluation Results

| Component | Dataset | Metric | Score |
|---|---|---|---|
| NER (fine-tuned) | NCBI Disease Corpus | F1 | **0.808** |
| NER (zero-shot) | NCBI Disease Corpus | F1 | 0.189 |
| BioBERT (reference) | NCBI Disease Corpus | F1 | 0.890 |
| Retrieval | PubMedQA (50 queries) | Hit Rate@5 | **1.000** |
| Retrieval | PubMedQA (50 queries) | MRR | **0.987** |
| Generation | PubMedQA (10 Q&A) | ROUGE-1 | 0.284 |
| Generation | PubMedQA (10 Q&A) | BERTScore F1 | **0.630** |
| Faithfulness | Patient record | LLM-as-Judge | 0.600 |
| Exact Match | 8 factual queries | Accuracy | **100%** |

> Evaluation datasets: `ncbi/ncbi_disease` and `pubmed_qa` from HuggingFace — no registration required

---

## 🗂️ Repository Structure

```
cotiviti-clinical-nlp/
│
├── app.py                              ← Streamlit web app (live demo)
├── requirements.txt                    ← Python dependencies
│
├── notebooks/
│   ├── clinical_nlp_rag_pipeline.ipynb     ← Full RAG pipeline (Colab)
│   └── clinical_nlp_full_evaluation.ipynb  ← Evaluation notebook (Colab)
│
├── report/
│   └── cotiviti_nlp_report.docx        ← 3-page Word report (APA)
│
├── presentation/
│   └── cotiviti_presentation_final.pptx ← 12-slide PowerPoint
│
├── eval_outputs/
│   ├── eval_dashboard.png              ← 6-panel evaluation dashboard
│   ├── eval_all_results.json           ← All metrics in JSON
│   ├── eval_retrieval_pubmedqa.csv
│   ├── eval_generation.csv
│   ├── eval_faithfulness.csv
│   └── eval_exact_match.csv
│
└── .streamlit/
    └── secrets.toml                    ← API keys (gitignored)
```

---

## 🛠️ Technical Stack

| Component | Technology | Details |
|---|---|---|
| NER | `arshadnaguru/biomedical-ner-ncbi` | Fine-tuned on NCBI Disease Corpus, 5 epochs, A100 |
| Embeddings | `BAAI/bge-large-en-v1.5` | 1024-dim, normalized, BGE query prefix |
| Vector DB | FAISS CPU | IndexFlatIP, cosine similarity |
| Chunking | Semantic + overlap | 256 tokens, 64-token overlap, sentence boundaries |
| LLM | Llama-3.1-8B via Groq | RAG-grounded, 0.2 temperature |
| Fine-tuning | HuggingFace Trainer | fp16, cosine LR schedule, batch 32 |
| Hardware | Google Colab Pro A100 | 40GB VRAM, ~30 min fine-tuning |

---

## 🚀 Run Locally

```bash
git clone https://github.com/arshadnaguru/cotiviti-clinical-nlp
cd cotiviti-clinical-nlp

pip install -r requirements.txt

# Add your API key
echo 'GROQ_API_KEY = "gsk_your_key"' > .streamlit/secrets.toml

streamlit run app.py
```

**Get a free Groq API key at:** [console.groq.com](https://console.groq.com) — no credit card needed

---

## 📄 Deliverables

| Deliverable | Description |
|---|---|
| [Word Report](report/cotiviti_nlp_report.docx) | 2-page report + APA bibliography |
| [PowerPoint](presentation/cotiviti_presentation_final.pptx) | 12-slide presentation |
| [RAG Pipeline Notebook](notebooks/clinical_nlp_rag_pipeline.ipynb) | Full pipeline implementation |
| [Evaluation Notebook](notebooks/clinical_nlp_full_evaluation.ipynb) | 6-metric evaluation |
| [Live Demo](https://arshadnaguru-cotiviti-clinical-nlp.streamlit.app) | Interactive Streamlit app |

---

## 🤖 Fine-Tuned NER Model

The NER model is publicly available on HuggingFace Hub:

```python
from transformers import pipeline

ner = pipeline("ner",
    model="arshadnaguru/biomedical-ner-ncbi",
    aggregation_strategy="simple"
)

results = ner("Patient presents with Type 2 diabetes mellitus and hypertension.")
```

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-arshadnaguru%2Fbiomedical--ner--ncbi-yellow)](https://huggingface.co/arshadnaguru/biomedical-ner-ncbi)

---

## 👤 Author

**Arshad Naguru**  
MS Artificial Intelligence · Rochester Institute of Technology · GPA 4.0  
📧 an2629@rit.edu  
🐙 [github.com/arshadnaguru](https://github.com/arshadnaguru)  
🔗 [linkedin.com/in/mahammadarshad](https://linkedin.com/in/mahammadarshad)

---

*Cotiviti Intern Assessment · Topic 1: Clinical Natural Language Technology for Health Care · June 2026*
