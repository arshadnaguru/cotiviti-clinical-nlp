# 🏥 Clinical NLP RAG Pipeline
### Cotiviti Intern Assessment — Topic 1: Clinical Natural Language Technology

**Author:** Arshad Naguru | MS Artificial Intelligence, Rochester Institute of Technology  
**Assessment:** Cotiviti Healthcare Informatics Intern | June 2026

---

## 🎥 Video Presentation

[![Watch Presentation](https://img.shields.io/badge/▶%20Watch-Video%20Presentation-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1hipti9KTWLKeaqBWwj4YVvL-m3wMVW4A/view?usp=drive_link)

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
| NER (zero-shot baseline) | NCBI Disease Corpus | F1 | 0.235 |
| BioBERT (reference) | NCBI Disease Corpus | F1 | 0.890 |
| Retrieval | PubMedQA (50 queries) | Hit Rate@5 | **1.000** |
| Retrieval | PubMedQA (50 queries) | MRR | **0.987** |
| Generation | PubMedQA (10 Q&A) | ROUGE-1 | 0.284 |
| Generation | PubMedQA (10 Q&A) | BERTScore F1 | **0.630** |
| Faithfulness | Patient record (5 queries) | LLM-as-Judge | 0.800 |
| Exact Match | 8 factual queries | Accuracy | **100%** |

> **Evaluation datasets:** `ncbi/ncbi_disease` and `pubmed_qa` — both loaded directly from HuggingFace, no registration required

---

## 🗂️ Repository Structure

```
cotiviti-clinical-nlp/
│
├── app.py                                    ← Streamlit web app (live demo)
├── requirements.txt                          ← Python dependencies
│
├── notebooks/
│   ├── clinical_nlp_rag_pipeline.ipynb       ← Full RAG pipeline (Google Colab)
│   └── clinical_nlp_full_evaluation.ipynb    ← 6-metric evaluation (Google Colab)
│
├── report/
│   └── cotiviti_nlp_report_final.docx        ← 3-page Word report (APA format)
│
├── presentation/
│   └── cotiviti_9slides_final.pptx           ← 9-slide PowerPoint with speaker notes
│
└── eval_outputs/
    ├── eval_dashboard.png                    ← 6-panel evaluation dashboard
    ├── eval_all_results.json                 ← All metrics in JSON
    ├── eval_retrieval_pubmedqa.csv           ← Retrieval evaluation (50 queries)
    ├── eval_generation.csv                   ← ROUGE + BERTScore results
    ├── eval_faithfulness.csv                 ← LLM-as-judge verdicts
    └── eval_exact_match.csv                  ← Factual accuracy results
```

---

## 🛠️ Technical Stack

| Component | Technology | Details |
|---|---|---|
| NER Model | `arshadnaguru/biomedical-ner-ncbi` | Fine-tuned on NCBI Disease Corpus · 5 epochs · A100 GPU |
| Base NER | `d4data/biomedical-ner-all` | DistilBERT-based · General biomedical NER |
| Embeddings | `BAAI/bge-large-en-v1.5` | 1024-dim · normalized · BGE query prefix |
| Vector DB | FAISS CPU | IndexFlatIP · cosine similarity · exact search |
| Chunking | Semantic + overlap | 256 tokens · 64-token overlap · sentence boundaries |
| LLM | Llama-3.1-8B via Groq API | RAG-grounded · temperature 0.2 · HIPAA-compliant |
| Fine-tuning | HuggingFace Trainer | fp16 · cosine LR schedule · batch 32 · LR 2e-5 |
| Training Hardware | Google Colab Pro A100 | 40GB VRAM · ~30 min fine-tuning |
| Deployment | Streamlit Cloud | Live URL · API keys in secrets |

---

## 📄 Deliverables

| # | Deliverable | Link |
|---|---|---|
| 1 | 🎥 Video Presentation | [Google Drive](https://drive.google.com/file/d/1hipti9KTWLKeaqBWwj4YVvL-m3wMVW4A/view?usp=drive_link) |
| 2 | 🌐 Live Streamlit App | [cotiviti-clinical-nlp.streamlit.app](https://cotiviti-clinical-nlp.streamlit.app/) |
| 3 | 📄 Word Report | [cotiviti_nlp_report_final.docx](report/cotiviti_nlp_report_final.docx) |
| 4 | 📊 PowerPoint | [cotiviti_9slides_final.pptx](presentation/cotiviti_9slides_final.pptx) |
| 5 | 💻 RAG Pipeline Notebook | [clinical_nlp_rag_pipeline.ipynb](notebooks/clinical_nlp_rag_pipeline.ipynb) |
| 6 | 📈 Evaluation Notebook | [clinical_nlp_full_evaluation.ipynb](notebooks/clinical_nlp_full_evaluation.ipynb) |
| 7 | 🤗 HuggingFace Model | [arshadnaguru/biomedical-ner-ncbi](https://huggingface.co/arshadnaguru/biomedical-ner-ncbi) |

---

## 🤖 Fine-Tuned NER Model

The NER model is publicly available on HuggingFace Hub — fine-tuned on NCBI Disease Corpus, F1 improved from **0.235 → 0.808** (BioBERT reference: 0.890).

```python
from transformers import pipeline

ner = pipeline(
    "ner",
    model="arshadnaguru/biomedical-ner-ncbi",
    aggregation_strategy="simple"
)

results = ner("Patient presents with Type 2 diabetes mellitus and hypertension.")
# Returns: [{'entity_group': 'Disease_disorder', 'word': 'Type 2 diabetes mellitus', 'score': 0.998}, ...]
```

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-arshadnaguru%2Fbiomedical--ner--ncbi-yellow)](https://huggingface.co/arshadnaguru/biomedical-ner-ncbi)

---

## 🚀 Run Locally

```bash
git clone https://github.com/arshadnaguru/cotiviti-clinical-nlp
cd cotiviti-clinical-nlp

pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_your_key_here"' > .streamlit/secrets.toml

streamlit run app.py
```

---

## 👤 Author

**Arshad Naguru**  
MS Artificial Intelligence · Rochester Institute of Technology · GPA 4.0  
📧 an2629@rit.edu  
🐙 [github.com/arshadnaguru](https://github.com/arshadnaguru)  
🔗 [linkedin.com/in/mahammadarshad](https://linkedin.com/in/mahammadarshad)

---

*Cotiviti Intern Assessment · Topic 1: Clinical Natural Language Technology for Health Care · June 2026*
