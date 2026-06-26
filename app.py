"""
Clinical NLP RAG Pipeline — Interactive Demo
Cotiviti Intern Assessment | Topic 1: Clinical Natural Language Technology
Author: Arshad Naguru | MS AI, Rochester Institute of Technology
"""

import streamlit as st
import os
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical NLP RAG Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLES ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0D4F8B 0%, #00A8CC 100%);
    padding: 28px 36px;
    border-radius: 14px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 8px 32px rgba(13,79,139,0.25);
}
.main-header h1 { color: white; font-size: 1.9rem; margin: 0 0 6px 0; font-weight: 700; }
.main-header p  { color: #CADCFC; margin: 0; font-size: 0.95rem; }

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    border-top: 4px solid #0D4F8B;
    height: 100%;
}
.metric-card .val  { font-size: 2rem; font-weight: 700; color: #0D4F8B; }
.metric-card .lbl  { font-size: 0.82rem; color: #64748B; margin-top: 4px; }
.metric-card .sub  { font-size: 0.75rem; color: #94A3B8; }

.entity-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 2px 3px;
}
.entity-Disease_disorder  { background: #FEE2E2; color: #991B1B; }
.entity-Medication         { background: #DBEAFE; color: #1E40AF; }
.entity-Sign_symptom       { background: #FEF9C3; color: #854D0E; }
.entity-Anatomical_location{ background: #F3E8FF; color: #6B21A8; }
.entity-Lab_value          { background: #DCFCE7; color: #166534; }
.entity-Diagnostic_procedure{ background: #FFE4E6; color: #9F1239; }
.entity-Therapeutic_procedure{ background: #E0F2FE; color: #075985; }
.entity-other              { background: #F1F5F9; color: #475569; }

.soap-section {
    background: #F8FAFC;
    border-left: 4px solid #0D4F8B;
    padding: 14px 18px;
    border-radius: 0 10px 10px 0;
    margin-bottom: 12px;
}
.soap-label {
    font-weight: 700;
    color: #0D4F8B;
    font-size: 1rem;
    margin-bottom: 6px;
}
.retrieved-chunk {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #00A8CC;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #334155;
}
.chunk-source {
    font-size: 0.75rem;
    color: #00A8CC;
    font-weight: 600;
    margin-bottom: 6px;
}
.step-badge {
    background: #0D4F8B;
    color: white;
    border-radius: 50%;
    width: 28px;
    height: 28px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 8px;
}
.info-box {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-size: 0.9rem;
    color: #1E40AF;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.92rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── SAMPLE CLINICAL NOTES ─────────────────────────────────────────────────────
SAMPLE_NOTES = {
    "🩺 Diabetes & Hypertension": """Patient is a 58-year-old male presenting with poorly controlled Type 2 diabetes mellitus and hypertension. HbA1c is 9.2%, fasting glucose 210 mg/dL. Current medications include Metformin 1000mg twice daily and Lisinopril 10mg daily. Patient reports increased thirst, polyuria, and mild peripheral neuropathy in bilateral lower extremities. Blood pressure today 148/92 mmHg. Creatinine 1.4 mg/dL, eGFR 52. Plan to add Empagliflozin 10mg and refer to endocrinology. Follow-up in 3 months with repeat HbA1c and renal panel.""",

    "🫁 Community-Acquired Pneumonia": """68-year-old female admitted with community-acquired pneumonia. Presents with productive cough, fever of 39.2°C, and dyspnea. Oxygen saturation 91% on room air. Chest CT shows right lower lobe consolidation consistent with bacterial pneumonia. WBC 14,200 with left shift. Sputum culture pending. Started on Ceftriaxone 1g IV and Azithromycin 500mg IV. Patient has history of COPD, currently on Tiotropium inhaler and Fluticasone-Salmeterol. Placed on 4L nasal cannula oxygen with improvement in saturation to 96%.""",

    "❤️ Acute STEMI": """45-year-old male presenting to ED with acute onset chest pain radiating to left arm, diaphoresis and nausea for past 2 hours. ECG shows ST elevation in leads II, III, aVF suggestive of inferior STEMI. Troponin I elevated at 2.8 ng/mL. Patient given Aspirin 325mg, Clopidogrel 600mg loading dose, and Heparin IV bolus. Cardiology activated cath lab. History of hyperlipidemia, on Atorvastatin 40mg. BP 110/70, HR 88. Patient transferred urgently for primary PCI.""",

    "🦴 Rheumatoid Arthritis": """55-year-old woman with rheumatoid arthritis presents with worsening joint pain, morning stiffness lasting over 2 hours, and fatigue. Current medications include Methotrexate 15mg weekly and Folic Acid 1mg daily. ESR elevated at 68 mm/hr, CRP 3.2 mg/L. Anti-CCP antibodies positive. X-ray of hands shows early erosive changes in MCP joints bilaterally. Plan to add Hydroxychloroquine 200mg twice daily and refer to rheumatology for consideration of biologic therapy. Monitor LFTs monthly due to Methotrexate.""",

    "✏️ Custom Note": ""
}

PATIENT_RECORD = {
    "Visit 1 - Initial Presentation": """Patient John M., 58-year-old male. Chief Complaint: increased thirst, frequent urination, fatigue for 3 months. BMI 31.2. Vitals: BP 152/94 mmHg, HR 82. Labs: Fasting glucose 218 mg/dL, HbA1c 9.4%, Total cholesterol 224 mg/dL, LDL 148 mg/dL, Creatinine 1.3 mg/dL, eGFR 58. Assessment: New diagnosis Type 2 diabetes mellitus with hypertension and dyslipidemia. Plan: Start Metformin 500mg twice daily, Lisinopril 10mg daily.""",
    "Visit 2 - Follow-up": """Patient John M., 58-year-old male. BP 138/88 mmHg, Weight 94.2kg. Labs: Fasting glucose 186 mg/dL, HbA1c 8.6%, Creatinine 1.4 mg/dL, eGFR 52. Peripheral neuropathy: reduced sensation bilateral lower extremities, monofilament abnormal. Assessment: Early diabetic peripheral neuropathy. CKD 3a. Plan: Add Empagliflozin 10mg, Gabapentin 300mg nightly.""",
    "Radiology Report": """PA and lateral chest radiograph. Findings: Heart size mildly enlarged, cardiothoracic ratio 0.52. Mild pulmonary vascular congestion bilaterally. Aortic knuckle prominent consistent with atherosclerotic changes. Impression: Mild cardiomegaly with early pulmonary vascular redistribution. Atherosclerotic aortic changes.""",
    "Endocrinology Consult": """Patient John M. Current meds: Metformin 1000mg BID, Empagliflozin 10mg, Lisinopril 10mg, Gabapentin 300mg. Allergies: Sulfa drugs. HbA1c 8.1%, Fasting glucose 174 mg/dL. Assessment: Type 2 DM inadequate control. Plan: Add Semaglutide 0.25mg weekly, Atorvastatin 40mg.""",
    "Lab Report": """Glucose: 174 mg/dL (H). Creatinine: 1.4 mg/dL (H). eGFR: 52 (L). Total Cholesterol: 218 mg/dL (H). LDL: 142 mg/dL (H). HDL: 39 mg/dL (L). Triglycerides: 198 mg/dL (H). HbA1c: 8.1% (H). Urine ACR: 48 mg/g (H). ALT: 28 U/L. AST: 24 U/L."""
}

LABEL_MAP = {
    "Disease_disorder":       ("🔴", "Disease/Disorder",    "entity-Disease_disorder"),
    "Medication":             ("💊", "Medication",           "entity-Medication"),
    "Sign_symptom":           ("🟡", "Sign/Symptom",         "entity-Sign_symptom"),
    "Anatomical_location":    ("🫀", "Anatomy",              "entity-Anatomical_location"),
    "Lab_value":              ("🧪", "Lab Value",            "entity-Lab_value"),
    "Diagnostic_procedure":   ("📋", "Diagnostic Procedure", "entity-Diagnostic_procedure"),
    "Therapeutic_procedure":  ("💉", "Therapeutic Procedure","entity-Therapeutic_procedure"),
    "Dosage":                 ("⚖️", "Dosage",               "entity-other"),
    "Age":                    ("👤", "Demographics",         "entity-other"),
}

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_ner_model():
    from transformers import pipeline
    ner = pipeline(
        "ner",
        model="arshadnaguru/biomedical-ner-ncbi",
        aggregation_strategy="simple",
        device=-1  # CPU on Streamlit Cloud
    )
    return ner

@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

@st.cache_resource(show_spinner=False)
def build_faiss_index(_embedder):
    import faiss
    from nltk.tokenize import sent_tokenize
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    def count_tokens(text): return int(len(text.split()) * 1.3)

    def semantic_chunk(text, doc_name, chunk_size=256, overlap=64):
        sentences = [s.strip() for s in sent_tokenize(text.strip()) if len(s.strip()) > 10]
        chunks, current, cur_tok = [], [], 0
        for sent in sentences:
            st = count_tokens(sent)
            if cur_tok + st > chunk_size and current:
                chunks.append({"text": " ".join(current), "doc": doc_name, "chunk_id": len(chunks)})
                overlap_sents, oc = [], 0
                for s in reversed(current):
                    t = count_tokens(s)
                    if oc + t <= overlap: overlap_sents.insert(0, s); oc += t
                    else: break
                current = overlap_sents + [sent]; cur_tok = oc + st
            else:
                current.append(sent); cur_tok += st
        if current:
            chunks.append({"text": " ".join(current), "doc": doc_name, "chunk_id": len(chunks)})
        return chunks

    all_chunks = []
    for doc_name, doc_text in PATIENT_RECORD.items():
        all_chunks.extend(semantic_chunk(doc_text, doc_name))

    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = _embedder.encode(
        chunk_texts, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, all_chunks

def run_ner(ner_pipe, text):
    raw = ner_pipe(text.strip())
    grouped = defaultdict(list)
    seen = set()
    for ent in raw:
        word  = ent["word"].strip()
        label = ent["entity_group"]
        score = round(ent["score"], 3)
        if "##" in word or len(word) < 3 or score < 0.6:
            continue
        key = (word.lower(), label)
        if key not in seen:
            seen.add(key)
            grouped[label].append({"entity": word, "confidence": score})
    return grouped

def retrieve_chunks(query, embedder, index, all_chunks, top_k=4):
    q_emb = embedder.encode(
        [f"Represent this sentence for searching relevant passages: {query}"],
        normalize_embeddings=True
    ).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    return [
        dict(**all_chunks[i], similarity_score=float(s))
        for s, i in zip(scores[0], idxs[0]) if i != -1
    ]

def call_groq(prompt, groq_api_key, max_tokens=600):
    import requests
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a clinical AI assistant. Generate structured, accurate medical summaries using only the provided context. Be concise and use clinical terminology."
            },
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers, json=payload, timeout=30
    )
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get free key at console.groq.com"
    )

    st.markdown("---")
    st.markdown("### 🔧 Pipeline Settings")

    top_k = st.slider("Retrieved chunks (top-k)", 2, 6, 4,
                      help="Number of relevant chunks retrieved from patient record")
    show_chunks = st.checkbox("Show retrieved chunks", value=True)
    show_entities = st.checkbox("Show NER entities", value=True)

    st.markdown("---")
    st.markdown("### 📊 Evaluation Results")

    metrics = [
        ("NER F1", "0.808", "NCBI Disease Corpus"),
        ("Hit Rate@5", "1.000", "PubMedQA Retrieval"),
        ("MRR", "0.987", "PubMedQA Retrieval"),
        ("Exact Match", "100%", "8 Factual Queries"),
        ("BERTScore", "0.630", "Generation Quality"),
        ("Faithfulness", "0.600", "LLM-as-Judge"),
    ]
    for label, val, sub in metrics:
        st.markdown(f"""
        <div style="background:#F0F6FC;border-radius:8px;padding:8px 12px;margin-bottom:6px;">
            <div style="font-size:1.1rem;font-weight:700;color:#0D4F8B;">{val}</div>
            <div style="font-size:0.8rem;font-weight:600;color:#334155;">{label}</div>
            <div style="font-size:0.72rem;color:#94A3B8;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem;color:#64748B;">
    <b>Models Used:</b><br>
    🏷️ arshadnaguru/biomedical-ner-ncbi<br>
    🔍 BAAI/bge-large-en-v1.5<br>
    🤖 Llama-3.1-8B (Groq)<br>
    📦 FAISS CPU Vector Index<br><br>
    <b>Author:</b> Arshad Naguru<br>
    MS AI · Rochester Institute of Technology<br>
    Cotiviti Intern Assessment · June 2026
    </div>
    """, unsafe_allow_html=True)

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Clinical NLP RAG Pipeline</h1>
    <p>
    Biomedical NER (Fine-tuned, F1=0.808) &nbsp;·&nbsp;
    Semantic Chunking &nbsp;·&nbsp;
    BGE-Large Embeddings &nbsp;·&nbsp;
    FAISS Vector Search &nbsp;·&nbsp;
    Llama-3.1-8B Clinical Generation
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Cotiviti Intern Assessment &nbsp;·&nbsp; Arshad Naguru · RIT MS AI
    </p>
</div>
""", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Clinical Note Analysis",
    "💬 Patient Record Q&A",
    "📊 Evaluation Dashboard",
    "ℹ️ About the Pipeline"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLINICAL NOTE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Paste a clinical note to extract entities and generate a structured SOAP note")

    col1, col2 = st.columns([1, 2])

    with col1:
        selected = st.selectbox(
            "Select a sample note or write your own:",
            list(SAMPLE_NOTES.keys())
        )

    note_text = SAMPLE_NOTES[selected]
    if selected == "✏️ Custom Note":
        note_text = st.text_area(
            "Paste your clinical note here:",
            height=220,
            placeholder="Patient is a 45-year-old male presenting with..."
        )
    else:
        st.text_area("Clinical Note:", value=note_text, height=180, disabled=True)

    if not groq_api_key:
        st.markdown("""
        <div class="info-box">
        ⚠️ Add your Groq API key in the sidebar to enable SOAP note generation.
        Get a free key at <b>console.groq.com</b> — no credit card needed.
        NER entity extraction works without the API key.
        </div>
        """, unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True)

    if run_btn and note_text.strip():

        # Load models
        with st.spinner("Loading models (first run takes ~60s)..."):
            try:
                ner_pipe = load_ner_model()
                embedder = load_embedder()
                faiss_index, all_chunks = build_faiss_index(embedder)
            except Exception as e:
                st.error(f"Model loading failed: {e}")
                st.stop()

        # Step 1: NER
        with st.spinner("🔍 Running fine-tuned biomedical NER..."):
            t0 = time.time()
            grouped = run_ner(ner_pipe, note_text)
            ner_time = round(time.time() - t0, 2)

        # Step 2: RAG Retrieval
        with st.spinner("📚 Retrieving relevant context from patient record..."):
            t0 = time.time()
            retrieved = retrieve_chunks(note_text[:500], embedder, faiss_index, all_chunks, top_k)
            ret_time = round(time.time() - t0, 2)

        # Step 3: Generate SOAP
        soap_text = ""
        gen_time  = 0
        if groq_api_key:
            with st.spinner("🤖 Generating SOAP note with Llama-3.1-8B..."):
                t0 = time.time()
                ctx = "\n\n".join([f"[{r['doc']}]\n{r['text']}" for r in retrieved])
                entity_lines = []
                for label, ents in grouped.items():
                    info = LABEL_MAP.get(label, ("", label, ""))
                    terms = ", ".join([e["entity"] for e in ents])
                    if terms:
                        entity_lines.append(f"- {info[1]}: {terms}")
                entity_str = "\n".join(entity_lines)

                prompt = f"""You are a clinical AI assistant. Based on the clinical note and extracted entities below, generate a structured SOAP note.

CLINICAL NOTE:
{note_text}

EXTRACTED ENTITIES:
{entity_str}

ADDITIONAL CONTEXT FROM PATIENT RECORD:
{ctx[:1500]}

Generate a complete SOAP note:

S (Subjective): Patient's reported symptoms and history
O (Objective): Measurable findings, vitals, lab values
A (Assessment): Primary diagnosis with ICD-10 codes in parentheses
P (Plan): Treatment plan, medications with doses, referrals, follow-up

Also provide:
Key Clinical Concerns (2-3 bullet points)
Suggested ICD-10 Codes (list the most relevant)"""

                try:
                    soap_text = call_groq(prompt, groq_api_key)
                    gen_time  = round(time.time() - t0, 2)
                except Exception as e:
                    st.error(f"Groq API error: {e}")

        # ── RESULTS ──────────────────────────────────────────────────────────
        st.markdown("---")

        # Timing badges
        c1, c2, c3 = st.columns(3)
        c1.success(f"✅ NER: {ner_time}s")
        c2.success(f"✅ Retrieval: {ret_time}s")
        if groq_api_key:
            c3.success(f"✅ Generation: {gen_time}s")

        st.markdown("---")

        left, right = st.columns(2)

        # NER Results
        with left:
            st.markdown("#### 🏷️ Extracted Clinical Entities")
            st.caption(f"Model: arshadnaguru/biomedical-ner-ncbi | F1: 0.808 on NCBI Disease Corpus")

            if not grouped:
                st.info("No entities detected.")
            else:
                for label, ents in grouped.items():
                    info = LABEL_MAP.get(label, ("🔹", label, "entity-other"))
                    icon, readable, css = info
                    with st.expander(f"{icon} {readable} ({len(ents)})", expanded=True):
                        for e in ents:
                            conf_color = "#22C55E" if e["confidence"] > 0.85 else "#F59E0B"
                            st.markdown(f"""
                            <div style="display:flex;justify-content:space-between;
                                        padding:5px 10px;background:#F8FAFC;
                                        border-radius:6px;margin-bottom:4px;">
                                <span class="entity-tag {css}">{e['entity']}</span>
                                <span style="font-size:0.75rem;color:{conf_color};font-weight:600;">
                                    {e['confidence']:.3f}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

            # Entity table
            if grouped:
                rows = []
                for label, ents in grouped.items():
                    info = LABEL_MAP.get(label, ("", label, ""))
                    for e in ents:
                        rows.append({
                            "Category": info[1],
                            "Entity": e["entity"],
                            "Confidence": e["confidence"]
                        })
                df = pd.DataFrame(rows)
                st.download_button(
                    "📥 Download Entities CSV",
                    df.to_csv(index=False),
                    "entities.csv",
                    "text/csv",
                    use_container_width=True
                )

        # SOAP Note
        with right:
            st.markdown("#### 🤖 Generated SOAP Note")
            st.caption("Model: Llama-3.1-8B via Groq | RAG-grounded generation")

            if soap_text:
                # Parse sections
                sections = {"S": "", "O": "", "A": "", "P": "", "extra": ""}
                current = "extra"
                for line in soap_text.split("\n"):
                    if line.strip().startswith("S ("):
                        current = "S"
                    elif line.strip().startswith("O ("):
                        current = "O"
                    elif line.strip().startswith("A ("):
                        current = "A"
                    elif line.strip().startswith("P ("):
                        current = "P"
                    sections[current] += line + "\n"

                soap_labels = {
                    "S": "Subjective", "O": "Objective",
                    "A": "Assessment", "P": "Plan"
                }
                for key, label in soap_labels.items():
                    if sections[key].strip():
                        st.markdown(f"""
                        <div class="soap-section">
                            <div class="soap-label">{key} — {label}</div>
                            <div style="font-size:0.91rem;color:#334155;white-space:pre-wrap;">
                                {sections[key].strip()}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                if sections["extra"].strip():
                    st.markdown(f"""
                    <div class="soap-section" style="border-color:#00A8CC;">
                        <div style="font-size:0.91rem;color:#334155;white-space:pre-wrap;">
                            {sections["extra"].strip()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.download_button(
                    "📥 Download SOAP Note",
                    soap_text,
                    "soap_note.txt",
                    "text/plain",
                    use_container_width=True
                )
            elif not groq_api_key:
                st.info("Add Groq API key in sidebar to generate SOAP note.")
            else:
                st.warning("Generation failed. Check API key.")

        # Retrieved chunks
        if show_chunks and retrieved:
            st.markdown("---")
            st.markdown("#### 📚 Retrieved Context (FAISS Semantic Search)")
            st.caption(f"Top-{top_k} chunks from patient record | BGE-Large embeddings | Hit Rate@5: 1.000")
            for r in retrieved:
                st.markdown(f"""
                <div class="retrieved-chunk">
                    <div class="chunk-source">
                        📄 {r['doc']} &nbsp;·&nbsp; 
                        Similarity: {r['similarity_score']:.4f}
                    </div>
                    {r['text'][:300]}{'...' if len(r['text']) > 300 else ''}
                </div>
                """, unsafe_allow_html=True)

    elif run_btn:
        st.warning("Please enter a clinical note first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT RECORD Q&A
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Ask any question about the patient record — the pipeline retrieves relevant context and generates a grounded answer")

    st.markdown("""
    <div class="info-box">
    📋 The patient record contains 5 documents: Visit 1 (initial presentation), 
    Visit 2 (follow-up), Radiology Report, Endocrinology Consult, and Lab Report.
    All documents are indexed in FAISS — ask anything about this patient.
    </div>
    """, unsafe_allow_html=True)

    # Sample questions
    sample_qs = [
        "What medications is the patient currently taking and what are the doses?",
        "What are the abnormal lab values and what do they indicate?",
        "What is the progression of HbA1c across visits and is glycemic control improving?",
        "What kidney function findings are documented and what is the CKD staging?",
        "What is the radiology report finding and clinical impression?",
        "What is the complete diagnosis list with ICD-10 codes?",
    ]

    selected_q = st.selectbox("Try a sample question:", ["Custom question..."] + sample_qs)
    if selected_q == "Custom question...":
        user_query = st.text_input("Your question:", placeholder="What medications is the patient on?")
    else:
        user_query = selected_q
        st.text_input("Question:", value=user_query, disabled=True)

    ask_btn = st.button("🔍 Ask the Pipeline", type="primary", use_container_width=True)

    if ask_btn and user_query.strip():
        if not groq_api_key:
            st.error("Groq API key required for Q&A. Add it in the sidebar.")
        else:
            with st.spinner("Loading models..."):
                try:
                    ner_pipe_qa = load_ner_model()
                    embedder_qa = load_embedder()
                    faiss_index_qa, all_chunks_qa = build_faiss_index(embedder_qa)
                except Exception as e:
                    st.error(f"Model loading failed: {e}")
                    st.stop()

            with st.spinner("Retrieving + generating answer..."):
                retrieved_qa = retrieve_chunks(user_query, embedder_qa, faiss_index_qa, all_chunks_qa, top_k)
                ctx_qa = "\n\n".join([f"[{r['doc']}]\n{r['text']}" for r in retrieved_qa])

                prompt_qa = f"""You are a clinical AI assistant. Answer the question using ONLY the provided patient record context.

PATIENT RECORD CONTEXT:
{ctx_qa}

QUESTION: {user_query}

Provide a structured, accurate answer. Cite the source document for each fact. Use clinical terminology."""

                try:
                    answer = call_groq(prompt_qa, groq_api_key, max_tokens=400)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
                    st.stop()

            st.markdown("---")
            col_a, col_b = st.columns([3, 2])

            with col_a:
                st.markdown("#### 🤖 Answer")
                st.markdown(f"""
                <div style="background:#F8FAFC;border-left:4px solid #0D4F8B;
                            padding:16px 20px;border-radius:0 10px 10px 0;
                            font-size:0.93rem;color:#334155;line-height:1.7;">
                    {answer.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.markdown("#### 📚 Retrieved Sources")
                for r in retrieved_qa:
                    st.markdown(f"""
                    <div class="retrieved-chunk">
                        <div class="chunk-source">
                            📄 {r['doc']} · {r['similarity_score']:.4f}
                        </div>
                        {r['text'][:200]}...
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Pipeline Evaluation — 6 metrics across 2 real HuggingFace datasets")

    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""<div class="metric-card">
            <div class="val">0.808</div>
            <div class="lbl">NER F1 Score</div>
            <div class="sub">NCBI Disease Corpus (fine-tuned)</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown("""<div class="metric-card">
            <div class="val">1.000</div>
            <div class="lbl">Hit Rate@5</div>
            <div class="sub">PubMedQA Retrieval</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown("""<div class="metric-card">
            <div class="val">0.987</div>
            <div class="lbl">MRR</div>
            <div class="sub">PubMedQA Retrieval</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown("""<div class="metric-card">
            <div class="val">100%</div>
            <div class="lbl">Exact Match</div>
            <div class="sub">8 Factual Queries</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### NER Fine-Tuning Results")
        ner_data = pd.DataFrame({
            "Stage": ["Zero-Shot (Before)", "Fine-Tuned (After)", "BioBERT (Reference)"],
            "F1 Score": [0.189, 0.808, 0.890],
            "Precision": [0.0, 0.783, 0.880],
            "Recall": [0.0, 0.835, 0.900]
        })
        st.dataframe(
            ner_data.style.highlight_max(axis=0, subset=["F1 Score"], color="#DCFCE7"),
            use_container_width=True, hide_index=True
        )
        st.caption("Dataset: ncbi/ncbi_disease | Model: d4data/biomedical-ner-all fine-tuned | 5 epochs, A100, fp16")

        st.markdown("#### Generation Quality (PubMedQA)")
        gen_data = pd.DataFrame({
            "Metric": ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"],
            "Score": [0.284, 0.086, 0.206, 0.630],
            "Expected Range": ["0.20–0.35", "0.05–0.15", "0.15–0.30", "0.58–0.72"]
        })
        st.dataframe(gen_data, use_container_width=True, hide_index=True)
        st.caption("10 PubMedQA expert Q&A pairs | Abstractive generation — ROUGE expected lower than extractive")

    with col_right:
        st.markdown("#### Retrieval Quality (PubMedQA — 50 queries)")
        ret_data = pd.DataFrame({
            "Metric": ["Precision@5", "Hit Rate@5", "MRR"],
            "Score": [0.588, 1.000, 0.987],
            "Interpretation": [
                "3/5 retrieved chunks relevant on average",
                "Perfect — never missed a relevant document",
                "Relevant chunk ranked #1 in 49/50 queries"
            ]
        })
        st.dataframe(
            ret_data.style.highlight_max(axis=0, subset=["Score"], color="#DCFCE7"),
            use_container_width=True, hide_index=True
        )

        st.markdown("#### Faithfulness (LLM-as-Judge)")
        faith_data = pd.DataFrame({
            "Verdict": ["Faithful", "Partially Faithful", "Hallucinated"],
            "Count": [2, 2, 1],
            "Percentage": ["40%", "40%", "20%"]
        })
        st.dataframe(faith_data, use_container_width=True, hide_index=True)
        st.caption("OpenBioLLM-8B judging its own answers | Mean faithfulness: 0.600")

        st.markdown("#### Exact Match (8 Factual Queries)")
        em_data = pd.DataFrame({
            "Query": ["HbA1c", "eGFR", "LDL", "Metformin dose", "Glucose", "Creatinine", "Semaglutide", "ACR"],
            "Expected": ["8.1%", "52", "142", "1000mg", "174 mg/dL", "1.4 mg/dL", "0.25mg", "48 mg/g"],
            "Match": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅"]
        })
        st.dataframe(em_data, use_container_width=True, hide_index=True)

    # Dashboard image
    st.markdown("---")
    st.markdown("#### Full Evaluation Dashboard")
    try:
        st.image("eval_outputs/eval_dashboard.png", use_column_width=True)
    except:
        st.info("Upload eval_dashboard.png to eval_outputs/ folder to display here.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### About the Clinical NLP RAG Pipeline")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        ### Pipeline Architecture
        ```
        Clinical Note / Query
               ↓
        Fine-tuned NER
        (arshadnaguru/biomedical-ner-ncbi)
               ↓
        BGE-Large Embeddings
        (BAAI/bge-large-en-v1.5, 1024-dim)
               ↓
        FAISS Vector Index
        (cosine similarity, CPU)
               ↓
        Top-k Chunk Retrieval
               ↓
        Llama-3.1-8B via Groq API
        (RAG-grounded generation)
               ↓
        SOAP Note + ICD-10 Codes
        ```
        """)

    with c2:
        st.markdown("""
        ### Technical Stack
        | Component | Technology |
        |---|---|
        | NER Model | `arshadnaguru/biomedical-ner-ncbi` |
        | Base NER | `d4data/biomedical-ner-all` |
        | Fine-tuning | NCBI Disease Corpus, 5 epochs |
        | Embeddings | `BAAI/bge-large-en-v1.5` |
        | Vector DB | FAISS CPU (IndexFlatIP) |
        | Chunking | Semantic, 256 tok, 64 overlap |
        | LLM | Llama-3.1-8B via Groq |
        | Training HW | Google Colab Pro A100 40GB |
        | Framework | HuggingFace Transformers |

        ### Evaluation Datasets
        | Dataset | Purpose | Source |
        |---|---|---|
        | NCBI Disease Corpus | NER evaluation | `ncbi/ncbi_disease` |
        | PubMedQA | Retrieval + Generation | `pubmed_qa` |
        """)

    st.markdown("---")
    st.markdown("""
    ### About the Author
    **Arshad Naguru** — MS Artificial Intelligence, Rochester Institute of Technology (GPA: 4.0)

    - 🔬 AI Engineer at RIT Office of VP for Research
    - 📄 Published at BEA 2026 Workshop @ ACL (vocabulary difficulty prediction, XLM-RoBERTa)
    - 🏥 Research Assistant — Healthcare AI & Informatics (50K+ patient EHR records)
    - 📧 an2629@rit.edu
    - 🐙 github.com/arshadnaguru
    - 🔗 linkedin.com/in/mahammadarshad

    ---
    *Built for Cotiviti Intern Assessment · Topic 1: Clinical Natural Language Technology · June 2026*
    """)
