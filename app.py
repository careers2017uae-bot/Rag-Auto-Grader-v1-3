# ============================================================
# RAG-based Student Work Auto-Grader
# Version 1.3 (Stable, Deployable)
# UI + Rubric IDENTICAL to v1.2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Optional imports (safe guards)
# ------------------------------
try:
    import pdfplumber
except:
    pdfplumber = None

try:
    import docx2txt
except:
    docx2txt = None

try:
    import language_tool_python
    grammar_tool = language_tool_python.LanguageTool('en-US')
except:
    grammar_tool = None

from sentence_transformers import SentenceTransformer

# ------------------------------
# Streamlit Config (UNCHANGED)
# ------------------------------
st.set_page_config(
    page_title="AI Student Work Auto-Grader",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📘 AI-Powered Student Work Auto-Grader")

# ------------------------------
# Load Embedding Model (STABLE)
# ------------------------------
@st.cache_resource(show_spinner="Loading semantic model...")
def load_embedding_model():
    return SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device="cpu"
    )

embedder = load_embedding_model()

# ------------------------------
# Helper Functions
# ------------------------------
def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""

    suffix = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == "pdf" and pdfplumber:
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text

        elif suffix in ["docx", "doc"] and docx2txt:
            return docx2txt.process(tmp_path)

        elif suffix == "txt":
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        else:
            return ""

    except Exception:
        return ""

def embed_texts(texts):
    if isinstance(texts, str):
        texts = [texts]
    return embedder.encode(texts, normalize_embeddings=True)

def similarity_score(student, model):
    emb_student = embed_texts(student)
    emb_model = embed_texts(model)
    return float(cosine_similarity(emb_student, emb_model)[0][0])

def grammar_check(text):
    if grammar_tool is None or not text.strip():
        return 0, []
    matches = grammar_tool.check(text)
    return len(matches), matches

# ------------------------------
# Rubric (IDENTICAL TO v1.2)
# ------------------------------
DEFAULT_RUBRIC = {
    "Relevance": {
        "weight": 0.25,
        "description": "Alignment with the given question or task."
    },
    "Conceptual Accuracy": {
        "weight": 0.25,
        "description": "Correctness of concepts and explanations."
    },
    "Depth & Clarity": {
        "weight": 0.20,
        "description": "Explanation depth, clarity, and coherence."
    },
    "Structure & Organization": {
        "weight": 0.15,
        "description": "Logical flow, formatting, and readability."
    },
    "Language & Grammar": {
        "weight": 0.15,
        "description": "Grammar, spelling, and language quality."
    }
}

def apply_rubric(student_text, model_text, rubric):
    base_similarity = similarity_score(student_text, model_text)

    grammar_errors, _ = grammar_check(student_text)
    grammar_score = max(0, 1 - grammar_errors * 0.05)

    results = {}
    total_score = 0

    for criterion, meta in rubric.items():
        if criterion == "Language & Grammar":
            score = grammar_score
        else:
            score = base_similarity

        weighted = score * meta["weight"]
        results[criterion] = {
            "raw_score": round(score, 2),
            "weight": meta["weight"],
            "weighted_score": round(weighted, 2),
            "description": meta["description"]
        }
        total_score += weighted

    return round(total_score * 100, 2), results

# ------------------------------
# Sidebar (UNCHANGED)
# ------------------------------
st.sidebar.header("📂 Upload Files")

student_file = st.sidebar.file_uploader(
    "Upload Student Submission",
    type=["pdf", "docx", "txt"]
)

model_file = st.sidebar.file_uploader(
    "Upload Model Solution",
    type=["pdf", "docx", "txt"]
)

use_default_rubric = st.sidebar.checkbox("Use Default Rubric", value=True)

if not use_default_rubric:
    rubric_json = st.sidebar.text_area(
        "Paste Custom Rubric JSON",
        value=json.dumps(DEFAULT_RUBRIC, indent=2),
        height=300
    )
    try:
        rubric = json.loads(rubric_json)
    except:
        rubric = DEFAULT_RUBRIC
else:
    rubric = DEFAULT_RUBRIC

# ------------------------------
# Main Panel
# ------------------------------
if st.button("🚀 Grade Submission"):

    with st.spinner("Reading files..."):
        student_text = read_uploaded_file(student_file)
        model_text = read_uploaded_file(model_file)

    if not student_text or not model_text:
        st.error("❌ Please upload BOTH student submission and model solution.")
    else:
        with st.spinner("Evaluating using semantic analysis..."):
            final_score, rubric_results = apply_rubric(
                student_text,
                model_text,
                rubric
            )

        st.success(f"✅ Final Score: **{final_score}%**")

        st.subheader("📊 Rubric Breakdown")

        rubric_df = pd.DataFrame([
            {
                "Criterion": k,
                "Raw Score": v["raw_score"],
                "Weight": v["weight"],
                "Weighted Score": v["weighted_score"]
            }
            for k, v in rubric_results.items()
        ])

        st.dataframe(rubric_df, use_container_width=True)

        with st.expander("🧠 Detailed Feedback"):
            for k, v in rubric_results.items():
                st.markdown(f"**{k}**")
                st.markdown(f"- Description: {v['description']}")
                st.markdown(f"- Raw Score: {v['raw_score']}")
                st.markdown(f"- Weighted Contribution: {v['weighted_score']}")
                st.markdown("---")

        st.subheader("📄 Student Submission Preview")
        st.text_area("", student_text, height=250)

        st.subheader("📄 Model Solution Preview")
        st.text_area("", model_text, height=250)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("© ai!nfluence — RAG-based Academic Evaluation System")
