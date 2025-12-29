"""
RAG-Based Student Work Auto-Grader (v1.3)
Author: Solution Academy
Features:
- Semantic similarity grading (BGE Large)
- Rubric-based scoring (JSON)
- Grammar feedback
- Streamlit-safe architecture
"""

# =========================
# Standard Library Imports
# =========================
import os
import io
import json
import time
import tempfile
from typing import List, Dict

# =========================
# Third-Party Imports
# =========================
import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional file readers
import docx2txt
import pdfplumber

# =========================
# Cached Resources
# =========================

@st.cache_resource(show_spinner="🔄 Loading embedding model (BGE Large)...")
def load_embedding_model():
    return SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device="cpu"
    )

@st.cache_resource
def load_language_tool():
    try:
        import language_tool_python
        return language_tool_python.LanguageTool("en-US")
    except Exception:
        return None

# =========================
# Utility Functions
# =========================

def read_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            return docx2txt.process(tmp.name)

    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    return ""

def embed_texts(texts: List[str], model) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True)

def semantic_similarity(a: str, b: str, model) -> float:
    emb = embed_texts([a, b], model)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

# =========================
# Rubric Logic
# =========================

def apply_rubric_json(
    student_text: str,
    model_text: str,
    rubric: Dict,
    model
) -> Dict:
    results = {}
    total_score = 0
    max_score = 0

    for criterion, levels in rubric.items():
        max_level_score = max(v["marks"] for v in levels.values())
        max_score += max_level_score

        sims = {}
        for level, details in levels.items():
            sim = semantic_similarity(student_text, details["description"], model)
            sims[level] = (sim, details["marks"])

        best_level = max(sims.items(), key=lambda x: x[1][0])
        score_awarded = best_level[1][1]
        total_score += score_awarded

        results[criterion] = {
            "selected_level": best_level[0],
            "score": score_awarded,
            "max": max_level_score
        }

    return {
        "total": total_score,
        "max": max_score,
        "details": results
    }

# =========================
# Grammar Feedback
# =========================

def grammar_feedback(text: str, tool):
    if not tool or not text.strip():
        return []

    matches = tool.check(text)
    feedback = []
    for m in matches[:10]:
        feedback.append(f"{m.message} → Suggested: {m.replacements[:1]}")
    return feedback

# =========================
# Streamlit App
# =========================

def main():
    st.set_page_config(
        page_title="RAG Auto-Grader v1.3",
        layout="wide"
    )

    st.title("📘 RAG-Based Student Work Auto-Grader (v1.3)")
    st.caption("Semantic grading • Rubrics • Actionable feedback")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_rubric = st.file_uploader(
            "Upload Rubric (JSON)",
            type=["json"]
        )

    # Load Models
    embedding_model = load_embedding_model()
    grammar_tool = load_language_tool()

    # Main UI
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Student Submission")
        student_file = st.file_uploader(
            "Upload student file",
            type=["txt", "docx", "pdf"]
        )
        student_text = read_text_file(student_file)
        st.text_area("Student Text", student_text, height=250)

    with col2:
        st.subheader("📘 Model Answer")
        model_text = st.text_area(
            "Paste model solution",
            height=250
        )

    if st.button("🚀 Start Grading"):
        if not student_text or not model_text:
            st.error("Please provide both student answer and model solution.")
            return

        with st.status("Grading in progress...", expanded=True):
            st.write("🔹 Computing semantic similarity...")
            sim = semantic_similarity(student_text, model_text, embedding_model)
            st.success(f"Semantic Similarity Score: **{sim:.2f}**")

            if uploaded_rubric:
                st.write("🔹 Applying rubric...")
                rubric = json.load(uploaded_rubric)
                rubric_result = apply_rubric_json(
                    student_text,
                    model_text,
                    rubric,
                    embedding_model
                )

                st.subheader("📊 Rubric Breakdown")
                for crit, data in rubric_result["details"].items():
                    st.write(
                        f"**{crit}**: {data['score']} / {data['max']} "
                        f"(Level: {data['selected_level']})"
                    )

                st.success(
                    f"🎯 Final Score: {rubric_result['total']} / {rubric_result['max']}"
                )

            st.write("🔹 Checking grammar...")
            grammar_issues = grammar_feedback(student_text, grammar_tool)
            if grammar_issues:
                st.subheader("✍️ Grammar Feedback")
                for g in grammar_issues:
                    st.write("•", g)
            else:
                st.success("No major grammar issues detected.")

        st.balloons()

# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    main()
