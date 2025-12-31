"""
RAG-Based Student Work Auto-Grader (v1.3)
- Stable Streamlit Deployment
- BAAI/bge-large-en-v1.5 Embeddings
- Human-readable Rubric (NO JSON)
- HCI-friendly UX (Progressive Disclosure)
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# Streamlit Page Config (MUST BE FIRST)
# -------------------------------------------------
st.set_page_config(
    page_title="AI Student Work Auto-Grader",
    page_icon="📘",
    layout="wide"
)

# -------------------------------------------------
# Load Embedding Model (SAFE FOR CLOUD)
# -------------------------------------------------
@st.cache_resource(show_spinner="🔄 Loading AI grading engine (BGE Large)...")
def load_embedding_model():
    return SentenceTransformer(
        "BAAI/bge-large-en-v1.5",
        device="cpu"
    )

embedding_model = load_embedding_model()

# -------------------------------------------------
# Embedding Function (NO BLOCKING / NO FAKE LOOPS)
# -------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    texts = [t if t else "" for t in texts]
    return embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

# -------------------------------------------------
# Rubric Parsing (Teacher-Friendly)
# -------------------------------------------------
def parse_teacher_rubric(rubric_text: str) -> Dict[str, float]:
    """
    Expected format (example):
    Content Quality: 40
    Organization: 25
    Language & Grammar: 20
    Relevance: 15
    """
    rubric = {}
    for line in rubric_text.splitlines():
        if ":" in line:
            try:
                key, value = line.split(":")
                rubric[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return rubric

# -------------------------------------------------
# Core Grading Logic (Semantic Alignment)
# -------------------------------------------------
def grade_student_answer(
    student_answer: str,
    model_answer: str,
    rubric: Dict[str, float]
) -> Dict[str, float]:

    texts = [student_answer, model_answer]
    embeddings = embed_texts(texts)

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    results = {}
    for criterion, weight in rubric.items():
        results[criterion] = round(similarity * weight, 2)

    results["Overall Similarity"] = round(similarity * 100, 2)
    results["Total Score"] = round(sum(results.values()), 2)

    return results

# -------------------------------------------------
# UI HEADER
# -------------------------------------------------
st.title("📘 RAG-Based Student Work Auto-Grader")
st.markdown(
    """
    **Pedagogically grounded, AI-assisted grading tool**  
    Uses semantic similarity — *not generative hallucination*.
    """
)

st.divider()

# -------------------------------------------------
# Progressive Disclosure UI
# -------------------------------------------------
with st.expander("🧑‍🏫 Step 1: Provide Model Answer & Rubric", expanded=True):
    model_answer = st.text_area(
        "📌 Model / Ideal Answer",
        height=180,
        placeholder="Paste the ideal answer here..."
    )

    rubric_text = st.text_area(
        "📋 Grading Rubric (Criterion : Weight)",
        height=150,
        placeholder="Content Quality: 40\nOrganization: 25\nLanguage & Grammar: 20\nRelevance: 15"
    )

with st.expander("👩‍🎓 Step 2: Provide Student Answer", expanded=True):
    student_answer = st.text_area(
        "✍️ Student Answer",
        height=220,
        placeholder="Paste the student response here..."
    )

st.divider()

# -------------------------------------------------
# Grading Action
# -------------------------------------------------
if st.button("🧠 Grade Student Answer", use_container_width=True):

    if not model_answer or not student_answer or not rubric_text:
        st.error("❌ Please provide model answer, student answer, and rubric.")
    else:
        with st.spinner("🔍 Analyzing semantic alignment..."):
            rubric = parse_teacher_rubric(rubric_text)

            if not rubric:
                st.error("❌ Rubric format invalid. Use 'Criterion: Weight'.")
            else:
                results = grade_student_answer(
                    student_answer,
                    model_answer,
                    rubric
                )

                st.success("✅ Grading Completed")

                # -------------------------------------------------
                # Results Display
                # -------------------------------------------------
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Criterion-wise Scores")
                    df = pd.DataFrame(
                        list(results.items()),
                        columns=["Criterion", "Score"]
                    )
                    st.dataframe(df, use_container_width=True)

                with col2:
                    st.subheader("🎯 Summary")
                    st.metric(
                        "Overall Semantic Similarity",
                        f"{results['Overall Similarity']}%"
                    )
                    st.metric(
                        "Total Weighted Score",
                        results["Total Score"]
                    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()
st.caption(
    "🔐 Academic Integrity Preserved | "
    "⚙️ Embedding-based Evaluation | "
    "🚫 No Generative Hallucination"
)
