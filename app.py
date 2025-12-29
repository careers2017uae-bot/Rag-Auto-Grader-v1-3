# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit)
Embedding upgrade:
- all-MiniLM-L6-v2 ❌
- BAAI/bge-large-en-v1.5 ✅
- Optional Jina AI (multilingual, free tier) ✅
"""

# ==================== STANDARD IMPORTS ====================
import os
import io
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ==================== OPTIONAL FILE SUPPORT ====================
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
except Exception:
    lang_tool = None

# ==================== PAGE CONFIG (HCI) ====================
st.set_page_config(
    page_title="📚 RAG-Based Intelligent Auto-Grader",
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# ==================== EMBEDDING CONFIG ====================
DEFAULT_EMBEDDING = "bge"   # bge | jina
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# ==================== LOAD LOCAL EMBEDDING MODEL ====================
@st.cache_resource(show_spinner="🔄 Loading semantic grading engine...")
def load_bge_model():
    return SentenceTransformer(BGE_MODEL_NAME)

bge_model = load_bge_model()

# ==================== EMBEDDING ABSTRACTION (CRITICAL) ====================
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Unified embedding interface.
    - Uses BGE locally by default
    - Switches to Jina if API key is present
    """
    texts = [t or "" for t in texts]

    use_jina = bool(os.getenv("JINA_API_KEY"))
    provider = "Jina AI (Multilingual)" if use_jina else "BGE Large v1.5 (Local)"

    with st.status(f"🔍 Generating embeddings ({provider})...", state="running"):
        if use_jina:
            return embed_with_jina(texts)
        return embed_with_bge(texts)

def embed_with_bge(texts: List[str]) -> np.ndarray:
    vectors = bge_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return vectors

def embed_with_jina(texts: List[str]) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "jina-embeddings-v2-base-en",
        "input": texts
    }

    resp = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()["data"]
    vectors = np.array([d["embedding"] for d in data], dtype=np.float32)

    # Normalize for cosine similarity consistency
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)

# ==================== COSINE SIM ====================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

# ==================== GRAMMAR CHECK ====================
def grammar_check(text: str) -> Dict[str, Any]:
    if not lang_tool or not text.strip():
        return {"available": False, "issues_count": 0, "examples": []}

    matches = lang_tool.check(text)
    examples = []

    for m in matches[:6]:
        examples.append({
            "message": m.message,
            "context": text[max(0, m.offset-30): m.offset+30],
            "suggestions": m.replacements[:3]
        })

    return {
        "available": True,
        "issues_count": len(matches),
        "examples": examples
    }

# ==================== HEURISTIC GRADING (UNCHANGED LOGIC) ====================
def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    vecs = embed_texts([model_ans, student_ans])
    sim = cosine_sim(vecs[0], vecs[1])
    sim_norm = max(0.0, min((sim + 1) / 2, 1.0))

    grammar = grammar_check(student_ans)
    penalty = min(40.0, grammar["issues_count"] * 1.5) if grammar["available"] else 0

    final = round(sim_norm * 100 - penalty, 2)

    return {
        "final_score": max(0, final),
        "similarity": sim_norm,
        "grammar": grammar,
        "grading_method": "heuristic",
        "breakdown": [
            {"criterion": "Content Similarity", "subscore": round(sim_norm*100, 2)},
            {"criterion": "Grammar", "subscore": round(100 - penalty, 2)}
        ]
    }

# ==================== UI NOTE ====================
# 🔹 Everything else (rubric parsing, Streamlit UI, analytics, export,
#     Groq feedback, progress bars, HCI CSS) remains EXACTLY the same.
# 🔹 No further changes required.

