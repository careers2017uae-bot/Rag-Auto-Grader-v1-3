# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit) - Enhanced UX Version
Applying HCI Principles: Progressive Disclosure, Immediate Feedback, Clear Affordances, etc.
"""
#import pandas as pd
#from io import BytesIO

# Optional PDF import
try:
   # import pdfplumber
except Exception:
    pdfplumber = None

#import os
#import json
#import time
#import streamlit as st
from typing import List, Dict, Any, Optional
#import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime

# Optional imports
try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
except Exception:
    lang_tool = None

# ==================== HCI ENHANCEMENTS ====================
st.set_page_config(
    page_title="RAG-Based Intelligent Auto-Grader", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.4rem !important;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ffa500, #ffd93d, #6bcf7f);
        transition: width 0.5s ease-in-out;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .feedback-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
    }
    .grammar-issue {
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load embedding model once
@st.cache_resource(show_spinner="🔄 Loading AI grading engine...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ---------------------------
# Enhanced Utilities with Progress Indicators
# ---------------------------

# Replace the entire parse_teacher_rubric function with this improved version:
def parse_teacher_rubric(text: str) -> Optional[dict]:
    """
    Convert teacher-friendly rubric table into internal rubric JSON.
    Expected format (flexible):
    Criterion | Weight | Description
    or
    Criterion, Weight, Description
    """
    if not text or not text.strip():
        return None
    
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    
    criteria = []
    line_count = 0
    
    # Try different delimiters
    for line in lines:
        line_count += 1
        
        # Skip header lines that don't contain numbers
        if line_count == 1:
            # Check if this looks like a header (contains words like criterion, weight, description)
            header_words = ['criterion', 'weight', 'description', 'criteria', 'score', 'points']
            if any(word in line.lower() for word in header_words):
                continue
        
        # Try pipe delimiter first, then comma, then tab
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
        elif ',' in line:
            parts = [p.strip() for p in line.split(',')]
        elif '\t' in line:
            parts = [p.strip() for p in line.split('\t')]
        else:
            # Try splitting by 2 or more spaces
            parts = [p.strip() for p in line.split('  ') if p.strip()]
        
        if len(parts) < 2:
            continue
        
        name = parts[0]
        if not name:
            continue
        
        # Extract weight - try to find a number in the parts
        weight = 0
        for part in parts[1:]:
            # Look for numbers in the part
            try:
                # Remove any non-numeric characters except decimal point
                import re
                weight_str = re.sub(r'[^\d.]', '', part)
                if weight_str:
                    weight = float(weight_str)
                    break
            except:
                continue
        
        # Determine criterion type
        criterion_type = "grammar_penalty" if "grammar" in name.lower() else "similarity"
        
        criteria.append({
            "name": name,
            "weight": weight,
            "type": criterion_type,
            "penalty_per_issue": 1.5 if "grammar" in name.lower() else 0
        })
    
    if not criteria:
        return None
    
    # Normalize weights so they sum to 1.0
    total_weight = sum(c.get("weight", 0) for c in criteria)
    if total_weight > 0:
        for criterion in criteria:
            criterion["weight"] = criterion["weight"] / total_weight
    
    return {"criteria": criteria}

# Also update the rubric parsing in the main execution section (around line 381):
# Replace the rubric processing section with this:

    # Process rubric with better error handling
    rubric_obj = None
    rubric_text = rubric_text_paste.strip()
    if not rubric_text and rubric_file:
        rubric_text = read_text_file(rubric_file)
    
    if rubric_text:
        with st.status("📊 Parsing rubric...", state="running") as status:
            rubric_obj = parse_teacher_rubric(rubbaric_text)
            if rubric_obj:
                # Show parsed rubric for verification
                st.session_state.parsed_rubric = rubric_obj
                status.update(label=f"✅ Rubric parsed: {len(rubric_obj.get('criteria', []))} criteria", state="complete")
                
                # Show the parsed rubric in an expander
                with st.expander("📋 View Parsed Rubric", expanded=False):
                    for i, criterion in enumerate(rubric_obj.get('criteria', [])):
                        st.write(f"**{i+1}. {criterion['name']}**")
                        st.write(f"   Weight: {criterion['weight']:.2%}")
                        st.write(f"   Type: {criterion['type']}")
                        if criterion['type'] == 'grammar_penalty':
                            st.write(f"   Penalty per issue: {criterion.get('penalty_per_issue', 1.5)} points")
                        st.write("---")
            else:
                status.update(label="⚠️ Could not parse rubric. Using default grading.", state="error")
                st.warning("Could not parse the rubric. Please check the format and try again.")

# Update the apply_rubric_json function with better logic:
def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    """Apply rubric-based grading with proper weight distribution."""
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans)
    
    with st.status("📊 Applying rubric criteria...", state="running") as status:
        # Calculate similarity
        vecs = embed_texts([model_ans, student_ans])
        similarity_score = cosine_sim(vecs[0], vecs[1])
        similarity_percent = max(0.0, min((similarity_score + 1) / 2.0, 1.0)) * 100
        
        # Grammar check
        grammar_result = grammar_check(student_ans)
        grammar_issues = grammar_result.get("issues_count", 0) if grammar_result.get("available") else 0
        
        total_score = 0.0
        breakdown = []
        
        for i, criterion in enumerate(criteria):
            name = criterion.get("name", f"Criterion {i+1}")
            weight = criterion.get("weight", 0)
            criterion_type = criterion.get("type", "similarity")
            penalty_per_issue = criterion.get("penalty_per_issue", 1.5)
            
            subscore = 0.0
            
            if criterion_type == "similarity":
                subscore = similarity_percent
            elif criterion_type == "grammar_penalty":
                if grammar_result.get("available"):
                    # Apply penalty: start from 100 and subtract for each issue
                    subscore = max(0.0, 100.0 - (grammar_issues * penalty_per_issue))
                else:
                    subscore = 100.0  # Full points if grammar check not available
            else:
                # Default to similarity for unknown types
                subscore = similarity_percent
            
            # Apply weighting
            weighted_score = subscore * weight
            total_score += weighted_score
            
            breakdown.append({
                "criterion": name,
                "weight": round(weight, 4),
                "subscore": round(subscore, 2),
                "type": criterion_type,
                "weighted_score": round(weighted_score, 2)
            })
        
        # Ensure total_score doesn't exceed 100
        total_score = min(100.0, total_score)
        
        status.update(label="✅ Rubric applied successfully", state="complete")
    
    return {
        "final_score": round(total_score, 2),
        "breakdown": breakdown,
        "similarity": similarity_percent / 100.0,
        "grammar": grammar_result,
        "grading_method": "rubric"
    }




def export_results_to_excel(results: list) -> bytes:
    """
    Convert grading results into a multi-sheet Excel file.
    """
    summary_rows = []
    breakdown_rows = []
    grammar_rows = []

    for r in results:
        # -------- Summary Sheet --------
        summary_rows.append({
            "Student Name": r.get("name"),
            "Final Score": r.get("final_score"),
            "Similarity (%)": round(r.get("details", {}).get("similarity", 0) * 100, 2),
            "Grammar Issues": r.get("details", {}).get("grammar", {}).get("issues_count"),
            "Grading Method": r.get("details", {}).get("grading_method"),
            "Timestamp": r.get("timestamp")
        })

        # -------- Breakdown Sheet --------
        for b in r.get("details", {}).get("breakdown", []):
            breakdown_rows.append({
                "Student Name": r.get("name"),
                "Criterion": b.get("criterion"),
                "Weight": b.get("weight"),
                "Subscore": b.get("subscore"),
                "Type": b.get("type")
            })

        # -------- Grammar Sheet --------
        grammar = r.get("details", {}).get("grammar", {})
        if grammar.get("available"):
            for g in grammar.get("examples", []):
                grammar_rows.append({
                    "Student Name": r.get("name"),
                    "Issue": g.get("message"),
                    "Context": g.get("context"),
                    "Suggestions": ", ".join(g.get("suggestions", []))
                })

    # Create DataFrames
    df_summary = pd.DataFrame(summary_rows)
    df_breakdown = pd.DataFrame(breakdown_rows)
    df_grammar = pd.DataFrame(grammar_rows)

    # Write to Excel in-memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_breakdown.to_excel(writer, index=False, sheet_name="Score Breakdown")
        if not df_grammar.empty:
            df_grammar.to_excel(writer, index=False, sheet_name="Grammar Issues")

    return output.getvalue()

def read_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    
    with st.status(f"📄 Processing {uploaded_file.name}...", state="running") as status:
        try:
            content = uploaded_file.getvalue()
            name = uploaded_file.name.lower()
            result = ""

            if name.endswith(".txt"):
                result = content.decode("utf-8")

            elif name.endswith(".docx"):
                if docx2txt:
                    tmp_path = f"/tmp/temp_{int(time.time())}.docx"
                    with open(tmp_path, "wb") as f:
                        f.write(content)
                    result = docx2txt.process(tmp_path)
                else:
                    st.warning("📝 docx2txt not installed")

            elif name.endswith(".pdf"):
                if pdfplumber:
                    tmp_path = f"/tmp/temp_{int(time.time())}.pdf"
                    with open(tmp_path, "wb") as f:
                        f.write(content)

                    text_pages = []
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_pages.append(page_text)

                    result = "\n".join(text_pages)
                else:
                    st.warning("📄 pdfplumber not installed; run `pip install pdfplumber`")

            else:
                result = content.decode("utf-8", errors="ignore")

            status.update(label=f"✅ Processed {uploaded_file.name}", state="complete")
            return result.strip()

        except Exception as e:
            status.update(label=f"❌ Error processing {uploaded_file.name}", state="error")
            st.error(str(e))
            return ""


def safe_load_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception as e:
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    texts = [t if t is not None else "" for t in texts]
    
    # Show embedding progress
    progress_text = "🔍 Analyzing text similarities..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(100):
        time.sleep(0.01)  # Simulate progress
        my_bar.progress(i + 1, text=progress_text)
    
    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    my_bar.empty()
    
    return vectors

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def grammar_check(text: str) -> Dict[str, Any]:
    if not lang_tool or not text.strip():
        return {"available": False, "issues_count": 0, "examples": []}
    
    with st.status("🔍 Checking grammar and spelling...", state="running") as status:
        matches = lang_tool.check(text)
        examples = []
        for m in matches[:6]:  # Show top 6 issues
            context = text[max(0, m.offset-30): m.offset+30]
            examples.append({
                "message": m.message, 
                "context": context,
                "suggestions": m.replacements[:3]  # Top 3 suggestions
            })
        status.update(label=f"✅ Found {len(matches)} grammar issues", state="complete")
        
    return {"available": True, "issues_count": len(matches), "examples": examples}

# ---------------------------
# Enhanced Grading Logic with Visual Feedback
# ---------------------------
def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str) -> Dict[str, Any]:
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans)

    # Show grading progress
    with st.status("📊 Applying rubric criteria...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        g = grammar_check(student_ans)
        issues = g["issues_count"] if g.get("available") else None

        total_weight = sum(c.get("weight", 0) for c in criteria) or 1.0
        total_score = 0.0
        breakdown = []
        
        for i, c in enumerate(criteria):
            name = c.get("name", f"Criterion {i+1}")
            w = c.get("weight", 0) / total_weight
            t = c.get("type", "similarity")
            subscore = 0.0
            
            if t == "similarity":
                subscore = sim_norm * 100
            elif t == "grammar_penalty":
                if issues is None:
                    subscore = 100.0
                else:
                    penalty_per = c.get("penalty_per_issue", 1.5)
                    subscore = max(0.0, 100.0 - penalty_per * issues)
            else:
                subscore = sim_norm * 100
                
            total_score += subscore * w
            breakdown.append({
                "criterion": name, 
                "weight": round(w,3), 
                "subscore": round(subscore,2),
                "type": t
            })
        
        status.update(label="✅ Rubric applied successfully", state="complete")

    final_score = round(total_score, 2)
    return {
        "final_score": final_score, 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g,
        "grading_method": "rubric"
    }

def heuristic_grade(model_ans: str, student_ans: str) -> Dict[str, Any]:
    with st.status("🎯 Computing similarity scores...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        base = sim_norm * 100
        g = grammar_check(student_ans)
        penalty = 0.0
        
        if g.get("available"):
            issues = g["issues_count"]
            penalty = min(40.0, issues * 1.5)
            
        final = round(max(0.0, base - penalty), 2)
        breakdown = [
            {"criterion": "Content Similarity", "weight": 0.8, "subscore": round(base,2), "type": "similarity"},
            {"criterion": "Grammar & Mechanics", "weight": 0.2, "subscore": round(max(0, 100 - penalty),2), "type": "grammar"}
        ]
        status.update(label="✅ Automatic grading completed", state="complete")
        
    return {
        "final_score": final, 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g, 
        "penalty": penalty,
        "grading_method": "heuristic"
    }

# ---------------------------
# Enhanced Groq Integration
# ---------------------------
def generate_feedback_with_groq(prompt_text: str) -> Optional[str]:
    base = os.getenv("GROQ_API_URL")
    key = os.getenv("GROQ_API_KEY")
    if not base or not key:
        return None
        
    with st.status("🤖 Generating AI feedback...", state="running") as status:
        if base.endswith("/"):
            url = base + "chat/completions"
        else:
            url = base + "/chat/completions"

        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are an objective grading assistant. Provide constructive, actionable feedback."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and len(choices) > 0:
                    msg = choices[0].get("message", {}).get("content")
                    status.update(label="✅ AI feedback generated", state="complete")
                    return msg
            else:
                status.update(label="🔶 AI feedback possible", state="error") #Label changed
            return None
        except Exception as e:
            status.update(label="🔶 AI feedback possible", state="error") #Label changed
            return None

# ---------------------------
# Enhanced Streamlit UI with HCI Principles
# ---------------------------
st.markdown('<div class="main-header">📚 RAG-Based Intelligent Auto-Grader</div>', unsafe_allow_html=True)

# Sidebar with clear information hierarchy
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Grading scale with better visual cues
    output_scale = st.selectbox(
        "**Grading Scale**", 
        ["numeric_100", "ielts_band_0-9"], 
        index=0,
        help="Select the scoring system for evaluations"
    )
    
    # Toggle options with icons
    st.markdown("### 👁️ Display Options")
    show_grammar_examples = st.toggle("Show grammar examples", value=True)
    show_detailed_breakdown = st.toggle("Show detailed score breakdown", value=True)
    enable_ai_feedback = st.toggle("Enable AI feedback (if available)", value=True)
    
    # System status
    st.markdown("### 🔍 System Status")
    st.success("✅ Embedding model loaded")
    st.info(f"📊 Grammar checking: {'✅ Available' if lang_tool else  '🔶 Basic checking available'}")
    # In the sidebar section, replace the grammar checking status line:
    #st.info(f"📊 Grammar checking: {'✅ Available' if lang_tool else '🔶 Basic checking available'}")
    st.info(f"🤖 AI feedback: {'✅ Available' if os.getenv('GROQ_API_KEY') else '❌ Not configured'}")
    
    # Quick tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - **Upload** or **paste** content - both work!
        - Use **rubrics** for consistent grading
        - Check **grammar feedback** for writing improvements
        - **Multiple students** can be graded at once
        """)

# Main content with tabbed interface
tab1, tab2, tab3 = st.tabs(["📥 Input Materials", "🎯 Grading Results", "📈 Analytics"])

with tab1:
    st.markdown('<div class="sub-header">📥 Input Materials</div>', unsafe_allow_html=True)
    
    # Use columns for better information density
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Exercise Description")
        ex_file = st.file_uploader(
            "Upload exercise document", 
            type=["txt","docx", "pdf"],
            help="Upload a .txt or .docx file with the exercise instructions"
        )
        ex_text_paste = st.text_area(
            "Or paste exercise description here", 
            height=120,
            placeholder="Paste the exercise description, prompt, or question here...",
            help="You can either upload a file or paste text directly"
        )
        
        st.markdown("#### 📝 Student Submissions")
        student_files = st.file_uploader(
            "Upload student files", 
            accept_multiple_files=True, 
            type=["txt","docx", "pdf"],
            help="Upload multiple student submissions at once"
        )
        student_paste = st.text_area(
            "Or paste student submissions", 
            height=150,
            placeholder="Paste student answers here. Separate different submissions with '---' on a new line.",
            help="Use '---' on a separate line to distinguish between different student submissions"
        )

    with col2:
        st.markdown("#### 📖 Model Solution")
        model_file = st.file_uploader(
            "Upload model solution",
            type=["txt","docx","pdf"],
            help="The ideal answer or reference solution for comparison"
        )
        model_text_paste = st.text_area(
            "Or paste model solution here",
            height=120,
            placeholder="Paste the model answer or ideal solution here..."
        )
    
        st.markdown("#### 📊 Grading Rubric (Optional)")
    
        with st.expander("📝 How to format your rubric"):
            st.markdown("""
    **Pipe-separated format:**
            Criterion | Weight | Description
            Content Accuracy | 50 | Covers key concepts correctly
            Organization | 30 | Logical structure and flow
            Grammar | 20 | Grammar, spelling, clarity
    
    **Comma-separated format also works.**
    Weights are auto-normalized.
            """)
    
        rubric_file = st.file_uploader(
            "Upload rubric file (TXT, DOCX, PDF)",
            type=["txt", "docx", "pdf"]
        )
    
        rubric_text_paste = st.text_area(
            "Or paste rubric here",
            height=160,
            value="""Criterion | Weight | Description
    Content Accuracy | 50 | Covers key concepts correctly
    Organization | 30 | Logical structure and flow
    Grammar | 20 | Grammar, spelling, sentence clarity"""
        )
    
        if st.button("🔍 Preview Rubric", type="secondary"):
            parsed = parse_teacher_rubric(rubric_text_paste.strip())
            if parsed:
                st.success("✅ Rubric parsed successfully")
                st.json(parsed)
            else:
                st.error("❌ Could not parse rubric")

        

    
    # Action button with clear visual hierarchy
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        grade_button = st.button(
            "🚀 Start Grading Process", 
            type="primary", 
            use_container_width=True,
            help="Click to begin grading all student submissions"
        )

with tab2:
    st.markdown('<div class="sub-header">🎯 Grading Results</div>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = []
        st.info("👆 Start by uploading materials and clicking 'Start Grading Process' in the Input tab.")
    else:
        for i, r in enumerate(st.session_state.results):
            with st.container():
                # Header with student name and score
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### 👨‍🎓 {r.get('name', 'Student')}")
                with col2:
                    score = r.get('final_score', 0)
                    score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{score}/100</h2>", unsafe_allow_html=True)
                
                # Progress bar visualization
                st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(score), unsafe_allow_html=True)
                
                # Score interpretation
                if score >= 80:
                    st.markdown('<div class="success-box">🎉 Excellent work! Strong understanding demonstrated.</div>', unsafe_allow_html=True)
                elif score >= 60:
                    st.markdown('<div class="warning-box">📚 Good effort, with room for improvement in key areas.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">💡 Needs significant improvement. Review fundamental concepts.</div>', unsafe_allow_html=True)
                
                # Detailed feedback in expanders
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📋 Detailed Feedback", expanded=True):
                        st.markdown("**Key Observations:**")
                        st.write(r["reasoning"])
                        
                        st.markdown("**Actionable Steps:**")
                        for line in r["feedback_lines"]:
                            st.markdown(f'<div class="feedback-item">💡 {line}</div>', unsafe_allow_html=True)
                        
                        if r.get("groq_feedback") and enable_ai_feedback:
                            st.markdown("**🤖 AI Insights:**")
                            st.write(r["groq_feedback"])
                
                with col2:
                    if show_detailed_breakdown:
                        with st.expander("📊 Score Breakdown", expanded=True):
                            for item in r["details"].get("breakdown", []):
                                col_a, col_b, col_c = st.columns([3, 1, 1])
                                with col_a:
                                    st.write(f"**{item['criterion']}**")
                                with col_b:
                                    st.write(f"{item['subscore']:.1f}")
                                with col_c:
                                    progress = item['subscore'] / 100
                                    st.progress(progress)
                    
                    if show_grammar_examples and r["details"].get("grammar", {}).get("available"):
                        with st.expander("🔍 Grammar Check", expanded=False):
                            g = r["details"]["grammar"]
                            st.write(f"**Issues found:** {g['issues_count']}")
                            for ex in g["examples"]:
                                st.markdown(f"""
                                <div class="grammar-issue">
                                    <strong>⚠️ {ex['message']}</strong><br>
                                    <em>Context:</em> ...{ex['context']}...<br>
                                    {f"<em>Suggestions:</em> {', '.join(ex.get('suggestions', []))}" if ex.get('suggestions') else ""}
                                </div>
                                """, unsafe_allow_html=True)
                
                st.divider()

with tab3:
    st.markdown('<div class="sub-header">📈 Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.info("No grading data available. Complete a grading session first.")
    else:
        # Basic analytics
        scores = [r.get('final_score', 0) for r in st.session_state.results if r.get('final_score') is not None]
        
        if scores:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Score", f"{np.mean(scores):.1f}")
            with col2:
                st.metric("Highest Score", f"{max(scores):.1f}")
            with col3:
                st.metric("Lowest Score", f"{min(scores):.1f}")
            with col4:
                st.metric("Students Graded", len(scores))
            
            # Score distribution
            st.markdown("#### Score Distribution")
            hist_values = np.histogram(scores, bins=10, range=(0, 100))[0]
            st.bar_chart(hist_values)
            
            # Export results
            st.markdown("#### Export Results")
            if st.button("📤 Export Results as Excel"):
                excel_bytes = export_results_to_excel(st.session_state.results)
            
                st.download_button(
                    label="⬇️ Download Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


# ==================== GRADING EXECUTION ====================
if grade_button:
    # Input validation with clear feedback
    exercise_text = ex_text_paste.strip() if ex_text_paste.strip() else read_text_file(ex_file)
    model_text = model_text_paste.strip() if model_text_paste.strip() else read_text_file(model_file)
    
    if not exercise_text:
        st.error("❌ Please provide the exercise description (upload or paste).")
        st.stop()
    
    if not model_text:
        st.error("❌ Please provide the model solution (upload or paste).")
        st.stop()
    
    # Process rubric
    rubric_obj = None

    rubric_text = rubric_text_paste.strip()
    if not rubric_text and rubric_file:
        rubric_text = read_text_file(rubric_file)
    
    if rubric_text:
        rubric_obj = parse_teacher_rubric(rubric_text)
        if rubric_obj is None:
            st.warning("⚠️ Rubric detected but could not be parsed. Default grading will be used.")

    
    # Process student submissions
    student_texts = []
    student_names = []
    
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt.strip():
                student_texts.append(txt.strip())
                student_names.append(f.name)
    
    if student_paste.strip():
        parts = [p.strip() for p in student_paste.split("\n---\n") if p.strip()]
        for i, p in enumerate(parts):
            student_texts.append(p)
            student_names.append(f"Student_{i+1}")
    
    if not student_texts:
        st.error("❌ No student submissions provided. Please upload files or paste submissions.")
        st.stop()
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'grading_complete' not in st.session_state:
        st.session_state.grading_complete = False
    
    # Grade submissions with progress tracking
    st.session_state.results = []
    progress_bar = st.progress(0, text=f"Grading 0/{len(student_texts)} students...")
    
    for idx, (s_text, s_name) in enumerate(zip(student_texts, student_names)):
        progress = (idx) / len(student_texts)
        progress_bar.progress(progress, text=f"Grading {idx+1}/{len(student_texts)}: {s_name}...")
        
        try:
            if rubric_obj:
                res = apply_rubric_json(rubric_obj, model_text, s_text)
            else:
                res = heuristic_grade(model_text, s_text)
            
            # Enhanced feedback generation
            sim_pct = round(res.get("similarity", 0) * 100, 2)
            issues = res.get("grammar", {}).get("issues_count", "N/A")
            reasoning = f"**Similarity to model answer:** {sim_pct}% | **Grammar issues:** {issues}"
            
            # Contextual feedback lines
            feedback_lines = []
            similarity = res.get("similarity", 0)
            
            if similarity >= 0.75:
                feedback_lines.append("Excellent content coverage and task achievement")
                feedback_lines.append("Well-structured response with clear organization")
            elif similarity >= 0.5:
                feedback_lines.append("Good content coverage with some minor gaps")
                feedback_lines.append("Consider expanding on key points for better depth")
            else:
                feedback_lines.append("Significant content gaps - review core concepts")
                feedback_lines.append("Focus on addressing all parts of the prompt")
            
            # Grammar-specific feedback
            if res.get("grammar", {}).get("available"):
                issues_count = res["grammar"]["issues_count"]
                if issues_count > 10:
                    feedback_lines.append("High number of grammar errors affecting readability")
                elif issues_count > 5:
                    feedback_lines.append("Moderate grammar issues - proofreading recommended")
                elif issues_count > 0:
                    feedback_lines.append("Minor grammar issues present")
                else:
                    feedback_lines.append("Excellent grammar and mechanics")
            
            # Generate AI feedback if enabled
            groq_feedback = None
            if enable_ai_feedback:
                groq_prompt = f"""
                Provide constructive feedback for this student work:
                
                Exercise: {exercise_text}
                Model Answer: {model_text}
                Student Answer: {s_text}
                Current Score: {res.get('final_score')}/100
                
                Provide 2-3 specific, actionable suggestions for improvement.
                """
                groq_feedback = generate_feedback_with_groq(groq_prompt)
            
            st.session_state.results.append({
                "name": s_name,
                "final_score": res.get("final_score"),
                "reasoning": reasoning,
                "feedback_lines": feedback_lines,
                "details": res,
                "groq_feedback": groq_feedback,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            st.session_state.results.append({
                "name": s_name, 
                "error": f"Grading failed: {str(e)}"
            })
    
    progress_bar.progress(1.0, text=f"✅ Completed grading {len(student_texts)} students!")
    time.sleep(0.5)
    progress_bar.empty()
    
    # Set grading complete flag
    st.session_state.grading_complete = True
    
    # Force display of results
    st.success(f"🎉 Successfully graded {len(student_texts)} submissions!")
    
    # Use rerun instead of switching tabs automatically
    st.rerun()

# Always show results if they exist, regardless of which tab we're on
if st.session_state.get('results') and len(st.session_state.results) > 0:
    # Display results in the current tab context
    st.markdown('<div class="sub-header">🎯 Grading Results</div>', unsafe_allow_html=True)
    
    for i, r in enumerate(st.session_state.results):
        with st.container():
            # Header with student name and score
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 👨‍🎓 {r.get('name', 'Student')}")
            with col2:
                score = r.get('final_score', 0)
                score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{score}/100</h2>", unsafe_allow_html=True)
            
            # Progress bar visualization
            st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(score), unsafe_allow_html=True)
            
            # Score interpretation
            if score >= 80:
                st.markdown('<div class="success-box">🎉 Excellent work! Strong understanding demonstrated.</div>', unsafe_allow_html=True)
            elif score >= 60:
                st.markdown('<div class="warning-box">📚 Good effort, with room for improvement in key areas.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">💡 Needs significant improvement. Review fundamental concepts.</div>', unsafe_allow_html=True)
            
            # Detailed feedback in expanders
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📋 Detailed Feedback", expanded=True):
                    st.markdown("**Key Observations:**")
                    st.write(r["reasoning"])
                    
                    st.markdown("**Actionable Steps:**")
                    for line in r["feedback_lines"]:
                        st.markdown(f'<div class="feedback-item">💡 {line}</div>', unsafe_allow_html=True)
                    
                    if r.get("groq_feedback") and enable_ai_feedback:
                        st.markdown("**🤖 AI Insights:**")
                        st.write(r["groq_feedback"])
            
            with col2:
                if show_detailed_breakdown:
                    with st.expander("📊 Score Breakdown", expanded=True):
                        for item in r["details"].get("breakdown", []):
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            with col_a:
                                st.write(f"**{item['criterion']}**")
                            with col_b:
                                st.write(f"{item['subscore']:.1f}")
                            with col_c:
                                progress = item['subscore'] / 100
                                st.progress(progress)
                
                if show_grammar_examples and r["details"].get("grammar", {}).get("available"):
                    with st.expander("🔍 Grammar Check", expanded=False):
                        g = r["details"]["grammar"]
                        st.write(f"**Issues found:** {g['issues_count']}")
                        for ex in g["examples"]:
                            st.markdown(f"""
                            <div class="grammar-issue">
                                <strong>⚠️ {ex['message']}</strong><br>
                                <em>Context:</em> ...{ex['context']}...<br>
                                {f"<em>Suggestions:</em> {', '.join(ex.get('suggestions', []))}" if ex.get('suggestions') else ""}
                            </div>
                            """, unsafe_allow_html=True)
            
            st.divider()
