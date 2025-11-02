# app.py
# AURA Frontend — Streamlit UI with file upload support

import streamlit as st
import requests
import os
from dotenv import load_dotenv
from io import StringIO
from PyPDF2 import PdfReader
import docx

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AURA | AI Unified Rewriting & Analysis", layout="wide")

st.title("🌌 AURA — AI Unified Rewriting & Analysis")
st.caption("Gemini-powered plagiarism checker and rewrite assistant")

# ------------------- Backend Connection -------------------
backend_url = st.text_input(
    "Backend URL",
    value=os.environ.get("AURA_BACKEND_URL", "http://localhost:8000/check"),
)

# ------------------- File Upload or Text Input -------------------
st.subheader("📝 Input Section")

upload_option = st.radio(
    "Choose Input Type:",
    ["Enter Text", "Upload File"],
    horizontal=True
)

text = ""
if upload_option == "Enter Text":
    text = st.text_area("Enter text to analyze", height=250)
else:
    uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_ext == "txt":
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                text = stringio.read()
            elif file_ext == "pdf":
                reader = PdfReader(uploaded_file)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif file_ext == "docx":
                doc = docx.Document(uploaded_file)
                text = " ".join([p.text for p in doc.paragraphs])
            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ------------------- Plagiarism Check -------------------
top_k = st.number_input("Number of sources to scan", min_value=1, max_value=10, value=3)
submitted = st.button("🔍 Analyze")

if submitted:
    if not text.strip():
        st.error("Please enter or upload some text first.")
    else:
        payload = {"text": text, "top_k": int(top_k)}
        try:
            with st.spinner("Analyzing... Please wait."):
                res = requests.post(backend_url, json=payload, timeout=90)

            if res.status_code != 200:
                st.error(f"Error {res.status_code}: {res.text}")
            else:
                data = res.json()
                score = data["plagiarism_score"]

                # Color-coded feedback
                if score < 30:
                    st.success(f"Plagiarism Score: {score}% — Minor or no plagiarism detected.")
                elif 30 <= score < 60:
                    st.warning(f"Plagiarism Score: {score}% — Moderate plagiarism detected. Needs major edits.")
                else:
                    st.error(f"Plagiarism Score: {score}% — Heavily plagiarized. Rewrite required.")

                # Sources
                st.subheader("🔗 Top Matched Sources")
                for src in data.get("sources", []):
                    st.write(f"**[{src['url']}]({src['url']})** — score: {src['score']:.2f}")
                    st.write(src.get("snippet", ""))
                    st.markdown("---")

                # Rewrite suggestion
                st.subheader("✨ Gemini Rewrite Suggestion")
                st.info(data.get("rewrite_suggestion", "(No suggestion returned)"))

        except Exception as e:
            st.error(f"Connection failed: {e}")
