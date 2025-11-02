# main.py
# AURA Backend — FastAPI service handling plagiarism detection

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import search_utils

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AURA Backend API")

# ------------------- Data Models -------------------

class CheckRequest(BaseModel):
    text: str
    top_k: int = 5

class Source(BaseModel):
    url: str
    snippet: str
    score: float

class CheckResponse(BaseModel):
    plagiarism_score: float
    sources: List[Source]
    rewrite_suggestion: str

# ------------------- API Endpoints -------------------

@app.post("/check", response_model=CheckResponse)
async def check(req: CheckRequest):
    """
    Main endpoint to check plagiarism and return rewrite advice.
    """
    # Validate input
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text input")

    # Step 1: Perform web search
    query = req.text[:200]  # truncate long text for efficient search
    candidates = search_utils.serpapi_search(query, num=req.top_k)

    # Use fallback if no results
    if not candidates:
        candidates = search_utils.fallback_search(query, num=req.top_k)

    # Still empty? Use demo sample
    if not candidates:
        candidates = [{"url": "https://example.com/article", "snippet": "Demo text content"}]

    # Step 2: Run embedding-based plagiarism check
    plagiarism_score, matched_sources = search_utils.check_plagiarism_embeddings(req.text, candidates)

    # Step 3: Format sources with clean scores
    sources = [
        {"url": x["url"], "snippet": x["snippet"], "score": round(x["similarity"], 4)}
        for x in matched_sources
    ]

    # Step 4: Add intelligent rewrite advice based on plagiarism score
    if plagiarism_score < 30:
        rewrite_suggestion = "✅ Your text looks quite original. Minor improvements or paraphrasing may help polish it."
    elif 30 <= plagiarism_score < 60:
        rewrite_suggestion = "⚠️ Your text shows moderate similarity. Consider rephrasing key sections and verifying originality."
    else:
        rewrite_suggestion = "🚨 Your text is heavily plagiarized. Rewrite it completely or make major changes to ensure originality."

    # Step 5: Return structured response
    return {
        "plagiarism_score": round(plagiarism_score, 2),
        "sources": sources,
        "rewrite_suggestion": rewrite_suggestion,
    }

@app.get("/")
def root():
    """
    Health check endpoint.
    """
    return {"status": "AURA backend is running!"}
