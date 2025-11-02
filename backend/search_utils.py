# search_utils.py
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# Load the embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors safely."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def check_plagiarism_embeddings(text: str, sources: list, threshold: float = 0.75):
    """
    Check plagiarism using embeddings.

    text: str, the text to check
    sources: list of {"url": ..., "snippet": ...}
    threshold: cosine similarity above which we consider text plagiarized
    """
    results = []

    # Split text into fragments (sentences) for more granular matching
    import re
    fragments = [f.strip() for f in re.split(r'(?<=[.!?]) +', text) if f.strip()]
    
    for src in sources:
        snippet = src.get("snippet", "")
        snippet_emb = model.encode(snippet)
        for frag in fragments:
            frag_emb = model.encode(frag)
            sim = cosine_similarity(frag_emb, snippet_emb)
            if sim >= threshold:
                results.append({
                    "url": src["url"],
                    "similarity": sim,
                    "fragment": frag,
                    "snippet": snippet
                })

    # Plagiarism score is average similarity of matched fragments, scaled to 100
    if results:
        plagiarism_score = np.mean([r["similarity"] for r in results]) * 100
    else:
        plagiarism_score = 0.0

    # Sort results by similarity descending
    results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return plagiarism_score, results_sorted


# --- Existing search functions ---
def serpapi_search(query, num=5):
    if not SERPAPI_KEY:
        return []
    try:
        from serpapi import GoogleSearch
        params = {"engine": "google", "q": query, "api_key": SERPAPI_KEY, "num": num}
        search = GoogleSearch(params)
        res = search.get_dict()
        results = []
        for item in res.get("organic_results", [])[:num]:
            results.append({
                "url": item.get("link"),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        print("SerpAPI search error:", e)
        return []

def fallback_search(query, num=5):
    try:
        from googlesearch import search
    except Exception as ex:
        print("googlesearch not available:", ex)
        return []
    results = []
    try:
        for url in search(query, num_results=num):
            snippet = fetch_snippet_from_url(url)
            results.append({"url": url, "snippet": snippet})
            if len(results) >= num:
                break
    except Exception as e:
        print("fallback_search error:", e)
    return results

def fetch_snippet_from_url(url, max_chars=400):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AURA/1.0)"}
        r = requests.get(url, timeout=5, headers=headers)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        ps = soup.find_all("p")
        text = " ".join(p.get_text().strip() for p in ps)
        return text[:max_chars]
    except Exception:
        return ""
