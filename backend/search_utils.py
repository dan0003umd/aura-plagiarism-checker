# search_utils.py
import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors safely."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_plagiarism_embeddings(text: str, sources: list, threshold: float = 0.7):
    """
    Check plagiarism using embeddings.

    Returns:
        plagiarism_score: float (0-100)
        matched_sources: list of dicts with url, snippet, similarity
    """
    results = []

    # Split text into fragments (sentences)
    fragments = [f.strip() for f in re.split(r'(?<=[.!?]) +', text) if f.strip()]
    if not fragments:
        return 0.0, []

    # Precompute embeddings for all fragments
    fragments_emb = model.encode(fragments)

    for src in sources:
        snippet = src.get("snippet", "")
        if not snippet:
            continue
        snippet_emb = model.encode(snippet)

        for frag, frag_emb in zip(fragments, fragments_emb):
            sim = cosine_similarity(frag_emb, snippet_emb)
            if sim >= threshold:
                results.append({
                    "url": src["url"],
                    "similarity": sim,
                    "fragment": frag,
                    "snippet": snippet
                })

    # Calculate plagiarism score as average similarity * 100
    plagiarism_score = np.mean([r["similarity"] for r in results]) * 100 if results else 0.0

    # Sort results by similarity descending
    results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return plagiarism_score, results_sorted


# --- SERPAPI search ---
def serpapi_search(query, num=5):
    """Search using SerpAPI."""
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


# --- Fallback search ---
def fallback_search(query, num=5):
    """Fallback search using googlesearch library and fetch snippets."""
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
    """Fetch text snippet from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AURA/1.0)"}
        r = requests.get(url, timeout=5, headers=headers)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        ps = soup.find_all("p")
        text = " ".join(p.get_text().strip() for p in ps)
        return text[:max_chars]
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""
