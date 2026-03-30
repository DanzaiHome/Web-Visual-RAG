import requests
import re
import base64

from RAG_hints.RAG_hints import images_to_hints

from bs4 import BeautifulSoup
from urllib.parse import quote
from sentence_transformers import SentenceTransformer
import numpy as np

from typing import List

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

RAG_SERVER_URL = "http://127.0.0.1:18000/predict"
VL_SERVER_BASE_URL = "http://127.0.0.1:18001"
RAG_ROUTER_THRESHOLD = 0.5
SIMILARITY_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Set a global proxy will make localhost servers unaccessible on my device.
PROXIES = {
    "http": "http://10.176.52.116:7890",
    "https": "http://10.176.52.116:7890",
}

# PROXIES = {
# }

proxy_session = requests.Session()
proxy_session.proxies.update(PROXIES)

local_session = requests.Session()
local_session.trust_env = False

def RAG_router(img_path: str, question: str) -> bool:
    try:
        with open(img_path, "rb") as f:
            files = {"file": f}
            data = {"question": question}
            response = requests.post(RAG_SERVER_URL, files=files, data=data)

        match = re.search(r'"probability"\s*:\s*([0-9\.eE+-]+)', response.text)
        if match:
            prob = float(match.group(1))
            print(f"RAG prob: {prob}")
            return prob >= RAG_ROUTER_THRESHOLD
        else:
            print("[Warning] RAG router parse error, enable RAG.")
            return True
    except Exception as e:
        print(f"[Warning] RAG router failed: {e}, enable RAG.")
        return True



def call_vl_model(img_paths: List[str], question: str) -> str:
    url = f"{VL_SERVER_BASE_URL}/v1/chat/completions"

    content = [{"type": "text", "text": question}]

    for img_path in img_paths:
        try:
            with open(img_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")

            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })
        except Exception as e:
            print(f"[Warning] Failed to load image {img_path}: {e}")

    payload = {
        "model": "Qwen3-VL",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})

    if response.status_code != 200:
        raise RuntimeError(f"VL API error: {response.status_code}, {response.text}")

    result = response.json()

    try:
        return result["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Invalid VL response format: {result}")


def build_query(hints: List[str], question: str, max_hints: int = 5) -> str:
    hints = [h.strip() for h in hints if h.strip()][:max_hints]
    query = " ".join(hints) + " " + question
    print(f"RAG Query: {query}")
    return query


# def search_duckduckgo(query: str, top_k: int = 5) -> List[str]:
#     # url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
#     url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json"
#     
#     resp = proxy_session.get(
#         url,
#         headers={"User-Agent": "Mozilla/5.0"},
#         timeout=10
#     )
# 
#     soup = BeautifulSoup(resp.text, "html.parser")
# 
#     links = []
#     for a in soup.select(".result__a", limit=top_k):
#         href = a.get("href")
#         if href:
#             links.append(href)
# 
#     return links


def search_bing(query, top_k=5):
    url = f"https://www.bing.com/search?q={quote(query)}"

    resp = local_session.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10
    )

    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.select("li.b_algo h2 a", limit=top_k):
        href = a.get("href")
        if href:
            links.append(href)

    return links

def fetch_page_text(url: str, max_len: int = 2000) -> str:
    try:
        resp = proxy_session.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_len]
    except Exception:
        return ""


def compute_similarity(query: str, documents: List[str]):
    query_emb = SIMILARITY_MODEL.encode([query])[0]
    doc_embs = SIMILARITY_MODEL.encode(documents)

    scores = np.dot(doc_embs, query_emb) / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    return scores


def retrieve_web_context(hints: List[str], question: str, top_k: int = 5):
    query = build_query(hints, question)

    urls = search_bing(query, top_k=10)

    docs, valid_urls = [], []
    seen = set()

    for url in urls:
        text = fetch_page_text(url)
        if len(text) > 200 and text not in seen:
            docs.append(text)
            valid_urls.append(url)
            seen.add(text)

    if not docs:
        return []

    scores = compute_similarity(query, docs)

    ranked = sorted(zip(valid_urls, docs, scores), key=lambda x: x[2], reverse=True)

    return [
        {
            "url": url,
            "score": float(score),
            "content": text[:500]
        }
        for url, text, score in ranked[:top_k]
    ]


def RAG_pipeline(img_paths: List[str], question: str) -> str:

    hints = images_to_hints(img_paths=img_paths)

    retrieved = retrieve_web_context(hints, question)

    context_text = "\n\n".join([
        f"[Doc {i+1}] {item['content']}"
        for i, item in enumerate(retrieved)
    ])

    enhanced_question = f"""
Answer the question based on the provided context.

Context:
{context_text}

Question:
{question}
"""
    
    # FOR DEBUG
    print(f"Final RAG query: {enhanced_question}")

    return call_vl_model(img_paths, enhanced_question)


def main():
    image_paths = [
        PROJECT_DIR / "RAG_identifier" / "data" / "inference_query" / "Sacred_Heart_Cathedral_of_Guangzhou_2025.06_02.jpg"
    ]
    question = "What is this made of?"

    use_rag = RAG_router(img_path=image_paths[0], question=question)

    if use_rag:
        print("RAG enabled.")
        response = RAG_pipeline(image_paths, question)
    else:
        print("RAG disabled.")
        response = call_vl_model(image_paths, question)

    print(f"Response: {response}")


if __name__ == "__main__":
    main()