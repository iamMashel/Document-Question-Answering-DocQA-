# pip install pypdf transformers torch scikit-learn
from pypdf import PdfReader
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re, textwrap
from pathlib import Path

# ---------- PDF → raw text ----------
def read_pdf_text(path):
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            t = p.extract_text() or ""
        except Exception as e:
            t = ""
        pages.append(t)
    raw = "\n\n".join(pages)
    # light cleanup
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()

# ---------- Chunking (for long docs) ----------
def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ---------- Build a simple TF-IDF retriever ----------
class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1,2),
            max_df=0.9,
            min_df=1,
            stop_words="english"
        )
        self.doc_mat = self.vectorizer.fit_transform(chunks)

    def top_k(self, query, k=5):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_mat)[0]
        idxs = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idxs]

# ---------- QA model ----------
qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ---------- Orchestrator ----------
class DocumentQA:
    def __init__(self, pdf_path, chunk_size=900, overlap=150):
        text = read_pdf_text(pdf_path)
        self.chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        self.retriever = TfidfRetriever(self.chunks)

    def answer(self, question, k=5):
        # 1) retrieve top-k chunks
        candidates = self.retriever.top_k(question, k=k)

        # 2) run QA on each, keep the best score
        best = None
        for ctx, sim in candidates:
            out = qa(question=question, context=ctx)
            # combine QA score and retrieval sim to rank
            score = out["score"] * 0.7 + sim * 0.3
            cand = {
                "answer": out["answer"],
                "score": score,
                "qa_score": float(out["score"]),
                "retrieval_sim": float(sim),
                "context": ctx
            }
            if (best is None) or (cand["score"] > best["score"]):
                best = cand
        return best

# ---------- Example ----------
if __name__ == "__main__":
    pdf = Path("your_document.pdf")
    dqa = DocumentQA(pdf)
    q = "What pipeline does the document recommend for question answering?"
    res = dqa.answer(q, k=5)
    print("Answer:", res["answer"])
    print("Confidence:", round(res["score"], 3))
    print("\nContext snippet:\n", textwrap.shorten(res["context"], width=400, placeholder=" ..."))
