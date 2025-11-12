# 🧠 Document Question Answering (DocQA)

A lightweight pipeline that lets you **ask natural-language questions** about the contents of a **PDF** file.
It combines PDF text extraction (`pypdf`), document chunking, a simple **TF-IDF retriever**, and a **Hugging Face question-answering** model.

---

## 🚀 Features

* 📄 **Read any PDF** file using [`pypdf`](https://pypi.org/project/pypdf/)
* 🧩 **Chunk** large documents for long-context handling
* 🔍 **Retrieve relevant sections** with TF-IDF similarity
* 🤖 **Extract answers** using a pretrained QA model (DistilBERT)
* ⚙️ **Automate queries** through a single callable `DocumentQA` class

---

## 🏗️ Project Structure

```
docqa/
│
├── docqa.py            # main pipeline (PDF → text → retrieval → QA)
├── requirements.txt     # dependencies
├── README.md            # project documentation
└── sample.pdf           # optional sample input
```

---

## 📦 Installation

```bash
# clone your repo
git clone https://github.com/iamMashel/Document-Question-Answering-DocQA-.git
cd docqa

# create environment (optional)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt
```

**requirements.txt**

```
pypdf
transformers
torch
scikit-learn
```

---

## 🧩 How It Works

### 1. Load and Read PDFs

```python
from pypdf import PdfReader
reader = PdfReader("document.pdf")
text = "\n".join(page.extract_text() for page in reader.pages)
```

### 2. Split into Chunks

The text is divided into overlapping word chunks to respect model token limits.

### 3. Retrieve the Most Relevant Chunks

TF-IDF vectors identify the top *k* paragraphs most related to your query.

### 4. Extract the Final Answer

A pretrained question-answering model (`distilbert-base-cased-distilled-squad`) extracts the best-matching span.

---

## ⚡ Quick Start

```python
from pathlib import Path
from docqa import DocumentQA

pdf_path = Path("sample.pdf")
dqa = DocumentQA(pdf_path)

question = "What pipeline does the document recommend for question answering?"
result = dqa.answer(question)

print("Answer:", result["answer"])
print("Confidence:", result["score"])
```

**Output**

```
Answer: question-answering pipeline
Confidence: 0.92
```

---

## 🧠 Design Overview

| Stage          | Tool                                          | Purpose                  |
| :------------- | :-------------------------------------------- | :----------------------- |
| **Load PDF**   | `PdfReader` (pypdf)                           | Extracts raw text        |
| **Chunking**   | Python                                        | Handles long contexts    |
| **Retrieval**  | `TfidfVectorizer`                             | Narrows context to top-k |
| **QA Model**   | Hugging Face `pipeline("question-answering")` | Finds answer span        |
| **Automation** | `DocumentQA` class                            | Unified API for queries  |

---

## 🔮 Future Enhancements

* **Semantic retrieval** using sentence embeddings (`sentence-transformers`)
* **Cross-encoder reranking**
* **FastAPI / Streamlit interface**
* **OCR fallback** for scanned PDFs (`pytesseract`)
* **Generative answering** (`flan-t5`, `mistral-7b-instruct`, etc.)

---

## 🧰 Example CLI

```bash
python docqa.py --pdf sample.pdf --q "What is the main goal of this document?"
```

---

## 🧾 License

MIT License © 2025 Mashel Odera

---


