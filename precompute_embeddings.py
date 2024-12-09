import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import re
import os

# Load the file
file_path = "scraped_data.txt"

with open(file_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split the text by entries (double newlines)
entries = raw_text.split("\n\n")

# Function to clean text content
def clean_text_content(text):
    patterns_to_remove = [
        r"Up\^Add To My Favorites",
        r"Add To My Favorites",
        r"Up",
        r"\[.*?\]",
        r"Code Text.*?:",
        r"DIVISION\s*\d+.*?CHAPTER\s*\d+",
        r"https?://\S+",
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to extract entry details
def extract_entry_details(entry):
    url_pattern = r"URL:\s*(https?://\S+)"
    code_pattern = r"DIVISION\s*(\d+).*(CHAPTER\s*\d+)"
    text_pattern = r"Text Content:\s*(.*)"

    url = re.search(url_pattern, entry)
    code = re.search(code_pattern, entry)
    text = re.search(text_pattern, entry)

    url = url.group(1) if url else None
    code = code.group(0) if code else None
    text = clean_text_content(text.group(1)) if text else None

    return {"url": url, "code": code, "text": text}

# Extract all entries
documents = [extract_entry_details(entry) for entry in entries]

# Remove entries with missing text content
documents = [doc for doc in documents if doc["text"]]

# Function to detect index-like entries
def is_index_like(text):
    # Common patterns in index-like entries
    index_keywords = ["CHAPTER", "PART", "ARTICLE", "DIVISION"]
    # Count occurrences of index keywords
    keyword_count = sum(text.count(keyword) for keyword in index_keywords)
    # Count the number of lines with enumerations
    enumeration_count = len(re.findall(r"\d+\.", text))  # Matches "1.", "2.", etc.
    # Count non-alphabetic characters
    non_alpha_ratio = sum(1 for c in text if not c.isalpha()) / max(1, len(text))
    
    # Thresholds for detecting index-like content
    if keyword_count > 5 or enumeration_count > 5 or non_alpha_ratio > 0.5:
        return True
    return False

# Filter out index-like documents
filtered_documents = [doc for doc in documents if not is_index_like(doc["text"])]

# Create a Hugging Face Dataset with the filtered data
corpus = Dataset.from_dict({
    "url": [doc["url"] for doc in filtered_documents],
    "code": [doc["code"] for doc in filtered_documents],
    "text": [doc["text"] for doc in filtered_documents]
})

# Create a Hugging Face Dataset with structured data
corpus = Dataset.from_dict({
    "url": [doc["url"] for doc in documents],
    "code": [doc["code"] for doc in documents],
    "text": [doc["text"] for doc in documents]
})

# Load embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

# Embedding function with GPU support
def encode(texts, batch_size=8):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# Embed the corpus text
corpus_embeddings = encode(corpus["text"])

# Create a FAISS index with GPU support
import faiss
res = faiss.StandardGpuResources()
index_flat = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index = faiss.index_cpu_to_gpu(res, 0, index_flat)
index.add(corpus_embeddings)

# Convert the GPU index to a CPU index
index_cpu = faiss.index_gpu_to_cpu(index)

# Now save the index to a file
faiss.write_index(index_cpu, "faiss_index.index")

# Save the corpus data
np.save("corpus_embeddings.npy", corpus_embeddings)

current_directory = os.getcwd()
corpus.save_to_disk(os.path.join(current_directory, "corpus"))