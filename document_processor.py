"""
Document Preprocessor for RAG Chatbot
---------------------------------------

This script:
1. Reads raw documents (PDF, TXT, DOCX)
2. Cleans extracted text
3. Splits long text into smaller, sentence-aware chunks
4. Stores processed chunks in JSON & CSV for embeddings

Usage:
- Place raw docs inside the `data/` folder
- Run: python document_processor.py
- Output is saved inside the `chunks/` folder
"""

import os
import re
import json
import logging
from typing import List, Dict

import pandas as pd
import PyPDF2
from docx import Document


# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class DocProcessor:
    """Handles reading, cleaning, chunking and saving documents."""

    def __init__(self, source_dir: str = "data", output_dir: str = "chunks"):
        self.source_dir = source_dir
        self.output_dir = output_dir

        os.makedirs(self.source_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # -------- File Readers --------
    def _read_txt(self, path: str) -> str:
        """Read plain text file content."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin1") as f:
                return f.read()

    def _read_pdf(self, path: str) -> str:
        """Extract text from a PDF."""
        text = ""
        try:
            with open(path, "rb") as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            log.error(f"PDF reading failed ({path}): {e}")
        return text

    def _read_docx(self, path: str) -> str:
        """Read text from a Word file."""
        try:
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            log.error(f"DOCX reading failed ({path}): {e}")
            return ""

    # -------- Cleaning --------
    def _clean(self, raw: str) -> str:
        """Basic cleaning: remove noise, extra spaces, unwanted chars."""
        text = re.sub(r"\s+", " ", raw)          # collapse whitespace
        text = re.sub(r"Page \d+", "", text)     # drop page numbers
        text = re.sub(r"\d+/\d+/\d+", "", text)  # remove dates
        text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)  # strip odd chars
        return text.strip()

    # -------- Chunking --------
    def _chunkify(self, text: str,
                  min_words: int = 100,
                  max_words: int = 300) -> List[str]:
        """Break text into sentence-aware chunks of ~100–300 words."""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        chunks, current, count = [], "", 0

        for s in sentences:
            words = len(s.split())
            if count + words > max_words and count >= min_words:
                chunks.append(current.strip())
                current, count = s, words
            else:
                current += " " + s
                count += words

        if current.strip():
            chunks.append(current.strip())

        return chunks

    # -------- Core Processing --------
    def process_one(self, path: str) -> List[Dict]:
        """Read, clean, chunk and return structured data for one file."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            raw = self._read_txt(path)
        elif ext == ".pdf":
            raw = self._read_pdf(path)
        elif ext == ".docx":
            raw = self._read_docx(path)
        else:
            log.warning(f"Unsupported file skipped: {path}")
            return []

        cleaned = self._clean(raw)
        parts = self._chunkify(cleaned)

        data = []
        for idx, chunk in enumerate(parts):
            data.append({
                "chunk_id": f"{os.path.basename(path)}_{idx}",
                "source_file": os.path.basename(path),
                "chunk_index": idx,
                "content": chunk,
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
            })

        log.info(f"{os.path.basename(path)} → {len(data)} chunks")
        return data

    def process_all(self) -> List[Dict]:
        """Process every file in source_dir and save output."""
        all_chunks = []
        files = [f for f in os.listdir(self.source_dir)
                 if os.path.isfile(os.path.join(self.source_dir, f))]

        if not files:
            log.warning("No documents found in data/ folder.")
            return []

        for f in files:
            path = os.path.join(self.source_dir, f)
            chunks = self.process_one(path)
            all_chunks.extend(chunks)

        # Save outputs
        json_path = os.path.join(self.output_dir, "processed_chunks.json")
        csv_path = os.path.join(self.output_dir, "processed_chunks.csv")

        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(all_chunks, jf, indent=2, ensure_ascii=False)

        pd.DataFrame(all_chunks).to_csv(csv_path, index=False)

        log.info(f"Saved {len(all_chunks)} chunks → {json_path}, {csv_path}")
        return all_chunks


# -------- Run as script --------
def main():
    proc = DocProcessor()
    print("🔄 Processing documents...\n")
    chunks = proc.process_all()

    if not chunks:
        print("⚠️ No data processed. Add TXT/PDF/DOCX files to 'data/' folder.")
        return

    print(f"✅ Done! {len(chunks)} chunks created.")
    sizes = [c["word_count"] for c in chunks]
    print(f"📊 Avg size: {sum(sizes)/len(sizes):.1f} words | "
          f"Min: {min(sizes)} | Max: {max(sizes)}")


if __name__ == "__main__":
    main()
