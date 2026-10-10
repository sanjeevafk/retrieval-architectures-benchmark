import re
import json
import time
from pathlib import Path
import pdf_inspector as pi

BOOKS = [
    {
        "filename": "06. Metasploit - The Penetration Tester's Guide.pdf",
        "book_tag": "Metasploit",
        "short_title": "Metasploit: The Penetration Tester's Guide",
    },
    {
        "filename": "05. SQL Injection Attacks and Defense.pdf",
        "book_tag": "SQLi_Attacks",
        "short_title": "SQL Injection Attacks and Defense",
    },
    {
        "filename": "15. The Web Application Hacker's Handbook.pdf",
        "book_tag": "WAHH",
        "short_title": "The Web Application Hacker's Handbook",
    },
]

BASE_DIR = Path("/home/sanjeev/Documents/Yeahhub - Hacking & Security E-Books")
OUT_DIR = Path("/home/sanjeev/Downloads/beir-ab/sec_eval")


def clean_page_text(text: str) -> str:
    # join hyphenated line breaks (e.g., vul- \n nerability -> vulnerability)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # remove soft hyphens
    text = text.replace('\xad', '')
    # collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_document(book_meta, pages, target_words=400, overlap_words=50):
    chunks = []
    current_tokens = []
    current_pages = set()
    current_heading = book_meta["short_title"]
    
    chunk_idx = 0
    for page_obj in pages:
        p_num = page_obj.page
        text = clean_page_text(page_obj.markdown)
        if not text:
            continue
            
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # track recent markdown heading
            heading_match = re.match(r'^(#{1,4}\s+.+)', para)
            if heading_match:
                current_heading = heading_match.group(1).lstrip('#').strip()
                
            words = para.split()
            current_tokens.extend(words)
            current_pages.add(p_num)
            
            if len(current_tokens) >= target_words:
                chunk_text = " ".join(current_tokens)
                doc_id = f"{book_meta['book_tag']}_p{min(current_pages):03d}_{chunk_idx:04d}"
                chunks.append({
                    "id": doc_id,
                    "book": book_meta["short_title"],
                    "book_tag": book_meta["book_tag"],
                    "pages": sorted(list(current_pages)),
                    "section": current_heading,
                    "title": f"{book_meta['short_title']} - {current_heading}",
                    "text": chunk_text,
                })
                chunk_idx += 1
                # apply overlap
                current_tokens = current_tokens[-overlap_words:] if overlap_words > 0 else []
                current_pages = {p_num}

    if current_tokens:
        chunk_text = " ".join(current_tokens)
        doc_id = f"{book_meta['book_tag']}_p{min(current_pages):03d}_{chunk_idx:04d}"
        chunks.append({
            "id": doc_id,
            "book": book_meta["short_title"],
            "book_tag": book_meta["book_tag"],
            "pages": sorted(list(current_pages)),
            "section": current_heading,
            "title": f"{book_meta['short_title']} - {current_heading}",
            "text": chunk_text,
        })
    return chunks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    print("Extracting and chunking security books...")
    t0 = time.perf_counter()

    for b in BOOKS:
        pdf_path = BASE_DIR / b["filename"]
        if not pdf_path.exists():
            print(f"Warning: {pdf_path} not found!")
            continue
            
        bt0 = time.perf_counter()
        res = pi.extract_pages_markdown(str(pdf_path))
        chunks = chunk_document(b, res.pages)
        all_chunks.extend(chunks)
        print(f"  [{b['book_tag']}] {len(res.pages)} pages -> {len(chunks)} chunks in {time.perf_counter() - bt0:.2f}s")

    out_file = OUT_DIR / "corpus.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")

    print(f"\nDone! Wrote {len(all_chunks)} chunks to {out_file} in {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
