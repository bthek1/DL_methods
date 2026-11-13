# Document Pre-processing for RAG

Before text can be chunked and embedded, raw documents must be parsed, cleaned, and structured. This stage is the often-overlooked foundation of retrieval quality.

Author

Benedict Thekkel

## 1. Why Pre-processing Matters

Garbage in, garbage out. The most sophisticated retrieval pipeline cannot recover from poorly extracted source text. Common failures:

| Root cause | Symptom at query time |
|----|----|
| Table rows merged into a single line | Numerical answers are garbled |
| PDF columns not separated | Sentences from two columns are fused |
| Scanned image with no OCR | Entire page is invisible to the retriever |
| Boilerplate headers/footers | Retrieved chunks polluted with irrelevant text |
| Missing section titles | Re-ranking has no structural signal |

Pre-processing is highly domain-specific — a pipeline that works for web pages may fail completely on engineering PDFs.

------------------------------------------------------------------------

## 2. Document Formats and Parsers

Each format has different parsing challenges:

### PDF

The hardest format. PDFs are a layout specification, not a text format. Text extraction order can be wrong, columns may merge, and embedded images are invisible.

**Parser options:**

``` python
# pdfminer — low-level, good for plain text PDFs
from pdfminer.high_level import extract_text
text = extract_text("doc.pdf")

# pymupdf (fitz) — fast, handles layout better
import fitz
doc = fitz.open("doc.pdf")
pages = [page.get_text("text") for page in doc]

# unstructured — highest quality, handles mixed content
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("doc.pdf", strategy="hi_res")
```

### Word / DOCX

``` python
from docx import Document
doc = Document("report.docx")
paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
tables = [[cell.text for cell in row.cells] for table in doc.tables for row in table.rows]
```

### HTML / Web pages

``` python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html.parser")
# Remove boilerplate
for tag in soup(["nav", "footer", "header", "script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n", strip=True)
```

### Markdown / plain text

Easiest — structure is already explicit. Use heading markers (`#`, `##`) as natural chunk boundaries.

------------------------------------------------------------------------

## 3. Table Extraction

Tables are especially fragile — naive text extraction destroys their structure.

**Strategy options:**

1.  **Convert to Markdown table** — preserves row/column relationships; works well in context
2.  **Convert to natural language** — use an LLM to summarise: *“Table showing Q1-Q4 revenue by region…”*
3.  **Store as structured data** — query with Text-to-SQL at retrieval time (TableGPT approach)
4.  **Hybrid** — store both the Markdown and a text summary; embed the summary, retrieve the Markdown

``` python
# Table Transformer (Microsoft) — extracts tables from PDFs as images
from transformers import pipeline
table_extractor = pipeline("object-detection", model="microsoft/table-transformer-detection")

# camelot — extracts tables from PDFs with cell coordinates
import camelot
tables = camelot.read_pdf("report.pdf", pages="all", flavor="lattice")
df = tables[0].df  # pandas DataFrame
markdown = df.to_markdown(index=False)
```

**Rule of thumb:** if a table is the primary source of the answer, convert to Markdown and include it as its own chunk with a descriptive header.

------------------------------------------------------------------------

## 4. OCR for Scanned Documents

Scanned PDFs and image files contain no machine-readable text — they must go through OCR before chunking.

``` python
# Tesseract via pytesseract
from PIL import Image
import pytesseract

image = Image.open("scanned_page.png")
text = pytesseract.image_to_string(image, lang="eng")

# For PDFs: convert pages to images first
from pdf2image import convert_from_path
images = convert_from_path("scanned.pdf", dpi=300)
pages = [pytesseract.image_to_string(img) for img in images]

# unstructured handles this automatically with strategy="hi_res"
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("scanned.pdf", strategy="hi_res", languages=["eng"])
```

**OCR quality tips:** - Use 300 DPI minimum for reliable character recognition - Pre-process images: deskew, remove noise, increase contrast - Validate output: check character-to-space ratios for garbage detection - For critical documents, consider cloud OCR (AWS Textract, Google Document AI) — significantly more accurate than Tesseract

------------------------------------------------------------------------

## 5. Metadata Extraction

Metadata attached to chunks at index time enables precise filtering at query time (narrowing scope before vector search).

**Standard metadata fields:**

``` python
metadata = {
    "source": "annual_report_2024.pdf",
    "page": 12,
    "section": "Financial Results",
    "author": "Finance Team",
    "created_at": "2024-03-15",
    "doc_type": "report",          # report | policy | faq | email
    "department": "finance",
    "language": "en",
    "classification": "internal",  # public | internal | confidential
}
```

**LLM-generated metadata:** For unstructured documents, use an LLM to extract structured metadata:

``` python
prompt = """
Extract the following from this document excerpt:
- topic (one of: policy, procedure, faq, technical, financial, legal)
- key_entities: list of company/product/person names mentioned
- date_references: any dates or time periods mentioned

Respond as JSON.

Document: {text}
"""
```

**Hypothetical question metadata (Reverse HyDE):** Generate questions this chunk can answer. At query time, match query → hypothetical question for better semantic alignment.

------------------------------------------------------------------------

## 6. Cleaning and Normalisation

Raw extracted text typically contains noise that degrades embedding quality.

``` python
import re

def clean_text(text: str) -> str:
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove repeated punctuation (OCR artefacts)
    text = re.sub(r"[.]{3,}", "...", text)
    text = re.sub(r"[-]{3,}", "---", text)
    
    # Remove page numbers and headers/footers (regex-based heuristics)
    text = re.sub(r"^Page \d+ of \d+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^CONFIDENTIAL\s*$", "", text, flags=re.MULTILINE)
    
    # Normalise Unicode (important for multilingual docs)
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    
    return text
```

**Boilerplate removal strategies:** - Heuristic: lines that appear in \>80% of documents are likely boilerplate - LLM-based: prompt to identify and strip headers, footers, legal disclaimers - Structural: for PDFs, detect repeated text at fixed page positions

------------------------------------------------------------------------

## 7. Pre-processing Pipeline Design

A production ingestion pipeline chains these stages:

    Raw file
      └─► Format detection
            └─► Parser (pdfminer / docx / html / ...)
                  ├─► OCR (if scanned)
                  ├─► Table extractor
                  └─► Text extractor
                        └─► Cleaner / normaliser
                              └─► Metadata extractor
                                    └─► Chunker  →  Embedder  →  Index

**Key design decisions:**

| Decision | Options |
|----|----|
| Parser choice | Speed vs quality — `pymupdf` (fast) vs `unstructured` (best quality) |
| Image handling | Skip images / OCR / multimodal embedding |
| Table handling | Markdown / natural language / structured store |
| Metadata source | File system attributes / LLM extraction / manual |
| Quality gate | Minimum chunk length, language detection, garbage detection |

**Quality gate example:**

``` python
def is_valid_chunk(text: str, min_length: int = 50) -> bool:
    if len(text.strip()) < min_length:
        return False
    # Detect garbage OCR output (high proportion of non-alpha chars)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.4:
        return False
    return True
```

**Bottom line:** Invest in pre-processing early. A small improvement here has multiplicative effects on retrieval quality across all downstream stages.

------------------------------------------------------------------------

Back to top
