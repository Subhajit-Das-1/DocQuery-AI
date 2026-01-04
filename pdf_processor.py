import fitz  # PyMuPDF
import re

# ---------------------------
# Text cleaning
# ---------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------
# Heading heuristic
# ---------------------------
def is_heading(text, font_size):
    if len(text) < 3:
        return False
    if len(text) > 120:
        return False
    if text.isdigit():
        return False
    if font_size >= 14:
        return True
    if text.isupper() and len(text.split()) <= 10:
        return True
    return False

# ---------------------------
# PDF extraction
# ---------------------------
def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []

    current_section = "Introduction"

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = line.get("spans", [])
                if not spans:
                    continue

                text = clean_text(" ".join(span["text"] for span in spans))
                if not text:
                    continue

                max_font_size = max(span["size"] for span in spans)

                # -------- Heading detection --------
                if is_heading(text, max_font_size):
                    current_section = text
                    continue

                # -------- Ignore noise --------
                if len(text) < 4:
                    continue

                pages.append({
                    "text": text,
                    "page": page_num + 1,
                    "section": current_section
                })

    return pages
