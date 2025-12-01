from pypdf import PdfReader
import sys

try:
    reader = PdfReader("Task 1 - BS.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Successfully wrote pdf_content.txt")
except Exception as e:
    print(f"Error reading PDF: {e}")
