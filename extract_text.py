from pdfminer.high_level import extract_text
import os

def extract_text_from_pdfs(pdf_folder):
    all_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text(pdf_path)
            all_text += text + "\n\n"
    return all_text


# Specify the PDF folder
pdf_folder = "data"
extracted_text = extract_text_from_pdfs(pdf_folder)

# Save extracted text to a file
with open("harry_potter_text.txt", "w", encoding="utf-8") as f:
    f.write(extracted_text)

print("Text extraction complete!")




