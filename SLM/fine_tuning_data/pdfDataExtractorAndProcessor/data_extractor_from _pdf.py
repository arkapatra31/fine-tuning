import os
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

pdf_path = os.getenv("PDF_TRAINING_DATA") #Make sure path is having / instead of \ as dir separators


if isinstance(pdf_path, str):
    contents = ""
    for filename in os.listdir(path=rf"{pdf_path}"):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(pdf_path, filename)

            pdf_reader = PyPDF2.PdfReader(pdf_file_path)

            for pages in pdf_reader.pages:
                contents += pages.extract_text()

        contents += "\n"

    with open(file=rf"{pdf_path}/extracts.txt", mode='w', encoding="utf-8") as file:
        file.write(contents)
    file.close()
