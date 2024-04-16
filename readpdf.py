import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def read_pdf_files(folder_path):

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("Folder '{folder_path}' does not exist.")
        return

    # Iterate over each file in the folder
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Extracting text from '{pdf_path}'...")
            try:
                raw_text = extract_text_from_pdf(pdf_path)
                pdf_chunks = get_text_chunks(raw_text)
                chunks.extend(pdf_chunks)
            except Exception as e:
                print(f"Error extracting text from '{pdf_path}': {e}")
    return chunks

