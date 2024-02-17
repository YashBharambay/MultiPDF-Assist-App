
# MultiPDF Assist App

The MultiPDF Assist App is a Python application designed for conversational interaction with multiple PDF documents. Through natural language queries, users can obtain relevant responses based on the content of the PDFs loaded into the application. It leverages a language model to provide accurate answers, limiting responses to questions pertinent to the loaded documents.

## Functionality Overview:

PDF Loading: The app reads and extracts text content from multiple PDF documents.

Text Chunking: Extracted text is segmented into smaller, manageable chunks for efficient processing.

Language Model: Utilizes a language model to create vector representations (embeddings) of text chunks.

Similarity Matching: When users pose questions, the app compares them with text chunks to identify semantically similar content.

Response Generation: Selected chunks are fed into the language model to generate responses based on relevant PDF content.

## How to Use:

Setup: Ensure the necessary dependencies are installed and add the OpenAI API key to the .env file.

1. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
2. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

Launch: Run the main.py file using the Streamlit CLI with the command 
    ```
    streamlit run app.py.
    ```
Interface: The application will open in your default web browser, presenting the user interface.

Load PDFs: Follow the provided instructions to upload multiple PDF documents into the app.

Chat Interface: Engage in conversation by asking questions in natural language about the loaded PDFs via the chat interface.

# To start the virtual environment in windows
 source venv/Scripts/activate
