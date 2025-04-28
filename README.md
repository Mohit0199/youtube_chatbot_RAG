# YouTube Video Q&A Chatbot

This project is a Streamlit-based web application that allows users to paste a YouTube video URL, fetch its transcript, and interact with a question-answer chatbot based on the video content. The app displays the video and its details on the left side and a chat interface on the right side for asking questions about the video.

## Core Technology: Langchain Retrieval-Augmented Generation (RAG) System

This project is built on the Langchain RAG architecture, which enhances language model responses by retrieving relevant context from external documents—in this case, YouTube video transcripts. The system works as follows:

- **Transcript Extraction:** The YouTube video transcript is fetched using the `youtube_transcript_api`.
- **Text Splitting:** The transcript is split into manageable chunks using Langchain's `RecursiveCharacterTextSplitter`.
- **Embeddings:** Each chunk is converted into vector embeddings using HuggingFace's sentence-transformer models.
- **Vector Store:** The embeddings are stored and indexed in a FAISS vector store for efficient similarity search.
- **Retriever:** A `MultiQueryRetriever` fetches the most relevant transcript chunks based on the user's question.
- **Language Model:** Google Generative AI (Gemini) is used as the language model to generate answers.
- **Prompting:** A custom prompt template guides the model to answer questions strictly based on the retrieved transcript context.
- **Chain:** Chaining all components together using both parallel and sequential processing to efficiently handle context retrieval, question answering, and response generation..

This RAG-based approach allows the chatbot to provide accurate, context-aware answers grounded in the actual video transcript, improving reliability over standalone language models.

## Features

- Paste a YouTube video URL and fetch its transcript.
- Display the video and video details.
- Interactive chat interface to ask questions about the video content.
- Supports resetting the chat and starting over with a new video.
- Responsive and user-friendly UI using Streamlit columns and custom CSS.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Open the URL provided by Streamlit in your browser.

## Configuration and API Keys

This project uses external APIs and services that require API keys. You need to obtain API keys from the following providers:

### 1. AI Studio Google (Google Cloud)

- Go to [Google AI Studio](https://aistudio.google.com/apikey).
- Create an API Key
- Copy your API key and set it in your environment or configuration as required by the app.

### 2. Hugging Face

- Sign up or log in at [Hugging Face](https://huggingface.co/).
- Go to your account settings and create an access token.
- Copy the token and set it in your environment or configuration as required by the app.

## Project Structure

- `app.py`: Main Streamlit application file handling UI and user interactions.
- `main.py`: Contains core logic for extracting video ID, fetching transcripts, creating the retrieval-augmented generation (RAG) chain, and fetching YouTube video info.
- `requirements.txt`: Python dependencies required for the project.
