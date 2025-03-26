# Conversational History-Aware Q&A Chatbot

This project is a **Streamlit-based AI chatbot** that provides **context-aware question answering from PDF documents** while maintaining conversational history. It leverages **Groq's LLaMA 3.3 70B model** for intelligent responses and **FAISS** for efficient document retrieval.

## Features

- **Conversational Memory**: Maintains chat history for contextual responses.
- **PDF-Based Question Answering**: Upload multiple PDFs and ask questions about their content.
- **Context-Aware Responses**: Dynamically retrieves relevant document sections and chat history.
- **Efficient Text Embedding & Retrieval**: Uses Hugging Face embeddings and FAISS for vector search.
- **Interactive UI**: Built with Streamlit for an intuitive user experience.

## Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Groq (LLaMA-3.3-70B-Versatile model)**
- **FAISS (Facebook AI Similarity Search)**
- **Hugging Face Embeddings**
- **PyPDFLoader**

## Installation

### Prerequisites

- Python 3.8+
- Groq API Key
- Hugging Face API Key

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/PrinceGupta8/Conversational-chatbot-chat-with-pdf-along-with-chat-history.git
   cd Conversational-chatbot-chat-with-pdf-along-with-chat-history
   ```
2. **Create a virtual environment** 
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Enter your Groq API Key** in the environment variables or Streamlit input box.
2. **Upload one or multiple PDF documents**.
3. **Click 'Embedding'** to process the documents and create searchable embeddings.
4. **Enter a Session ID** to keep chat history separate for different users.
5. **Ask questions** related to the uploaded PDFs.
6. **Receive AI-generated responses** that consider both document content and chat history.

## Example

**Input:**

```
What is the main topic discussed in the document?
```

**Output:**

```
The document primarily discusses the concept of self-attention in transformer models...
```

## API Keys

- Get your **Groq API Key** from [Groq](https://groq.com/).
- Get your **Hugging Face API Key** from [Hugging Face](https://huggingface.co/).
- Set these as environment variables (`groq_api_key` and `huggingface_api_key`) or input them in the Streamlit UI.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss your ideas.

---

**Enhance your document-based conversations with AI-powered context-aware answers!**

