# Healthcare Policy Research Assistant

A Retrieval Augmented Generation (RAG) application built with AWS Bedrock, LangChain, and Streamlit. This application allows users to query healthcare policy documents using natural language and get AI-powered responses based on the document content.

## Features

- PDF document ingestion and processing
- Vector store creation using FAISS and AWS Bedrock embeddings
- Question-answering using Claude 3 Sonnet or LLama 2 models
- Interactive web interface built with Streamlit
- Document chunk optimization for better context preservation

## Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- PDF documents to query (place them in a `data` directory)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/healthcare-policy-assistant.git
   cd healthcare-policy-assistant
2. **Create a Virtual Environment:**
   ``` bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install required packages:**
   ``` bash
   pip install -r requirements.txt
4. **Configure AWS Credentials**
      ``` bash
      aws configure
5. **Create a data directory and add your PDF documents**
     ``` bash
     mkdir data

## Project Structure

```healthcare-policy-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Project dependencies
├── data/                  # Directory for PDF documents
└── README.md              # Project documentation```





