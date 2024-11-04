import json
import os
import sys
import boto3
import streamlit as st

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize AWS Bedrock client
# Make sure you have configured AWS credentials using aws configure
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock
)

def data_ingestion():
    """
    Load and process PDF documents from the data directory
    Returns:
        list: List of document chunks after splitting
    """
    try:
        loader = PyPDFDirectoryLoader("data")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []

def get_vector_store(docs):
    """
    Create and save FAISS vector store from documents
    Args:
        docs (list): List of document chunks
    """
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    """
    Initialize Claude 3 Sonnet model
    Returns:
        BedrockChat: Configured Claude 3 LLM instance
    """
    try:
        llm = BedrockChat(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            client=bedrock,
            model_kwargs={
                "max_tokens": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 250,
                "anthropic_version": "bedrock-2023-05-31"
            }
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Claude 3: {str(e)}")
        return None

def get_llama2_llm():
    """
    Initialize LLama 2 70B model
    Returns:
        BedrockChat: Configured LLama 2 LLM instance
    """
    llm = BedrockChat(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock,
        model_kwargs={'max_gen_len': 512}
    )
    return llm

# Prompt template for direct and concise answers
prompt_template = """
Use the following pieces of context to answer the question directly and concisely.
Begin your response with the relevant information without any prefacing statements.

Context:
{context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    """
    Get response from LLM using RAG
    Args:
        llm: Language model instance
        vectorstore_faiss: FAISS vector store
        query (str): User question
    Returns:
        str: LLM response
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Healthcare Policy Assistant",
        layout="wide"
    )
    
    st.header("Healthcare Policy Research Assistant")

    user_question = st.text_input("Ask a Question")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Vector store updated successfully!")

    if user_question:
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local(
                "faiss_index", 
                bedrock_embeddings, 
                allow_dangerous_deserialization=True
            )
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, user_question))

if __name__ == "__main__":
    main()