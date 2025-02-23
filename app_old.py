"""
This code is building an interactive Streamlit app that allows users to ask questions based on the content of PDF documents. It uses Amazon Bedrock for AI/LLM services to generate embeddings and retrieve relevant information from a vector database (FAISS).
"""

import json
import os
import sys
import boto3
import streamlit as st

## We will be using Titan Embeddings Model To generate Embedding
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

## Data Ingestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

## LLM Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set page configuration at the top
st.set_page_config("Chat PDF")

## Bedrock Clients
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# This initializes the embeddings model for text vectors using Amazon Titan
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)


## Data ingestion
def data_ingestion():
    """
    Loads all PDFs from a folder called data.
    Splits the documents into smaller chunks using the RecursiveCharacterTextSplitter
    """
    loader=PyPDFDirectoryLoader("data")
    documents=loader.load()

    # Character split works better with this PDF data set
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs


## Vector Embedding and vector store - Creating and Saving the Vector Store
def get_vector_store(docs):
    """
    Converts the document chunks into embeddings and stores them in a FAISS (Facebook AI Similarity Search) vector store.
    Saves the vector store locally as faiss_index.
    """
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")



# LLM Model Setup
def get_claude_llm():
    """
    Uses the Anthropic Claude model (ai21.jamba-1-5-mini-v1).
    Ensures payload structure adheres to Bedrock expectations.
    """
    llm = Bedrock(
        model_id="ai21.jamba-1-5-mini-v1:0", 
        client=bedrock,
        model_kwargs={
            "messages": [],  # Placeholder for prompt
            "max_tokens": 512, 
            "temperature": 0.8, 
            "top_p": 0.8
        }
    )
    return llm


def get_llama3_llm():
    """
    get_llama3_llm: Uses the Meta Llama 2 model (meta.llama2-70b-chat-v1).
    """
    ##create the LLama3 Model
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock, model_kwargs={'max_gen_len':512})
    return llm


# Prompt template
# The prompt is designed to:

# Use relevant document context to generate a detailed answer (at least 250 words).
# Avoid generating made-up answers if the model doesn’t know the answer.

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end but atleast summarize with 250 words with detailed explantions. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)



# Retrieval and Response
def get_response_llm(llm, vectorstore_faiss, query):
    """
    Uses RetrievalQA to query the vector store and retrieve relevant chunks based on similarity.
    Passes the retrieved chunks and query to the LLM to generate a response.
    """
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query":query})
    return answer['result']


def main():
    """
    Now here we are using strealit app which is designed to:

    1. Allow users to update or create a new vector store from PDFs using the "Vectors Update" button.
    2. Provide options to generate responses using Claude or Llama2 based on the user's question.
    """

    # Displays the title "Chat with PDF using AWS Bedrock💁" at the top of the web page
    st.header("Chat with PDF using AWS Bedrock💁")

    # Provides a text input box where users can type a question about the PDF files
    user_question = st.text_input("Ask a Question from the PDF Files")

    # Sidebar with "Vectors Update" Button
    # Adds a sidebar with the title "Update Or Create Vector Store".
    # The "Vectors Update" button performs the following when clicked:
        # Calls data_ingestion() to load all PDFs from a data directory.
        # Splits the text into smaller chunks for easier processing.
        # Embeds the text chunks using AWS Bedrock's Embedding Model and stores them in a FAISS Vector Store.
        # Saves the vector store locally under the name "faiss_index".
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    # When clicked, this button:
        # Loads the "faiss_index" vector store.
        # Retrieves the Claude LLM model (using AWS Bedrock).
        # Calls get_response_llm to:
            # Query the vector store for relevant documents based on the user's question.
            # Generate a detailed response from the Claude LLM.
        # Displays the response on the page.
    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            print("FAISS is loaded from the database...")
            llm = get_claude_llm()
            print("Claude LLM model is loaded...")
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")


    # When clicked, this button:
        # Loads the "faiss_index" vector store.
        # Retrieves the Llama3 LLM model (using AWS Bedrock).
        # Calls get_response_llm to:
            # Query the vector store for relevant documents based on the user's question.
            # Generate a detailed response from the Llama3 LLM.
        # Displays the response on the page.
    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llama3_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm, faiss_index, user_question))
            st.success("Done")



if __name__ == "__main__":
    main()













