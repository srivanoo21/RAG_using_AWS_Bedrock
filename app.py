import json
import os
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


# Function to invoke Claude model using AWS Bedrock
def get_claude_response(prompt_data):
    payload = {
        "messages": [
            {"role": "user", "content": prompt_data}
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.8
    }
    body = json.dumps(payload)
    response = bedrock.invoke_model(
        body=body,
        modelId="ai21.jamba-1-5-mini-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("choices")[0].get("message").get("content")


# Function to invoke Llama 3 model using AWS Bedrock
def get_llama3_response(prompt_data):
    payload = {
        "prompt": f"[INST] {prompt_data} [/INST]",
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }
    body = json.dumps(payload)
    response = bedrock.invoke_model(
        body=body,
        modelId="meta.llama3-8b-instruct-v1:0",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("generation", "No response generated.")



# Initialize Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)


## Data ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs



## Vector Embedding and vector store
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")



# Retrieval and Response
def get_response_claude(vectorstore_faiss, query):
    retrieved_docs = vectorstore_faiss.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know.
    <context>
    {context}
    </context>
    
    Question: {query}
    
    Assistant:
    """
    return get_claude_response(prompt)


# Retrieval and Response for Llama3
def get_response_llama3(vectorstore_faiss, query):
    retrieved_docs = vectorstore_faiss.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""
    Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know.
    <context>
    {context}
    </context>
    
    Question: {query}
    
    Assistant:
    """
    return get_llama3_response(prompt)


def main():
    st.header("Chat with PDF using AWS Bedrock💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Claude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            st.write(get_response_claude(faiss_index, user_question))
            st.success("Done")

    elif st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            st.write(get_response_llama3(faiss_index, user_question))
            st.success("Done")



if __name__ == "__main__":
    main()
