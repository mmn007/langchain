import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
from dotenv import load_dotenv

print(os.environ)
load_dotenv()


llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based on the context only. Please provide the most accurate response based
    on the question.
    <context>
    {context}
    </context>
    Question: {input}

    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings() #OllamaEmbeddings() #Use OpenAI Embeddings to make it faster
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.documents = st.session_state.loader.load() #Complete document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


user_prompt = st.text_input("Enter the query to search in research papers")
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Your vector db is ready")

import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    stime = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    etime = time.process_time()
    print("Time taken: ", etime-stime)
    st.write(response['answer'])

    ## Streamlit expander
    with st.expander("Document similarity search"):
        for i,d in enumerate(response['context']):
            st.write(d.page_content)
