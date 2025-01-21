import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

import os
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant, please responds to the following queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question":question})
    return answer

st.title("My first chatbot with open AI")
api_key = st.sidebar.text_input("Enter your API Key:", type="password")

llm = st.sidebar.selectbox("Select the model", ["gemma:2b", "llama3", "phi3", "mxbai-embed-large"])
max_tokens = st.sidebar.slider("Max tokens", 50, 300, 50)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

st.write("Ask any question that comes to your mind")
q = st.text_input("Question")

if q:
    answer = generate_response(q, api_key, llm, temperature, max_tokens)
    st.write(answer)
else:
    st.write("Please provide a question to get an answer")
