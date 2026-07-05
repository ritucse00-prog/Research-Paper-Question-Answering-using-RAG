import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from pydantic import BaseModel,Field

from dotenv import load_dotenv
load_dotenv()

## load the groq api key
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
groq_api_key=os.getenv('GROQ_API_KEY')

## creating llm model
llm = ChatGroq(groq_api_key=groq_api_key, model='Gemma-7b-It')

## structured output using pydantic
class output_struc(BaseModel):
    introduction : str = Field(description="Provide a brief introduction about the asked query.")
    definition : str = Field(description="Provide a precise definition of the asked query topic.")
    explanation : str = Field(description="Provide a detailed context.")

structured_llm = llm.with_structured_output(output_struc)

## prompt
chat_prompt=ChatPromptTemplate(
    [
        ('system',"Answer the questions based on the provided context only."
        "Please provide the most accurate response based on the context."
        "context: {context}"),
        ("user", "input:{input}")
    ]
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        st.session_state.loader = PyPDFDirectoryLoader('research_papers')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("Research Paper Question Answering System using RAG")
user_prompt=st.text_input("Enter your query from the research paper")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector database is ready")

if user_prompt:
    retriver = st.session_state.vectors.as_retriever()
    rag_chain=(
        {
            "context": retriver | format_docs,
            "input" : RunnablePassthrough()
        }
        | chat_prompt
        | structured_llm
    )
    response = rag_chain.invoke(user_prompt)
