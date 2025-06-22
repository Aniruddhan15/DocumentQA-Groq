import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st 

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

groq_api =os.environ['GROQ_API_KEY']

#step 1 loading the document

doc_loader = PyPDFLoader('LS.pdf')
document = doc_loader.load()

#step 2: chunks
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
document_chunk = text_splitter.split_documents(document)

model = ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_api)

#step 3: storing in vector store
embed = OllamaEmbeddings(model='mistral:latest')

vectorstore = FAISS.from_documents(documents=document_chunk, embedding=embed)

# step 4: prompt

promt = ChatPromptTemplate.from_template(
    """ 
    You are a helpful assistant that answers questions based on the provided 
    <context> 
    {context}.
    <context>
    Question: {query} from user.
    
    Answer: ..
    """
)

#step 5: create document chain
from langchain.chains.combine_documents import create_stuff_documents_chain

document_chain = create_stuff_documents_chain(llm=model, prompt=promt)

#step 6: retriever
retriever = vectorstore.as_retriever()

from langchain.chains import create_retrieval_chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)



# 2nd method


import os 
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

import streamlit as st
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langchain_core .prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings

st.title('Document Question Answering with Groq')

llm = ChatGroq(model='gemma2-9b-It', groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """ 
    You are a helpful assistant that answers questions based on the provided {context}.
    Question: {query}
    
    Answer: ..
    """
)

def vector_embeddings():
    
    if "vectors" not in st.session_state:
        
        st.session_state.loader = PyPDFLoader('LS.pdf')
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
        st.session_state.docs_split = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.emb = OllamaEmbeddings(model='mistral:latest')
        st.session_state.vectors = Chroma.from_documents(
            documents=st.session_state.docs_split,
            embedding=st.session_state.emb
        )
    return st.session_state.vectors

prompt1 = st.text_input('Enter your question:', key='prompt1')

submit = st.button('Submit', key='submit')

if submit:
    vector_embeddings()
    st.write('Retrieving relevant documents...')
    

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    response = retrieval_chain.invoke({"input": prompt1})
    st.write('Response:', response['answer'])

