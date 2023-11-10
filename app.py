import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from styling import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

#Taking text from the pdf's 
def get_pdf_text(pdf_docs):
     text = ""
     for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
             text += page.extract_text()    

     return text

#Making Chunks using langchain         
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap= 200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

#Creating Embeddings using the chunks
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore= FAISS.from_texts(text = text_chunks, embedding = embeddings)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()
    st.set_page_config(page_title = "PDF's GPT" , page_icon=":books:")
    st.header("Chat With Your PDF's :books:")
    st.text_input("Ask Me Anything..")

    with st.sidebar:
        st.subheader("It reads your Documents")
        #Storing our pdfs in this
        pdf_docs = st.file_uploader("Import Your PDF's here", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                #get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
                
                #the chuncks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                
                #Create vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)

                #Conversation session saving
                

if __name__ == '__main__':
        main()