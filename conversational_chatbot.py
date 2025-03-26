#Import required libraries
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#initialize model
os.environ['Huggingface_api_key']=os.getenv('Huggingface_api_key')
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title('Conversational History Aware Q&A Chatbot')

api_key=st.text_input('Enter groq api key',type='password')

if api_key:
    llm=ChatGroq(model='llama-3.3-70b-versatile',api_key=api_key)
    session_id=st.text_input('Session ID',value='default session')
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader('Upload File',type='pdf',accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf='./temppdf'
            with open(temppdf,'wb') as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
                docs=PyPDFLoader(temppdf).load()
                documents.extend(docs)
        #split and create document
        splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        split_docs=splitter.split_documents(documents)
        vectorstores=FAISS.from_documents(split_docs,embeddings)
        retriever=vectorstores.as_retriever()

        #make history aware retriever   
        context_system_prompt='''
    You are a history aware assistant 
    so take reference context from chat history to give conversational answer'''

        contexualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',context_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contexualize_q_prompt)

        
        #make the model
        system_prompt='''
            you are a assistant for question-answering tasks.
            Use the following pieces  of retrieved context to answer.
            if you don't know the answer say i don't know and keep the answer consize
            {context}'''
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human','{input}')
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key='input',
            output_messages_key='answer',
            history_messages_key='chat_history'
        )

        user_input=st.text_input('message')
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {'input':user_input},
                config={
                    'configurable':{'session_id':session_id}
                }
            )
            st.write(response['answer'])
else:
    st.warning('Please enter groq api key')


                





