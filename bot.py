# General Imports
import os
from time import sleep
from dotenv import load_dotenv

# UI & MongoDB Library
import streamlit as st
from pymongo import MongoClient

# Library Required for Vector Store
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# Since the Vector Database is already created, there is no need to regenerate the vector store
# from langchain_community.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader


# Library Required for LLMs and Chats
# for LLM
from langchain_openai import ChatOpenAI
# These are the commonly used chat messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

##

load_dotenv(override=True)

st.title("Simple Chatbot for CC3")
st.text("Ask a question!")


def search_chunks(query):
    search_result = st.session_state['retrieval'].invoke(query)
    context = []
    for r in search_result:
        context.append(r.page_content)

    instruction = "try to understand the userquery and answer based on the context given below:\n"
    return SystemMessage(content=f"{instruction}'context':{context}, 'userquery':{query}")


if "text_embedding" not in st.session_state:
    st.session_state['text_embedding'] = OpenAIEmbeddings(
        api_key=os.environ['OPENAI_API_KEY'])
    vectorDB = FAISS.load_local(
        "resources/db", st.session_state['text_embedding'], allow_dangerous_deserialization=True)
    st.session_state['retrieval'] = vectorDB.as_retriever(
        search_kwargs={"k": 5})

    st.session_state['llm'] = ChatOpenAI(
        model="gpt-3.5-turbo", api_key=os.environ['OPENAI_API_KEY'])

    persona = "You are a teaching assistant at for the course CC0003 at NTU."
    task = "your task is to answer student query about the Ethics and Civics in a Multicultural World."
    context = "the context will be provided based on the course information and notes along with the user query"
    condition = "If user ask any query beyond Ethics and Civics in a Multicultural World, tell the user you are not an expert of the topic the user is asking and say sorry. If you are unsure about certain query, say sorry and advise the user to contact the instructor at instructor@ntu.edu.sg"
    # any other things to add on

    # Constructing initial system message
    sysmsg = f"{persona} {task} {context} {condition}"
    st.session_state['conversations'] = [SystemMessage(content=sysmsg)]

    greetings = '''Hello my name is Bin, and I am a Automated Teaching Assistant for CC0003 - Ethics and Civics in a Multicultural World. I am here to help, feel free to ask any questions.'''
    st.session_state['conversations'].append(AIMessage(content=greetings))
    st.session_state['msgtypes'] = {
        HumanMessage: "Human", AIMessage: "AI", SystemMessage: "System"}


if 'conversations' in st.session_state:
    for conv in st.session_state['conversations']:
        if isinstance(conv, SystemMessage):
            continue
        role = st.session_state.msgtypes[type(conv)]
        with st.chat_message(role):
            st.markdown(conv.content)

if query := st.chat_input("Your Message"):
    st.chat_message("Human").markdown(query)
    st.session_state['conversations'].append(HumanMessage(content=query))

    context = search_chunks(query)
    templog = st.session_state['conversations'] + [context]
    response = st.session_state['llm'].invoke(templog)
    # response = st.session_state['llm'].invoke(st.session_state['conversations'])
    st.chat_message("AI").markdown(response.content)
    st.session_state['conversations'].append(response)

# st.markdown(st.session_state['conversations'])
