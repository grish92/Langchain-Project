import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from src.vectorestore import get_vectorstore_from_url
from src.retrieval import get_context_retriever_chain, get_conversational_rag_chain


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(
        st.session_state.vector_store)

    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


def handle_user_input(user_query, website_url, pdf):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot.How I can help you?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(
            website_url, pdf)

    if user_query is not None and user_query != '':
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
