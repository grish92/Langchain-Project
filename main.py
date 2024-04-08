import streamlit as st
from src.response import handle_user_input
from dotenv import load_dotenv


load_dotenv()


st.set_page_config(page_title="Chat with website", page_icon='')

st.title("Chat with website")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("website URL")
    pdf = st.file_uploader('Upload PDf', type='pdf')

if pdf is None or website_url is None or website_url == '':
    st.info("Please enter a website URL")
else:
    user_query = st.chat_input("Type your message here ...")
    handle_user_input(user_query, website_url, pdf)
