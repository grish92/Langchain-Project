from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from PyPDF2 import PdfReader


def get_vectorstore_from_url(website_url, pdf):
    if pdf is not None:
        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=200
        )
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
            pdf_chunks = text_splitter.split_text(
                text)

        vector_store = Chroma.from_texts(pdf_chunks, OpenAIEmbeddings())
    else:
        text_splitter = RecursiveCharacterTextSplitter()
        loader = WebBaseLoader(website_url)
        document = loader.load()
        document_chunks = text_splitter.split_documents(document)

        vector_store = Chroma.from_documents(
            document_chunks, OpenAIEmbeddings())

    return vector_store
