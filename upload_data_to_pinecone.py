import os

import pinecone
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFium2Loader


def main():
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY')
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT')

    pinecone.init(api_key=api_key, environment=env)

    embeddings = OpenAIEmbeddings()  # ada is OpenAI's only embedding model (very cheap)

    load_dotenv()
    file_path = os.getenv("FILE_PATH")
    file_name = os.getenv("FILE_NAME")
    chunks = PyPDFium2Loader(file_path=file_path + file_name).load_and_split(CharacterTextSplitter(
        separator=". ",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    ))
    vectorstore = Pinecone.from_documents(chunks, embeddings, index_name='my-books')


if __name__ == "__main__":
    main()
