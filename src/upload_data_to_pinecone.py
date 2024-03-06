import os
from uuid import uuid4

from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader


def main():
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY')
    pinecone = Pinecone(api_key=api_key)

    load_dotenv()
    file_path = os.getenv("FILE_PATH")
    file_name = os.getenv("FILE_NAME")
    chunks = PyPDFium2Loader(file_path=file_path + file_name).load_and_split(CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    ))
    book_title = file_name.replace(".pdf", "")

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # create individual metadata dicts for each chunk
    metadata_dicts = [{
        "chunk": j, "text": chunk.page_content, "book_title": book_title, "page": chunk.metadata["page"]
    } for j, chunk in enumerate(chunks)]
    ids = [str(uuid4()) for _ in range(len(chunks))]
    embeds = embed.embed_documents([chunk.page_content for chunk in chunks])
    with pinecone.Index(os.getenv("PINECONE_INDEX_NAME")) as index:
        index.upsert(vectors=zip(ids, embeds, metadata_dicts))


if __name__ == "__main__":
    main()
