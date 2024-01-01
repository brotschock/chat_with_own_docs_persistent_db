from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFium2Loader
import os

# from langchain.globals import set_verbose, set_debug
# set_debug(True)
# set_verbose(True)


def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()  # ada is OpenAI's only embedding model (very cheap)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    model_name=os.getenv("MODEL_NAME")  # e.g. 'gpt-4-1106-preview' - costs  $0.01  / 1K tokens
    llm = ChatOpenAI(model=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_input(conversation, question):
    response = conversation({'question': question})
    print('AI: ' + response["answer"])


def main():
    load_dotenv()
    file_path = os.getenv("FILE_PATH")
    file_name = os.getenv("FILE_NAME")
    chunks = PyPDFium2Loader(file_path=file_path + file_name).load_and_split(CharacterTextSplitter(
        separator=". ",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    ))
    vectorstore = get_vectorstore(chunks)
    conversation = get_conversation_chain(vectorstore)
    print("\nFrag mich was zu " + file_name)
    while True:
        user_question = input(">> ")
        if user_question:
            handle_input(conversation, user_question)
        else:
            break


if __name__ == "__main__":
    main()
