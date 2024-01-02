import os
from typing import Dict, Any

import pinecone
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone


# from langchain.globals import set_verbose, set_debug
# set_debug(True)
# set_verbose(True)


class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs, {'response': outputs['answer']})


def get_conversation_chain():
    embeddings = OpenAIEmbeddings()
    index = pinecone.Index('my-books')

    vectorstore = Pinecone(index, embeddings, "text")
    model_name = os.getenv("MODEL_NAME")  # e.g. 'gpt-4-1106-preview' - costs  $0.01  / 1K tokens
    llm = ChatOpenAI(model=model_name)
    memory = AnswerConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain


def handle_input(conversation, question):
    response = conversation({'question': question})
    print("\n\nQuellen: ")
    for i, doc in enumerate(response["source_documents"]):
        print("\n::::::::::::::: " + str(i + 1) + ". Quelle :::::::::::::::")
        print(doc.page_content.replace('\ufffe', ''))  # remove Unicode Byte Order Marks

    print("\n\n::::::::::::::: Antwort :::::::::::::::" + response["answer"])


def main():
    load_dotenv()
    file_name = os.getenv("FILE_NAME")
    conversation = get_conversation_chain()
    print("\nFrag mich was zu " + file_name)
    while True:
        user_question = input(">> ")
        if user_question:
            handle_input(conversation, user_question)
        else:
            break


if __name__ == "__main__":
    main()
