import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_verbose, set_debug
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

use_serverless = True


set_debug(True)
set_verbose(True)


class AnswerConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super(AnswerConversationBufferMemory, self).save_context(inputs, {'response': outputs['answer']})


def get_conversation_chain():
    load_dotenv()
    embeddings = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX_NAME")
    api_key = os.getenv("PINECONE_API_KEY")
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)

    vectorstore = PineconeVectorStore(index, embeddings, "text")
    model_name = os.getenv("MODEL_NAME")  # e.g. 'gpt-4-1106-preview' - costs  $0.01  / 1K tokens
    chat_gpt = ChatOpenAI(model=model_name)
    llm = chat_gpt
    memory = AnswerConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_system_template = """Always answer in German! If you cannot get the answer from the following context answer with: 'Das kann ich aus deinen Dokumenten leider nicht beantworten.'
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(qa_system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    qa_system_prompt = ChatPromptTemplate.from_messages(messages)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        # some k limit necessary for google/flan because of input size restrictions
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_system_prompt}
    )
    return conversation_chain


def handle_input(conversation, question):
    response = conversation({'question': question})
    print("\n\nQuellen: ")
    for i, doc in enumerate(response["source_documents"]):
        print("\n::::::::::::::: " + str(i + 1) + ". Quelle :::::::::::::::")
        print(doc.page_content.replace('\ufffe', ''))  # remove Unicode Byte Order Marks

    print("\n\n::::::::::::::: Antwort :::::::::::::::\n" + response["answer"])


def main():
    load_dotenv()
    file_name = os.getenv("FILE_NAME")
    conversation = get_conversation_chain()
    print("\nFrag mich was zu deinen Büchern")
    while True:
        user_question = input(">> ")
        if user_question:
            handle_input(conversation, user_question)
        else:
            break


if __name__ == "__main__":
    main()
