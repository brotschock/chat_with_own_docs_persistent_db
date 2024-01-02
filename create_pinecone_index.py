import time
import pinecone
import os


def main():
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY')
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT')

    print(api_key + "    " + env)
    pinecone.init(api_key=api_key, environment=env)
    index_name = 'jakobs-books'
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)



if __name__ == "__main__":
    main()