import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

use_serverless = True


def main():
    load_dotenv()
    # Load Pinecone API key
    api_key = os.getenv('PINECONE_API_KEY')
    # Set Pinecone environment. Find next to API key in console
    env = os.getenv('PINECONE_ENVIRONMENT')

    print(api_key + "    " + env)
    pinecone = Pinecone(api_key=api_key)
    spec = ServerlessSpec(cloud='aws', region='us-west-2')  # serverless is currently only available in us-west-2
    index_name = "jakobs-test-index"
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)
    # we create a new index
    pinecone.create_index(
        spec=spec,
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
    # wait for index to be initialized
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)


if __name__ == "__main__":
    main()
