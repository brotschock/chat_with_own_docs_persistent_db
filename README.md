# Chat with a PDF in your terminal
* first create a Pinecone index running **'create_pinecone_index.py'**
* then upload the semantic vectors of your chunked text  to a free Pinecone vector db with **'upload_data_to_pinecone.py'**
* now you can query your doc any time in natural language with **'run_chat.py'** and get an answer plus the relevant source chunks (harnessing a ChatGPT model of your choice)
