from src.helper import load_pdf, filter_to_minimal_docs, split_docs,download_embeddings 
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec





load_dotenv()




pinecone_api_key=os.getenv("PINECONE_API_KEY")

pc=Pinecone(api_key=pinecone_api_key)

extracted_data=load_pdf(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=split_docs(filter_data)

embeddings=download_embeddings()

index_name="medical-chatbot"

if not  pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")
    )

index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(
    index_name=index_name,
    embedding=embeddings,
    documents=text_chunks)