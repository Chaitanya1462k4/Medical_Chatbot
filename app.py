
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate

from src.helper import download_embeddings
from src.prompt import system_prompt


# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Medical Chatbot")

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Request schema
class ChatRequest(BaseModel):
    message: str


# Build the RAG pipeline
def build_rag_chain():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    if not openrouter_api_key or not pinecone_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY and PINECONE_API_KEY "
            "must be set in .env"
        )

    # Load embedding model from helper.py
    embeddings = download_embeddings()

    # Connect to the existing Pinecone index
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-chatbot",
        embedding=embeddings,
    )

    # Retrieve relevant documents
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Connect to OpenRouter
    llm = ChatOpenAI(
        model="nvidia/nemotron-3.5-lightning:free",
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Combine retrieved documents and generate an answer
    question_answer_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    # Create retrieval chain
    rag_chain = create_retrieval_chain(
        retriever,
        question_answer_chain,
    )

    return rag_chain


# Initialize the chain during application startup
rag_chain = None


@app.on_event("startup")
async def startup_event():
    global rag_chain
    rag_chain = build_rag_chain()
    print("Medical chatbot RAG pipeline initialized.")


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="The chatbot is still initializing.",
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        response = rag_chain.invoke({"input": message})

        return {
            "answer": response["answer"],
        }

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a response. Please try again.",
        )
