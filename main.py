import os

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from models import DocumentModel, DocumentResponse
from store import AsnyPgVector
from store_factory import get_vector_store

load_dotenv(find_dotenv())

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ui-langchain-fastapi.vercel.app","http://localhost:3000"],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' not found.")
    return value


try:
    USE_ASYNC = os.getenv("USE_ASYNC", "False").lower() == "true"
    if USE_ASYNC:
        print("Async project used")

    POSTGRES_DB = get_env_variable("POSTGRES_DB")
    POSTGRES_USER = get_env_variable("POSTGRES_USER")
    POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD")
    DB_HOST = get_env_variable("DB_HOST")
    DB_PORT = get_env_variable("DB_PORT")

    CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{DB_HOST}:{DB_PORT}/{POSTGRES_DB}"

    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings()

    mode = "async" if USE_ASYNC else "sync"
    pgvector_store = get_vector_store(
        connection_string=CONNECTION_STRING,
        embeddings=embeddings,
        collection_name="testcollection",
        mode=mode,
    )
    retriever = pgvector_store.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )


except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-documents/")
async def add_documents(documents: list[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "digest": doc.generate_digest(),
                    "room_number": doc.room_number,
                    "description": doc.description,
                    "room_size": doc.room_size,
                    "image_url": doc.image_url,
                    "is_booked": doc.is_booked
                },
            )
            for doc in documents
        ]
        ids = (
            await pgvector_store.aadd_documents(docs)
            if isinstance(pgvector_store, AsnyPgVector)
            else pgvector_store.add_documents(docs)
        )
        return {"message": "Documents added successfully", "ids": ids}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-all-ids/")
async def get_all_ids():
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            ids = await pgvector_store.get_all_ids()
        else:
            ids = pgvector_store.get_all_ids()

        return ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get-documents-by-ids/", response_model=list[DocumentResponse])
async def get_documents_by_ids(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            documents = await pgvector_store.get_documents_by_ids(ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            documents = pgvector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return documents
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-documents/")
async def delete_documents(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            await pgvector_store.delete(ids=ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            pgvector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return {"message": f"{len(ids)} documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Simple in-memory conversation memory
dialogue_memory = {}

@app.post("/chat/")
async def quick_response(msg: str, session_id: str):
    # Get history for session_id
    history = dialogue_memory.get(session_id, [])
    # Prepare context string from history
    context = "\n".join([f"User: {m['user']}\nAssistant: {m['assistant']}" for m in history]) if history else ""
    # Compose question with context
    question_with_context = f"{context}\nUser: {msg}" if context else msg
    # Get response from chain
    result = chain.invoke(question_with_context)
    # Save to memory
    history.append({"user": msg, "assistant": result})
    dialogue_memory[session_id] = history
    return result

@app.get("/get-documents/", response_model=list[DocumentResponse])
async def get_documents(skip: int = 0, limit: int = 10):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            ids = await pgvector_store.get_all_ids()
            # Apply pagination
            paginated_ids = ids[skip:skip+limit]
            documents = await pgvector_store.get_documents_by_ids(paginated_ids)
        else:
            ids = pgvector_store.get_all_ids()
            # Apply pagination
            paginated_ids = ids[skip:skip+limit]
            documents = pgvector_store.get_documents_by_ids(paginated_ids)

        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))