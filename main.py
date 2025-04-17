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
    model = ChatOpenAI(model_name="gpt-4")
    # --- Add tool definition ---
    def add_tool(a: int, b: int) -> str:
        return str(a + b)

    # --- Add tool usage in chain context ---
    from langchain.tools import Tool
    tools = [
        Tool(
            name="add",
            func=lambda x: add_tool(x["a"], x["b"]),
            description="Add two numbers. Input: {\"a\": int, \"b\": int}"
        )
    ]

    # If you want to use tools in the chain, you need to use an agent or similar pattern.
    # For now, just expose the function in the context for prompt use.

    chain = (
        {"context": retriever, "question": RunnablePassthrough(), "add": add_tool}
        | prompt
        | model
        | StrOutputParser()
    )


except ValueError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-rooms/")
async def add_rooms(rooms: list[DocumentModel]):
    try:
        docs = [
            Document(
                page_content=doc.room_name,
                metadata={
                    **doc.metadata,
                },
            )
            for doc in rooms
        ]
        ids = (
            await pgvector_store.aadd_rooms(docs)
            if isinstance(pgvector_store, AsnyPgVector)
            else pgvector_store.add_rooms(docs)
        )
        return {"message": "rooms added successfully", "ids": ids}
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


@app.post("/get-rooms-by-ids/", response_model=list[DocumentResponse])
async def get_rooms_by_ids(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            rooms = await pgvector_store.get_rooms_by_ids(ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            rooms = pgvector_store.get_rooms_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return rooms
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-room/")
async def delete_rooms(ids: list[str]):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            existing_ids = await pgvector_store.get_all_ids()
            await pgvector_store.delete(ids=ids)
        else:
            existing_ids = pgvector_store.get_all_ids()
            pgvector_store.delete(ids=ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        return {"message": f"{len(ids)} rooms deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-rooms/", response_model=list[DocumentResponse])
async def get_rooms(skip: int = 0, limit: int = 10):
    try:
        if isinstance(pgvector_store, AsnyPgVector):
            ids = await pgvector_store.get_all_ids()
            # Apply pagination
            paginated_ids = ids[skip:skip+limit]
            langchain_docs = await pgvector_store.get_rooms_by_ids(paginated_ids)
        else:
            ids = pgvector_store.get_all_ids()
            # Apply pagination
            paginated_ids = ids[skip:skip+limit]
            langchain_docs = pgvector_store.get_rooms_by_ids(paginated_ids)
        
        # Convert LangChain Document objects to DocumentResponse objects
        rooms = []
        for doc in langchain_docs:
            metadata = doc.metadata
            rooms.append(DocumentResponse(
                room_name=metadata.get("room_name", ""),
                room_number=metadata.get("room_number", ""),
                description=metadata.get("description", ""),
                room_size=metadata.get("room_size", 0.0),
                image_url=metadata.get("image_url"),
                is_booked=metadata.get("is_booked", False),
                metadata=metadata
            ))
            
        return rooms
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

