import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from rag_pipeline import OptimizedRAGSystem

load_dotenv(override=True)

os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "RAG-Practical"
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")

app = FastAPI()


if os.path.exists("medicare_vector_store"):
    print("Removing existing medicare_vector_store directory...")
    shutil.rmtree("medicare_vector_store")


# Load RAG system once at startup
rag = OptimizedRAGSystem(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    groq_model="llama3-70b-8192"
)

rag.build_vector_store("medicare.pdf")
rag.save_vector_store("medicare_vector_store")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    query = request.query.strip()

    if not query:
        return {
            "answer": "Please provide a valid, non-empty question.",
            "source_page": 0,
            "confidence_score": 0.0,
            "chunk_size": 0
        }

    try:
        result = rag.query(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
