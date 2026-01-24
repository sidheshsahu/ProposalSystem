from fastapi import FastAPI
from pydantic import BaseModel

from biasPredictor import bias_summary
from chat import run_rag_chat

app = FastAPI(title="Course Proposal RAG API")

# -------------------------------
# Request Schema
# -------------------------------
class QueryRequest(BaseModel):
    query: str

# -------------------------------
# CSE Voting Endpoint
# -------------------------------
@app.post("/cse_vote")
def cse_vote(request: QueryRequest):
    result = bias_summary(request.query)
    return {"result": result}

# -------------------------------
# RAG Chat Endpoint
# -------------------------------
@app.post("/rag_chat")
def rag_chat(request: QueryRequest):
    answer, history = run_rag_chat(request.query)
    return {
        "answer": answer,
        "chat_history": history
    }
