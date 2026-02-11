from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import json

from core.document_store import get_document_store
from core.ingest import ingest_pdf
from services.outcome_service import run_outcome
from services.bias_service import run_bias
from services.chat_service import run_chat

app = FastAPI(title="Proposal Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

document_store = get_document_store()

# -------------------------------
# EVALUATE PROPOSAL (PDF + JSON)
# -------------------------------
@app.post("/evaluate")
async def evaluate_proposal(
    file: UploadFile = File(...),
    notes: str = Form(...)
):
    """
    notes: JSON string from Node backend
    """

    # 1. Parse notes JSON
    try:
        notes_data = json.loads(notes)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in notes"}

    # 2. Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # 3. Ingest PDF
    ingest_pdf(pdf_path, document_store)

    # 4. Run outcome pipeline
    result = run_outcome(
        document_store=document_store,
        notes=json.dumps(notes_data, indent=2)
    )

    # 5. Return result
    return {
        "status": "success",
        "evaluation": result
    }

# -------------------------------
# BIAS EVALUATION (PDF + BIAS)
# -------------------------------
@app.post("/bias-evaluate")
async def bias_evaluate(
    file: UploadFile = File(...),
    bias: str = Form(...)
):
    """
    bias: JSON or plain text describing stakeholder bias
    """

    # 1. Parse bias (if JSON)
    try:
        bias_data = json.loads(bias)
        bias_text = json.dumps(bias_data, indent=2)
    except json.JSONDecodeError:
        bias_text = bias  # allow plain text too

    # 2. Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # 3. Ingest proposal PDF
    ingest_pdf(pdf_path, document_store)

    # 4. Run bias pipeline
    result = run_bias(
        document_store=document_store,
        bias_text=bias_text
    )

    # 5. Return result
    return {
        "status": "success",
        "bias_evaluation": result
    }

# -------------------------------
# CHAT RAG (PDF + HISTORY + QUERY)
# -------------------------------
@app.post("/chat-evaluate")
async def chat_evaluate(
    file: UploadFile = File(...),
    history: str = Form(""),
    query: str = Form(...)
):
    """
    history: chat history (JSON string or plain text)
    query: user message
    """

    # 1. Normalize chat history
    try:
        history_data = json.loads(history)
        history_text = json.dumps(history_data, indent=2)
    except json.JSONDecodeError:
        history_text = history

    # 2. Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # 3. Ingest PDF
    ingest_pdf(pdf_path, document_store)

    # 4. Run chat pipeline
    reply = run_chat(
        document_store=document_store,
        history=history_text,
        query=query
    )

    # 5. Return reply
    return {
        "status": "success",
        "reply": reply
    }
