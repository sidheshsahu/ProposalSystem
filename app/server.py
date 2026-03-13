from fastapi import FastAPI, UploadFile, File, Form
from typing import List
from bson import ObjectId
from datetime import datetime
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from core.summarizer import generic_summarizer
import json
import os
from services.db_service import (
    get_org_context,
    get_all_organizations,
    get_org_memberships,
    create_proposal,
    create_proposal_choices,
    create_proposal_data, get_messages,
    save_message
)
from core.document_store import get_document_store
from core.ingest import ingest_pdf
from services.outcome_service import run_outcome
from services.bias_service import run_bias
from services.bias_background import process_member_bias
from services.chat_service import run_chat

app = FastAPI(title="Proposal Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_namespace(filename: str) -> str:
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext.replace(" ", "_")

# document_store = get_document_store()

# -------------------------------
# EVALUATE PROPOSAL (PDF + JSON)
# -------------------------------

@app.get("/")
def health():
    return {"status": "API running"}

@app.get("/organizations")
async def list_organizations():
    """
    Returns all organizations in the database
    """
    orgs = await get_all_organizations()

    return {
        "status": "success",
        "count": len(orgs),
        "organizations": orgs
    }

@app.post("/evaluate")
async def evaluate_proposal(
    file: UploadFile = File(...),
    org_id: str = Form(...)
):
    """
    Receives:
    - PDF proposal
    - organization_id
    - notes JSON
    """
    
    # 2. Fetch organization context
    org_context = await get_org_context(org_id)

    if org_context is None:
        return {"error": "Organization not found"}

    namespace = get_namespace(file.filename)

    document_store = get_document_store(namespace=namespace)

    # 3. Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    # 4. Ingest PDF
    if document_store.count_documents() == 0:
        ingest_pdf(pdf_path, document_store)

    # 5. Combine context for LLM
    combined_notes = f"""
    ORGANIZATION CONTEXT:
    {org_context}
    """

    # 6. Run evaluation
    result = run_outcome(
        document_store=document_store,
        notes=combined_notes
    )

    document_store.delete_all_documents()

    try:
        result_json = json.loads(result)
    except:
        result_json = {"raw_output": result}

    return {
        "status": "success",
        "evaluation": result_json
    }

# -------------------------------
# BIAS EVALUATION (PDF + BIAS)
# -------------------------------
@app.post("/bias-evaluate")
async def bias_evaluate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    org_id: str = Form(...),
    title: str = Form(...),
    mediaUrl: str = Form(...),
    deadline: str = Form(...),
    proposalChoices: List[str] = Form(...)
):
    """
    Creates proposal + bias-aware summaries
    """

    choices = proposalChoices

    # -------------------------------
    # 2. Get org context
    # -------------------------------
    org = await get_org_context(org_id)
    if not org:
        return {"error": "Organization not found"}

    # -------------------------------
    # 3. Save PDF
    # -------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    

    # -------------------------------
    # 5. Generate generic summary
    # -------------------------------
    summary = generic_summarizer(
       pdf_path=pdf_path
    )

    # -------------------------------
    # 6. Create Proposal
    # -------------------------------
    proposal_data = {
        "title": title,
        "mediaUrl": mediaUrl,
        "deadline": datetime.fromisoformat(deadline),
        "summary": summary,
        "orgId": ObjectId(org_id),
        "proposalStatus": "UPCOMING",
        "createdAt": datetime.utcnow(),
        "type": "TRANSPARENT"
    }

    proposal_id = await create_proposal(proposal_data)

    # -------------------------------
    # 7. Create Proposal Choices
    # -------------------------------
    await create_proposal_choices(proposal_id, choices)

    namespace = get_namespace(proposal_id)

    document_store = get_document_store(namespace=namespace)

    if document_store.count_documents() == 0:
        ingest_pdf(pdf_path, document_store)

    # -------------------------------
    # 8. Background: generate bias summaries
    # -------------------------------
    background_tasks.add_task(
        process_member_bias,
        org_id,
        proposal_id,
        document_store
    )

    return {
        "status": "success",
        "proposal_id": proposal_id,
        "summary": summary
    }

# -------------------------------
# CHAT RAG (PDF + HISTORY + QUERY)
# -------------------------------
@app.post("/chat-evaluate")
async def chat_evaluate(
    user_id: str = Form(...),
    proposal_id: str = Form(...),
    query: str = Form(...)
):
    """
    history: chat history (JSON string or plain text)
    query: user message
    """

    # 1. Normalize chat history
    history = await get_messages(user_id, proposal_id)

   
    # namespace = get_namespace(proposal_id)
    namespace = "69b1865ea6e1ae6303a680ed"

    document_store = get_document_store(namespace=namespace)

    reply_text = run_chat(document_store=document_store, history=history, query=query)


    await save_message(user_id, proposal_id, author="AI", text=reply_text)

    return {
        "status":"success",
        "reply": reply_text
    }