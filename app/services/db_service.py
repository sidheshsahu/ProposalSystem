from motor.motor_asyncio import AsyncIOMotorClient
from typing import List
from bson import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()


MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "arbiter"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

# -------------------------------
# ORGANIZATION
# -------------------------------

async def get_all_organizations():
    orgs = await db.Organization.find({}).to_list(None)

    # Convert ObjectId to string
    for org in orgs:
        org["_id"] = str(org["_id"])

    return orgs

async def get_org_context(org_id: str):
    org = await db.Organization.find_one(
        {"_id": ObjectId(org_id)},
        {"context": 1, "name": 1}
    )
    return org

# -------------------------------
# MEMBERSHIPS
# -------------------------------

async def get_org_memberships(org_id: str):
    return await db.Membership.find({"orgId": ObjectId(org_id)}).to_list(None)

# -------------------------------
# PROPOSAL
# -------------------------------

async def create_proposal(data: dict):
    result = await db.Proposal.insert_one(data)
    return str(result.inserted_id)

async def create_proposal_choices(proposal_id: str, choices: List[str]):
    docs = [
        {
            "proposalId": ObjectId(proposal_id),
            "value": choice
        }
        for choice in choices
    ]
    if docs:
        await db.ProposalChoice.insert_many(docs)

async def create_proposal_data(entries: list):
    if entries:
        await db.ProposalData.insert_many(entries)


async def get_messages(user_id: str, proposal_id: str):
    messages = await db.Message.find(
        {
            "userId": ObjectId(user_id),
            "proposalId": ObjectId(proposal_id)
        }
    ).sort("createdAt", 1).to_list(None)

    history = []
    for m in messages[:-1]:
        role = "assistant" if m["author"] == "AI" else "user"
        history.append({
            "role": role,
            "content": m["text"]
        })

    query = messages[-1]["text"]

    return history, query

async def save_message(user_id: str, proposal_id: str, author: str, text: str):
    """
    Save a chat message with embedded user info for RAG
    """
    doc = {
        "author": author,  # "USER" or "AI"
        "text": text,
        "user": ObjectId(user_id),
        "proposalId": ObjectId(proposal_id),
        "createdAt": datetime.now()
    }
    await db.Message.insert_one(doc)


