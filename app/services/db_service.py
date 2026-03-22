"""Database service for MongoDB operations.

This module handles all database interactions for the proposal system including:
- Organization and membership management
- Proposal creation and tracking
- Voting and outcome recording
- Chat message history persistence
"""
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List
from bson import ObjectId
from datetime import datetime, timezone
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
        "userId": ObjectId(user_id),
        "proposalId": ObjectId(proposal_id),
        "createdAt": datetime.now(timezone.utc)
    }
    await db.Message.insert_one(doc)



async def get_proposal_outcome(proposal_id: str):

    proposal = await db.Proposal.find_one(
        {"_id": ObjectId(proposal_id)},
        {"title":1,"summary":1,"orgId":1}
    )

    pipeline = [
        {
            "$match": {
                "proposalId": ObjectId(proposal_id)
            }
        },
        {
            "$group": {
                "_id": "$choiceId",
                "voteCount": {"$sum": "$voteValue"}
            }
        },
        {
            "$sort": {
                "voteCount": -1
            }
        }
    ]

    vote_results = await db.Vote.aggregate(pipeline).to_list(length=None)

    winning_choice = None
    winner_votes = 0
    total_votes = 0


    if not vote_results:
        return {
            "title": proposal["title"],
            "summary": proposal["summary"],
            "orgId": proposal["orgId"],
            "totalVotes": 0,
            "winnerVotes": 0,
            "winningChoice": None
        }
    
    top = vote_results[0]
    winner_choice_id = top["_id"]
    winner_votes = top["voteCount"]

    
    total_votes = sum(v["voteCount"] for v in vote_results)

   
    choice = await db.ProposalChoice.find_one(
        {"_id": winner_choice_id},
        {"value": 1}
    )

    if choice:
        winning_choice = choice["value"]

    return {
        "title": proposal["title"],
        "summary": proposal["summary"],
        "orgId": proposal["orgId"],
        "totalVotes": total_votes,     
        "winnerVotes": winner_votes,   
        "winningChoice": winning_choice 
    }


async def append_org_context(org_id: str, new_context: str):

    org = await db.Organization.find_one(
        {"_id": ObjectId(org_id)},
        {"context":1}
    )

    old_context = org.get("context","")

    updated_context = old_context + "\n" + new_context

    await db.Organization.update_one(
        {"_id": ObjectId(org_id)},
        {"$set": {"context": updated_context}}
    )