from motor.motor_asyncio import AsyncIOMotorClient
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
    return await db.memberships.find({"orgId": ObjectId(org_id)}).to_list(None)

# -------------------------------
# PROPOSAL
# -------------------------------

async def create_proposal(data: dict):
    result = await db.proposals.insert_one(data)
    return str(result.inserted_id)

async def create_proposal_choices(proposal_id: str, choices: list):
    docs = [
        {
            "proposalId": ObjectId(proposal_id),
            "value": choice
        }
        for choice in choices
    ]
    if docs:
        await db.proposalchoices.insert_many(docs)

async def create_proposal_data(entries: list):
    if entries:
        await db.proposaldatas.insert_many(entries)
