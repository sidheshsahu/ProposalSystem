from datetime import datetime, timezone
from services.db_service import get_org_memberships, create_proposal_data
from services.bias_service import run_bias
from bson import ObjectId
from services.db_service import db
import json
import os
from core.document_store import get_document_store

async def process_member_bias(org_id: str, proposal_id: str, namespace: str):
    memberships = await get_org_memberships(org_id)

    proposal_entries = []

    document_store = get_document_store(namespace=namespace)

    for member in memberships:
        bias_text = member.get("bias", "")

        # Run AI bias evaluation
        result = run_bias(
            
            document_store=document_store,
            bias_text=f"""
            MEMBER BIAS:
            {bias_text}
            """
        )

        try:
            result_json = json.loads(result)
        except:
            result_json = {"raw_output": result}

        proposal_entries.append({
            "summary": result_json.get("reason", ""),
            "vote": result_json.get("final_vote", ""),
            "userId": member["userId"],
            "proposalId": ObjectId(proposal_id)
        })

    await create_proposal_data(proposal_entries)

    

    await db.Proposal.update_one(
        {"_id": ObjectId(proposal_id)},
        {"$set": {"proposalStatus": "ACTIVE",
        "startTime": datetime.now(timezone.utc).isoformat()}}
    )
