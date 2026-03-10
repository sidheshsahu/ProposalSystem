from datetime import datetime, timezone
from services.db_service import get_org_memberships, create_proposal_data
from services.bias_service import run_bias
from bson import ObjectId
from services.db_service import db
import json
async def process_member_bias(org_id: str, proposal_id: str, proposal_summary: str):
    memberships = await get_org_memberships(org_id)

    proposal_entries = []

    for member in memberships:
        bias_text = member.get("bias", "")

        # Run AI bias evaluation
        result = run_bias(
            document_store=None,
            bias_text=f"""
            MEMBER BIAS:
            {bias_text}
            """
        )

        result_text = result["llm"]["replies"][0]

        try:
            result_json = json.loads(result_text)
        except:
            result_json = {"raw_output": result_text}

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
