from services.db_service import get_org_memberships, create_proposal_data
from services.bias_service import run_bias
from bson import ObjectId

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

            PROPOSAL SUMMARY:
            {proposal_summary}
            """
        )

        proposal_entries.append({
            "summary": result.get("summary", ""),
            "vote": result.get("suggested_vote", ""),
            "userId": member["userId"],
            "proposalId": ObjectId(proposal_id)
        })

    await create_proposal_data(proposal_entries)

    # 🔥 Activate proposal after processing
    from services.db_service import db
    await db.proposals.update_one(
        {"_id": ObjectId(proposal_id)},
        {"$set": {"proposalStatus": "ACTIVE"}}
    )
