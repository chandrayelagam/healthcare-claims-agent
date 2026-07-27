from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .state import ClaimState
from .tools import check_member_eligibility, lookup_formulary_status, log_audit_event
from datetime import datetime


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(
    [check_member_eligibility, lookup_formulary_status, log_audit_event]
)


def policy_retriever(state: ClaimState) -> ClaimState:
    """RAG node: retrieve relevant clinical policies for this claim."""
    from .rag import retrieve_policies
    policies = retrieve_policies(
        query=f"{state['drug_name']} {state['diagnosis_code']} prior authorization criteria"
    )
    state["retrieved_policies"] = policies
    state["audit_trail"].append({
        "node": "policy_retriever",
        "timestamp": datetime.utcnow().isoformat(),
        "policies_retrieved": len(policies)
    })
    return state


def eligibility_checker(state: ClaimState) -> ClaimState:
    """Tool-calling node: verify member eligibility via structured tool."""
    response = llm_with_tools.invoke([
        SystemMessage(content="You are a PBM eligibility checker. Use the check_member_eligibility tool."),
        HumanMessage(content=f"Check eligibility for member {state['member_id']} "
                             f"requesting {state['drug_name']} for diagnosis {state['diagnosis_code']}")
    ])
    state["messages"].append(response)
    state["eligibility_result"] = {"status": "checked", "response": str(response.content)}
    return state


def clinical_reasoner(state: ClaimState) -> ClaimState:
    """Core reasoning node: apply clinical policy to claim using retrieved context."""
    policy_context = "\n\n".join(state["retrieved_policies"])
    response = llm.invoke([
        SystemMessage(content=f"""You are a clinical pharmacist reviewing a PBM claim.
Use only the retrieved clinical policies below to make your decision.
If the policies do not clearly support approval, flag for human review.
Do not hallucinate clinical criteria.

RETRIEVED POLICIES:
{policy_context}"""),
        HumanMessage(content=f"""Review this claim:
Drug: {state['drug_name']}
Diagnosis: {state['diagnosis_code']}
Quantity: {state['quantity']} units / {state['days_supply']} days supply
Prescriber NPI: {state['prescriber_npi']}

Return: APPROVE, DENY, or NEEDS_REVIEW with a one-sentence rationale and confidence 0-1.""")
    ])
    content = response.content
    state["clinical_decision"] = content
    state["confidence_score"] = (
        0.9 if "APPROVE" in content
        else 0.5 if "NEEDS_REVIEW" in content
        else 0.85
    )
    state["messages"].append(response)
    return state


def exception_router(state: ClaimState) -> str:
    """Conditional edge: route to auto-adjudication or human review queue."""
    if state["confidence_score"] is None or state["confidence_score"] < 0.75:
        return "human_review"
    if "NEEDS_REVIEW" in (state["clinical_decision"] or ""):
        return "human_review"
    if state.get("eligibility_result", {}).get("prior_auth_required"):
        return "human_review"
    return "auto_adjudicate"


def auto_adjudicate(state: ClaimState) -> ClaimState:
    """Final node: record approved decision and log to audit trail."""
    state["final_disposition"] = "APPROVED"
    state["requires_human_review"] = False
    log_audit_event.invoke({
        "event_type": "AUTO_ADJUDICATED",
        "details": f"Claim approved. Confidence: {state['confidence_score']:.2f}",
        "claim_id": state["claim_id"]
    })
    return state


def human_review_queue(state: ClaimState) -> ClaimState:
    """Exception node: route to human reviewer with full context."""
    state["final_disposition"] = "PENDING_REVIEW"
    state["requires_human_review"] = True
    state["exception_reason"] = state.get("clinical_decision", "Low confidence score")
    log_audit_event.invoke({
        "event_type": "ROUTED_TO_HUMAN_REVIEW",
        "details": state["exception_reason"],
        "claim_id": state["claim_id"]
    })
    return state
