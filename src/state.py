from typing import TypedDict, Annotated, List, Optional
from langgraph.graph.message import add_messages


class ClaimState(TypedDict):
    # Core claim data
    claim_id: str
    member_id: str
    drug_name: str
    diagnosis_code: str
    prescriber_npi: str
    quantity: int
    days_supply: int

    # Agent working memory
    messages: Annotated[list, add_messages]
    retrieved_policies: List[str]
    eligibility_result: Optional[dict]
    clinical_decision: Optional[str]
    confidence_score: Optional[float]

    # Routing
    requires_human_review: bool
    exception_reason: Optional[str]

    # Audit
    audit_trail: List[dict]
    final_disposition: Optional[str]  # APPROVED | DENIED | PENDING_REVIEW
