from langchain_core.tools import tool
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime


class EligibilityInput(BaseModel):
    member_id: str
    drug_name: str
    diagnosis_code: str


class FormularyInput(BaseModel):
    drug_name: str
    plan_id: str = "PBM_STANDARD_2024"


@tool("check_member_eligibility", args_schema=EligibilityInput)
def check_member_eligibility(member_id: str, drug_name: str, diagnosis_code: str) -> dict:
    """Check if a member is eligible for a drug given their diagnosis.
    Returns eligibility status, coverage tier, and copay information."""
    # In production: call internal eligibility API
    # Demo: returns structured mock response
    return {
        "eligible": True,
        "coverage_tier": 2,
        "copay_amount": 45.00,
        "prior_auth_required": drug_name.lower() in ["humira", "keytruda", "dupixent"],
        "checked_at": datetime.utcnow().isoformat()
    }


@tool("lookup_formulary_status", args_schema=FormularyInput)
def lookup_formulary_status(drug_name: str, plan_id: str) -> dict:
    """Look up a drug's formulary status and any step therapy requirements."""
    return {
        "on_formulary": True,
        "tier": 2,
        "step_therapy_required": False,
        "quantity_limits": {"max_quantity": 30, "days_supply": 30},
        "alternatives": []
    }


@tool
def log_audit_event(event_type: str, details: str, claim_id: str) -> str:
    """Log a decision event to the HIPAA-compliant audit trail."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "claim_id": claim_id,
        "details": details,
        "logged_by": "claims_agent_v1"
    }
    # In production: write to append-only audit store
    print(f"[AUDIT] {json.dumps(entry)}")
    return f"Audit event logged: {event_type}"
