from dotenv import load_dotenv
from src.graph import build_claims_graph
from src.rag import index_policies

load_dotenv()


if __name__ == "__main__":
    print("Indexing clinical policy documents...")
    index_policies()

    print("Building claims agent graph...")
    graph = build_claims_graph()

    # Sample claim — Dupixent for atopic dermatitis
    initial_state = {
        "claim_id": "CLM-2024-00891",
        "member_id": "MBR-447821",
        "drug_name": "Dupixent",
        "diagnosis_code": "L20.9",       # Atopic dermatitis, unspecified
        "prescriber_npi": "1234567890",
        "quantity": 2,
        "days_supply": 28,
        "messages": [],
        "retrieved_policies": [],
        "eligibility_result": None,
        "clinical_decision": None,
        "confidence_score": None,
        "requires_human_review": False,
        "exception_reason": None,
        "audit_trail": [],
        "final_disposition": None,
    }

    print(f"\nProcessing claim {initial_state['claim_id']}...")
    print(f"Drug: {initial_state['drug_name']}  |  Diagnosis: {initial_state['diagnosis_code']}\n")

    result = graph.invoke(initial_state)

    print(f"\n{'='*52}")
    print(f"  DISPOSITION : {result['final_disposition']}")
    print(f"  Human review: {result['requires_human_review']}")
    if result.get("exception_reason"):
        print(f"  Exception   : {result['exception_reason']}")
    print(f"  Audit events: {len(result['audit_trail'])}")
    print(f"{'='*52}\n")
