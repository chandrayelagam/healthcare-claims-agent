from langgraph.graph import StateGraph, END
from .state import ClaimState
from .nodes import (
    policy_retriever,
    eligibility_checker,
    clinical_reasoner,
    exception_router,
    auto_adjudicate,
    human_review_queue,
)


def build_claims_graph() -> StateGraph:
    graph = StateGraph(ClaimState)

    # Register nodes
    graph.add_node("policy_retriever", policy_retriever)
    graph.add_node("eligibility_checker", eligibility_checker)
    graph.add_node("clinical_reasoner", clinical_reasoner)
    graph.add_node("auto_adjudicate", auto_adjudicate)
    graph.add_node("human_review_queue", human_review_queue)

    # Linear flow through the pipeline
    graph.set_entry_point("policy_retriever")
    graph.add_edge("policy_retriever", "eligibility_checker")
    graph.add_edge("eligibility_checker", "clinical_reasoner")

    # Conditional routing at the exception boundary
    graph.add_conditional_edges(
        "clinical_reasoner",
        exception_router,
        {
            "auto_adjudicate": "auto_adjudicate",
            "human_review": "human_review_queue",
        },
    )

    graph.add_edge("auto_adjudicate", END)
    graph.add_edge("human_review_queue", END)

    return graph.compile()
