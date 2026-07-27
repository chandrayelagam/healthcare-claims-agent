# Healthcare Claims Agent

A multi-agent LangGraph pipeline that automates pharmacy benefit management (PBM)
claim adjudication — grounded in clinical policy documents via RAG, with
exception-aware routing and full audit trail design for HIPAA-regulated environments.

## Architecture

```
Claim Input
    │
    ▼
┌─────────────────────────────────────────────────┐
│              LangGraph State Machine             │
│                                                  │
│  [policy_retriever] → [eligibility_checker]     │
│         │                      │                │
│         ▼                      ▼                │
│  [clinical_reasoner] → [exception_router]       │
│         │                      │                │
│         ▼                      ▼                │
│    [auto_adjudicate]    [human_review_queue]    │
│         │                                       │
│         ▼                                       │
│    [audit_logger]                               │
└─────────────────────────────────────────────────┘
```

## What this demonstrates

- **Stateful multi-agent orchestration** using LangGraph — each node handles one concern, state flows through the graph with full type safety
- **RAG-grounded decisioning** — clinical policy documents are chunked, embedded, and retrieved at runtime to ground LLM outputs in authoritative sources
- **MCP-style tool interfaces** — internal workflow APIs exposed as structured tools with input validation and typed outputs
- **Exception-aware routing** — agents detect edge cases and route to human review rather than hallucinating a decision
- **Audit trail by design** — every decision, retrieved document, and tool call is logged with timestamps for HIPAA compliance

## Stack

- **LangGraph** — stateful agent graph orchestration
- **LangChain** — LLM abstraction + tool definitions
- **Groq** (demo) / **AWS Bedrock** (production) — LLM backend
- **FAISS** — local vector store for policy document retrieval
- **sentence-transformers** — local embeddings (no API cost)

## Quick start

```bash
git clone https://github.com/chandrayelagam/healthcare-claims-agent
cd healthcare-claims-agent
pip install -r requirements.txt
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)
python main.py
```

## Production mapping

This pattern maps directly to real PBM workflows:

| Demo component | Production equivalent |
|---|---|
| `policy_retriever` | Clinical formulary + PA criteria RAG |
| `eligibility_checker` | Member eligibility API tool call |
| `clinical_reasoner` | LLM-assisted PA decision support |
| `exception_router` | Prior auth exception queue routing |
| `audit_logger` | HIPAA-compliant decision audit trail |

In production (Cigna-Evernorth), this pattern reduced manual review touchpoints in high-volume processing pipelines using AWS Bedrock.

## Author

Chandra Kiran Yelagam — Agentic AI Architect
[LinkedIn](https://linkedin.com/in/chandrayelagam)
