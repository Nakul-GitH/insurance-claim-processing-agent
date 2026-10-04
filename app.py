import os
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

os.environ.pop("SSL_CERT_FILE", None)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY was not found. Please check the .env file.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# ---------------------------------------------------------
# Data Models
# ---------------------------------------------------------

class InsuranceClaim(BaseModel):
    claim_id: str
    policy_id: str
    claimant_name: str
    policy_status: str
    policy_expiry_date: date
    claim_date: date
    claim_type: str
    claim_amount: float = Field(gt=0)
    required_documents: list[str] = Field(default_factory=list)
    submitted_documents: list[str] = Field(default_factory=list)


class ClaimProcessingState(TypedDict, total=False):
    claim: InsuranceClaim
    document_status: str
    missing_documents: list[str]
    eligibility_status: str
    eligibility_reason: str
    fraud_risk_level: str
    fraud_indicators: list[str]
    claim_summary: str
    final_decision: str
    human_review_required: bool
    human_review_reason: str


class FraudAssessment(BaseModel):
    risk_level: str
    indicators: list[str]


# ---------------------------------------------------------
# Agents
# ---------------------------------------------------------

def document_verification_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    claim = state["claim"]

    required_documents = {
        document.strip().lower()
        for document in claim.required_documents
    }

    submitted_documents = {
        document.strip().lower()
        for document in claim.submitted_documents
    }

    missing_documents = sorted(
        required_documents - submitted_documents
    )

    document_status = (
        "Incomplete"
        if missing_documents
        else "Complete"
    )

    return {
        "document_status": document_status,
        "missing_documents": missing_documents
    }


def eligibility_check_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    claim = state["claim"]

    if claim.policy_status.strip().lower() != "active":
        return {
            "eligibility_status": "Not Eligible",
            "eligibility_reason": (
                "The insurance policy is not active."
            )
        }

    if claim.claim_date > claim.policy_expiry_date:
        return {
            "eligibility_status": "Not Eligible",
            "eligibility_reason": (
                "The claim was submitted after the policy expiry date."
            )
        }

    if claim.claim_amount <= 0:
        return {
            "eligibility_status": "Not Eligible",
            "eligibility_reason": (
                "The claim amount must be greater than zero."
            )
        }

    return {
        "eligibility_status": "Eligible",
        "eligibility_reason": (
            "The policy is active and the claim satisfies "
            "the basic eligibility checks."
        )
    }


def fraud_detection_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    claim = state["claim"]

    if claim.claim_amount > 25000:
        amount_risk = "High"
    elif claim.claim_amount >= 10000:
        amount_risk = "Medium"
    else:
        amount_risk = "Low"

    structured_llm = llm.with_structured_output(
        FraudAssessment
    )

    prompt = f"""
You are an insurance fraud risk assessment assistant.

Assess the following insurance claim for potential
fraud indicators.

Claim ID: {claim.claim_id}
Policy ID: {claim.policy_id}
Claim Type: {claim.claim_type}
Claim Amount: ${claim.claim_amount:,.2f}
Policy Status: {claim.policy_status}
Document Status: {state.get("document_status", "Not assessed")}
Amount-Based Risk: {amount_risk}

Use the following risk thresholds:
- Below $10,000: Low
- $10,000 to $25,000: Medium
- Above $25,000: High

Return:
1. A risk level of Low, Medium, or High.
2. A list of specific indicators supporting the assessment.

Do not invent facts that are not present in the claim.
"""

    assessment = structured_llm.invoke(prompt)

    return {
        "fraud_risk_level": assessment.risk_level,
        "fraud_indicators": assessment.indicators
    }


def claim_summary_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    claim = state["claim"]

    prompt = f"""
You are an insurance claim processing assistant.

Generate a concise professional summary of the claim.

Claim ID: {claim.claim_id}
Policy ID: {claim.policy_id}
Claimant: {claim.claimant_name}
Claim Type: {claim.claim_type}
Claim Amount: ${claim.claim_amount:,.2f}
Policy Status: {claim.policy_status}
Policy Expiry Date: {claim.policy_expiry_date}
Claim Date: {claim.claim_date}

Assessment Results:
Document Status: {state.get("document_status")}
Missing Documents: {state.get("missing_documents", [])}
Eligibility Status: {state.get("eligibility_status")}
Eligibility Reason: {state.get("eligibility_reason")}
Fraud Risk Level: {state.get("fraud_risk_level")}
Fraud Indicators: {state.get("fraud_indicators", [])}

Do not invent information.
"""

    response = llm.invoke(prompt)

    return {
        "claim_summary": response.content
    }


def reject_claim_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    if state.get("document_status") != "Complete":
        reason = (
            "Required documents are missing: "
            + ", ".join(state.get("missing_documents", []))
            + "."
        )

    elif state.get("eligibility_status") != "Eligible":
        reason = state.get(
            "eligibility_reason",
            "The claim is not eligible."
        )

    else:
        reason = (
            "The claim does not meet the "
            "automatic processing requirements."
        )

    return {
        "final_decision": "Rejected",
        "human_review_required": False,
        "human_review_reason": reason
    }


def auto_approve_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    return {
        "final_decision": "Auto Approved",
        "human_review_required": False
    }


def human_approval_agent(
    state: ClaimProcessingState
) -> ClaimProcessingState:

    claim = state["claim"]

    review_reason = []

    if state.get("fraud_risk_level") == "High":
        review_reason.append(
            "High fraud risk detected."
        )

    if claim.claim_amount > 25000:
        review_reason.append(
            "Claim amount exceeds the automatic approval threshold."
        )

    if not review_reason:
        review_reason.append(
            "Claim requires manual review."
        )

    human_review_reason = " ".join(review_reason)

    approval_request = {
        "message": "Human approval is required.",
        "claim_id": claim.claim_id,
        "claimant_name": claim.claimant_name,
        "claim_amount": claim.claim_amount,
        "fraud_risk_level": state.get("fraud_risk_level"),
        "fraud_indicators": state.get(
            "fraud_indicators", []
        ),
        "eligibility_status": state.get(
            "eligibility_status"
        ),
        "document_status": state.get(
            "document_status"
        ),
        "claim_summary": state.get(
            "claim_summary", ""
        ),
        "review_reason": human_review_reason
    }

    human_decision = interrupt(approval_request)

    return {
        "final_decision": human_decision,
        "human_review_required": True,
        "human_review_reason": human_review_reason
    }


# ---------------------------------------------------------
# Routing
# ---------------------------------------------------------

def decision_router(
    state: ClaimProcessingState
) -> str:

    claim = state["claim"]

    if state["document_status"] != "Complete":
        return "reject"

    if state["eligibility_status"] != "Eligible":
        return "reject"

    if state["fraud_risk_level"] == "High":
        return "human_review"

    if claim.claim_amount > 25000:
        return "human_review"

    return "auto_approve"


def summary_router(
    state: ClaimProcessingState
) -> str:

    claim = state["claim"]

    if state.get("fraud_risk_level") == "High":
        return "human_approval"

    if claim.claim_amount > 25000:
        return "human_approval"

    return "auto_approve"


# ---------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------

workflow = StateGraph(ClaimProcessingState)

workflow.add_node(
    "document_verification",
    document_verification_agent
)

workflow.add_node(
    "eligibility_check",
    eligibility_check_agent
)

workflow.add_node(
    "fraud_detection",
    fraud_detection_agent
)

workflow.add_node(
    "decision_router",
    lambda state: state
)

workflow.add_node(
    "reject_claim",
    reject_claim_agent
)

workflow.add_node(
    "claim_summary",
    claim_summary_agent
)

workflow.add_node(
    "human_approval",
    human_approval_agent
)

workflow.add_node(
    "auto_approve",
    auto_approve_agent
)

workflow.add_edge(
    START,
    "document_verification"
)

workflow.add_edge(
    START,
    "eligibility_check"
)

workflow.add_edge(
    START,
    "fraud_detection"
)

workflow.add_edge(
    "document_verification",
    "decision_router"
)

workflow.add_edge(
    "eligibility_check",
    "decision_router"
)

workflow.add_edge(
    "fraud_detection",
    "decision_router"
)

workflow.add_conditional_edges(
    "decision_router",
    decision_router,
    {
        "reject": "reject_claim",
        "human_review": "claim_summary",
        "auto_approve": "claim_summary"
    }
)

workflow.add_edge(
    "reject_claim",
    END
)

workflow.add_conditional_edges(
    "claim_summary",
    summary_router,
    {
        "human_approval": "human_approval",
        "auto_approve": "auto_approve"
    }
)

workflow.add_edge(
    "human_approval",
    END
)

workflow.add_edge(
    "auto_approve",
    END
)

memory = MemorySaver()

claim_graph = workflow.compile(
    checkpointer=memory
)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="Insurance Claim Processing Agent",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Insurance Claim Processing Agent")
st.write(
    "AI-powered insurance claim assessment using LangGraph."
)

st.divider()

st.subheader("Claim Information")

col1, col2 = st.columns(2)

with col1:
    claim_id = st.text_input(
        "Claim ID",
        value="CLM-2001"
    )

    policy_id = st.text_input(
        "Policy ID",
        value="POL-6001"
    )

    claimant_name = st.text_input(
        "Claimant Name",
        value="John Smith"
    )

    claim_type = st.selectbox(
        "Claim Type",
        [
            "Vehicle Damage",
            "Property Damage",
            "Medical",
            "Travel"
        ]
    )

    claim_amount = st.number_input(
        "Claim Amount ($)",
        min_value=1.0,
        value=8500.0,
        step=500.0
    )

with col2:
    policy_status = st.selectbox(
        "Policy Status",
        ["Active", "Inactive"]
    )

    policy_expiry_date = st.date_input(
        "Policy Expiry Date",
        value=date(2027, 12, 31)
    )

    claim_date = st.date_input(
        "Claim Date",
        value=date.today()
    )

st.subheader("Required Documents")

available_documents = [
    "Claim Form",
    "Policy Document",
    "Identity Proof",
    "Repair Estimate",
    "Damage Assessment"
]

required_documents = st.multiselect(
    "Select required documents",
    available_documents,
    default=[
        "Claim Form",
        "Policy Document",
        "Identity Proof",
        "Repair Estimate"
    ]
)

uploaded_files = st.file_uploader(
    "Upload supporting documents",
    accept_multiple_files=True
)

uploaded_document_names = (
    [file.name for file in uploaded_files]
    if uploaded_files
    else []
)

st.caption(
    "For this assignment, document verification uses "
    "the document selections below. Uploaded files are "
    "accepted as supporting documents."
)

submitted_documents = st.multiselect(
    "Documents submitted with this claim",
    required_documents,
    default=required_documents
)

st.divider()

if st.button(
    "Process Claim",
    type="primary"
):

    claim = InsuranceClaim(
        claim_id=claim_id,
        policy_id=policy_id,
        claimant_name=claimant_name,
        policy_status=policy_status,
        policy_expiry_date=policy_expiry_date,
        claim_date=claim_date,
        claim_type=claim_type,
        claim_amount=claim_amount,
        required_documents=required_documents,
        submitted_documents=submitted_documents
    )

    initial_state = {
        "claim": claim,
        "document_status": "",
        "missing_documents": [],
        "eligibility_status": "",
        "eligibility_reason": "",
        "fraud_risk_level": "",
        "fraud_indicators": [],
        "claim_summary": "",
        "final_decision": "",
        "human_review_required": False,
        "human_review_reason": ""
    }

    config = {
        "configurable": {
            "thread_id": claim_id
        }
    }

    result = claim_graph.invoke(
        initial_state,
        config=config
    )

    st.session_state["claim_result"] = result
    st.session_state["claim_config"] = config

if "claim_result" in st.session_state:

    result = st.session_state["claim_result"]

    st.divider()
    st.subheader("Claim Assessment Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            result.get("document_status", "N/A")
        )

    with col2:
        st.metric(
            "Eligibility",
            result.get("eligibility_status", "N/A")
        )

    with col3:
        st.metric(
            "Fraud Risk",
            result.get("fraud_risk_level", "N/A")
        )

    if result.get("missing_documents"):
        st.warning(
            "Missing Documents: "
            + ", ".join(result["missing_documents"])
        )

    if result.get("eligibility_reason"):
        st.info(
            "Eligibility: "
            + result["eligibility_reason"]
        )

    if result.get("fraud_indicators"):
        with st.expander("Fraud Indicators"):
            for indicator in result["fraud_indicators"]:
                st.write("- " + indicator)

    if result.get("claim_summary"):
        st.subheader("Claim Summary")
        st.markdown(result["claim_summary"])

    if "__interrupt__" in result:

        st.warning(
            "⚠️ Human approval is required for this claim."
        )

        interrupt_data = result["__interrupt__"][0].value

        st.write(
            "**Review Reason:**",
            interrupt_data["review_reason"]
        )

        approval_col, rejection_col = st.columns(2)

        with approval_col:
            if st.button("Approve Claim"):
                final_result = claim_graph.invoke(
                    Command(resume="Approved by Human"),
                    config=st.session_state["claim_config"]
                )

                st.session_state["claim_result"] = final_result
                st.rerun()

        with rejection_col:
            if st.button("Reject Claim"):
                final_result = claim_graph.invoke(
                    Command(resume="Rejected by Human"),
                    config=st.session_state["claim_config"]
                )

                st.session_state["claim_result"] = final_result
                st.rerun()

    elif result.get("final_decision"):

        if result["final_decision"] == "Auto Approved":
            st.success(
                "Final Decision: Auto Approved"
            )

        elif result["final_decision"] == "Rejected":
            st.error(
                "Final Decision: Rejected"
            )
            st.write(
                "**Reason:**",
                result.get("human_review_reason", "")
            )

        else:
            st.success(
                f"Final Decision: {result['final_decision']}"
            )