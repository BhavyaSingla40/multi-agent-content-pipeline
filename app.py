
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from pipeline import workflow


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Content Studio",
    page_icon="🤖",
    layout="wide"
)


# ==========================================
# SESSION STATE
# ==========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "stage" not in st.session_state:
    st.session_state.stage = "new"

if "result" not in st.session_state:
    st.session_state.result = {}

if "completed_agents" not in st.session_state:
    st.session_state.completed_agents = []


def get_config():
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id
        },
        "recursion_limit": 100
    }


def execute_workflow(input_value):
    """Run the graph and display agent progress."""

    with st.status(
        "AI agents are working...",
        expanded=True
    ) as status:

        try:
            for update in workflow.stream(
                input_value,
                get_config(),
                stream_mode="updates"
            ):
                for node_name in update:
                    if node_name == "__interrupt__":
                        continue

                    st.write(f"✅ {node_name.replace('_', ' ').title()}")

                    if node_name not in st.session_state.completed_agents:
                        st.session_state.completed_agents.append(node_name)

            snapshot = workflow.get_state(get_config())

            st.session_state.result = snapshot.values

            if "human_review" in snapshot.next:
                st.session_state.stage = "review"
                status.update(
                    label="Article ready for human review",
                    state="complete"
                )
            else:
                st.session_state.stage = "completed"
                status.update(
                    label="Workflow completed",
                    state="complete"
                )

        except Exception as error:
            status.update(
                label="Workflow failed",
                state="error"
            )
            st.error(f"Error: {error}")
            return False

    return True


# ==========================================
# HEADER
# ==========================================

st.title("🤖 Multi-Agent Content Studio")

st.caption(
    "Research • Write • Fact-check • Edit • Human Approval"
)

st.divider()


# ==========================================
# NEW CONTENT
# ==========================================

if st.session_state.stage == "new":

    st.subheader("Create New Content")

    with st.form("content_form"):

        topic = st.text_input(
            "Enter your topic",
            placeholder="Agentic AI in software development"
        )

        content_type = st.selectbox(
            "Content Type",
            ["blog", "newsletter"]
        )

        submitted = st.form_submit_button(
            "🚀 Generate Content",
            type="primary"
        )

    if submitted:

        if not topic.strip():
            st.warning("Please enter a topic.")

        else:
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.completed_agents = []

            initial_state = {
                "topic": topic.strip(),
                "content_type": content_type,
                "revisions": 0,
                "approved": False
            }

            if execute_workflow(initial_state):
                st.rerun()


# ==========================================
# AGENT STATUS
# ==========================================

if st.session_state.stage in ("review", "completed"):

    st.subheader("Agent Execution")

    agents = [
        ("researcher", "🔍 Researcher"),
        ("writer", "✍️ Writer"),
        ("fact_checker", "🛡️ Fact-Checker"),
        ("editor", "📝 Editor")
    ]

    columns = st.columns(4)

    for column, (name, label) in zip(columns, agents):
        with column:
            if name in st.session_state.completed_agents:
                st.success(f"✅ {label}")
            else:
                st.info(f"⏳ {label}")

    st.divider()


# ==========================================
# ARTICLE PREVIEW
# ==========================================

if st.session_state.stage in ("review", "completed"):

    state = st.session_state.result

    article = (
        state.get("final_article")
        or state.get("draft", "")
    )

    article_tab, research_tab, verification_tab = st.tabs(
        ["📄 Article", "🔍 Research", "🛡️ Fact-Check"]
    )

    with article_tab:
        st.markdown(article)

    with research_tab:
        st.markdown(state.get("research", "No research available."))

    with verification_tab:

        verified = state.get("verified", False)

        if verified:
            st.success("Automated verification passed.")
        else:
            st.warning(
                "Automated verification did not pass. "
                "Review carefully before approval."
            )

        st.markdown(
            state.get("fact_check", "No report available.")
        )


# ==========================================
# HUMAN REVIEW
# ==========================================

if st.session_state.stage == "review":

    st.divider()
    st.subheader("👤 Human Approval")

    st.info(
        "Review the article and source evidence "
        "before approving publication."
    )

    feedback = st.text_area(
        "Revision feedback (optional)",
        placeholder="Explain what you want to change..."
    )

    col1, col2 = st.columns(2)

    with col1:

        approve = st.button(
            "✅ Approve Article",
            type="primary",
            use_container_width=True
        )

    with col2:

        revise = st.button(
            "🔄 Request Changes",
            use_container_width=True
        )

    if approve or revise:

        if revise and not feedback.strip():
            st.warning(
                "Please describe the changes you want."
            )

        else:
            decision = {
                "approved": bool(approve),
                "feedback": feedback.strip()
            }

            if execute_workflow(
                Command(resume=decision)
            ):
                st.rerun()


# ==========================================
# COMPLETED ARTICLE
# ==========================================

if st.session_state.stage == "completed":

    state = st.session_state.result

    st.success("🎉 Your article has been approved and saved!")

    output_path = state.get("output_path")

    if output_path and Path(output_path).is_file():

        article_content = Path(
            output_path
        ).read_text(encoding="utf-8")

        st.download_button(
            label="⬇️ Download Markdown",
            data=article_content,
            file_name=Path(output_path).name,
            mime="text/markdown"
        )


# ==========================================
# START NEW PROJECT
# ==========================================

if st.session_state.stage in ("review", "completed"):

    st.divider()

    if st.button("➕ Start New Article"):
        st.session_state.thread_id = None
        st.session_state.stage = "new"
        st.session_state.result = {}
        st.session_state.completed_agents = []
        st.rerun()
