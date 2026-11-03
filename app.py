
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from pipeline import workflow
from database import (
    init_db,
    create_article,
    update_article_status,
    get_articles,
    get_article
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Agent Content Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()


# ============================================================
# 2. SESSION STATE
# ============================================================

DEFAULTS = {
    "thread_id": None,
    "stage": "new",
    "result": {},
    "completed_agents": [],
    "current_topic": ""
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

AGENT_NAMES = {
    "researcher": "Researcher",
    "writer": "Writer",
    "fact_checker": "Fact-Checker",
    "editor": "Editor"
}


def get_config(thread_id=None):
    """Get LangGraph configuration for a workflow."""

    thread_id = thread_id or st.session_state.thread_id

    return {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 100
    }


def get_completed_agents(state):
    """Determine completed agents from saved state."""

    mapping = {
        "researcher": "research",
        "writer": "draft",
        "fact_checker": "fact_check",
        "editor": "final_article"
    }

    return [
        agent
        for agent, field in mapping.items()
        if state.get(field)
    ]


def get_article_text(state):
    """Return the latest article version."""

    return (
        state.get("final_article")
        or state.get("draft")
        or ""
    )


def reset_workspace():
    """Reset UI without deleting saved articles."""

    st.session_state.thread_id = None
    st.session_state.stage = "new"
    st.session_state.result = {}
    st.session_state.completed_agents = []
    st.session_state.current_topic = ""


def restore_article(thread_id):
    """Load an existing article from SQLite checkpoints."""

    metadata = get_article(thread_id)

    if metadata is None:
        st.error("Article not found in the database.")
        return False

    try:
        snapshot = workflow.get_state(
            get_config(thread_id)
        )

        state = snapshot.values

        if not state:
            st.error(
                "No saved LangGraph checkpoint was found."
            )
            return False

        if "human_review" in snapshot.next:
            stage = "review"

        elif not snapshot.next and state.get("output_path"):
            stage = "completed"

        else:
            st.warning(
                "This workflow is not ready for review. "
                "It may have failed during generation."
            )
            return False

        st.session_state.thread_id = thread_id
        st.session_state.stage = stage
        st.session_state.result = state
        st.session_state.current_topic = state.get(
            "topic",
            metadata["topic"]
        )
        st.session_state.completed_agents = (
            get_completed_agents(state)
        )

        return True

    except Exception as error:
        st.error(f"Unable to restore article: {error}")
        return False


# ============================================================
# 4. LANGGRAPH EXECUTION
# ============================================================

def execute_workflow(input_value):
    """
    Run or resume the multi-agent workflow.
    Track completed agents and update article status.
    """

    thread_id = st.session_state.thread_id

    if not thread_id:
        st.error("No active workflow thread found.")
        return False

    with st.status(
        "🤖 AI agents are working...",
        expanded=True
    ) as status:

        try:
            for update in workflow.stream(
                input_value,
                get_config(),
                stream_mode="updates"
            ):

                for node_name in update.keys():

                    if node_name == "__interrupt__":
                        st.write(
                            "⏸️ Workflow paused for human review."
                        )
                        continue

                    display_name = AGENT_NAMES.get(
                        node_name,
                        node_name.replace("_", " ").title()
                    )

                    st.write(
                        f"✅ {display_name} completed"
                    )

                    if node_name in AGENT_NAMES:
                        if node_name not in (
                            st.session_state.completed_agents
                        ):
                            st.session_state.completed_agents.append(
                                node_name
                            )

            snapshot = workflow.get_state(
                get_config()
            )

            state = snapshot.values

            st.session_state.result = state
            st.session_state.completed_agents = (
                get_completed_agents(state)
            )

            # Waiting for approval
            if "human_review" in snapshot.next:

                st.session_state.stage = "review"

                update_article_status(
                    thread_id,
                    "pending_review"
                )

                status.update(
                    label="✅ Article ready for human review",
                    state="complete",
                    expanded=False
                )

            # Article approved and saved
            elif (
                not snapshot.next
                and state.get("approved")
                and state.get("output_path")
            ):

                st.session_state.stage = "completed"

                update_article_status(
                    thread_id,
                    "approved"
                )

                status.update(
                    label="✅ Article approved and saved",
                    state="complete",
                    expanded=False
                )

            else:
                raise RuntimeError(
                    "Workflow stopped unexpectedly. "
                    "No approval checkpoint or saved "
                    "output was found."
                )

            return True

        except Exception as error:

            update_article_status(
                thread_id,
                "failed"
            )

            status.update(
                label="❌ Workflow failed",
                state="error",
                expanded=True
            )

            st.error(
                f"Workflow error: {error}"
            )

            st.info(
                "Check your Gemini API quota, Tavily "
                "credits, and terminal output. "
                "The saved checkpoint may still be "
                "available for recovery."
            )

            return False


# ============================================================
# 5. SIDEBAR — ARTICLE LIBRARY
# ============================================================

with st.sidebar:

    st.title("📚 Content Library")

    st.caption(
        "Manage your AI-generated articles"
    )

    articles = get_articles()

    total_articles = len(articles)

    approved_count = sum(
        article["status"] == "approved"
        for article in articles
    )

    pending_count = sum(
        article["status"] == "pending_review"
        for article in articles
    )

    st.metric(
        "Total Articles",
        total_articles
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Approved",
            approved_count
        )

    with col2:
        st.metric(
            "Pending",
            pending_count
        )

    st.divider()

    if st.button(
        "➕ New Article",
        use_container_width=True,
        type="primary"
    ):
        reset_workspace()
        st.rerun()

    st.subheader("Recent Articles")

    search_query = st.text_input(
        "Search articles",
        placeholder="Search by topic..."
    )

    status_filter = st.selectbox(
        "Filter by status",
        [
            "All",
            "Approved",
            "Pending Review",
            "Generating",
            "Failed"
        ]
    )

    filtered_articles = []

    for article in articles:

        matches_search = (
            search_query.lower()
            in article["topic"].lower()
        )

        matches_status = (
            status_filter == "All"
            or article["status"].replace(
                "_", " "
            ).lower() == status_filter.lower()
        )

        if matches_search and matches_status:
            filtered_articles.append(article)

    if not filtered_articles:
        st.info("No articles found.")

    for article in filtered_articles:

        with st.container(border=True):

            st.markdown(
                f"**{article['topic']}**"
            )

            status_text = article[
                "status"
            ].replace("_", " ").title()

            st.caption(
                f"{article['content_type'].title()}"
                f" · {status_text}"
            )

            st.caption(
                f"Created: {article['created_at'][:10]}"
            )

            if st.button(
                "📂 Open",
                key=f"open_{article['thread_id']}",
                use_container_width=True
            ):

                if restore_article(
                    article["thread_id"]
                ):
                    st.rerun()


# ============================================================
# 6. MAIN HEADER
# ============================================================

st.title("🤖 Multi-Agent Content Studio")

st.write(
    "Create research-backed blogs and newsletters "
    "using collaborative AI agents."
)

st.caption(
    "Powered by Gemini · Tavily · LangGraph · SQLite"
)

st.divider()


# ============================================================
# 7. NEW CONTENT FORM
# ============================================================

if st.session_state.stage == "new":

    st.subheader("✨ Create New Content")

    with st.form("content_form"):

        topic = st.text_input(
            "Article Topic",
            placeholder=(
                "e.g. Agentic AI in software development"
            )
        )

        content_type = st.selectbox(
            "Content Type",
            ["blog", "newsletter"]
        )

        st.caption(
            "The Researcher will gather sources, "
            "the Writer will create a draft, "
            "the Fact-Checker will verify claims, "
            "and the Editor will refine the article."
        )

        submitted = st.form_submit_button(
            "🚀 Generate Content",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not topic.strip():
            st.warning(
                "Please enter an article topic."
            )

        else:
            thread_id = str(uuid.uuid4())

            st.session_state.thread_id = thread_id
            st.session_state.current_topic = topic.strip()
            st.session_state.completed_agents = []
            st.session_state.result = {}

            create_article(
                thread_id=thread_id,
                topic=topic.strip(),
                content_type=content_type
            )

            initial_state = {
                "topic": topic.strip(),
                "content_type": content_type,
                "revisions": 0,
                "approved": False
            }

            if execute_workflow(initial_state):
                st.rerun()


# ============================================================
# 8. ARTICLE WORKSPACE
# ============================================================

if st.session_state.stage in (
    "review",
    "completed"
):

    state = st.session_state.result

    st.subheader(
        st.session_state.current_topic
    )

    if st.session_state.stage == "review":
        st.warning(
            "⏳ Awaiting human approval"
        )
    else:
        st.success(
            "✅ Approved and saved"
        )

    st.divider()

    # Agent execution status
    st.subheader("🤖 Agent Execution")

    agent_columns = st.columns(4)

    agent_details = [
        ("researcher", "🔍 Researcher"),
        ("writer", "✍️ Writer"),
        ("fact_checker", "🛡️ Fact-Checker"),
        ("editor", "📝 Editor")
    ]

    for column, (name, label) in zip(
        agent_columns,
        agent_details
    ):

        with column:

            if name in (
                st.session_state.completed_agents
            ):
                st.success(f"✅ {label}")

            else:
                st.info(f"⏳ {label}")

    st.divider()

    # Article and evidence tabs
    article_tab, research_tab, fact_tab = st.tabs(
        [
            "📄 Article Preview",
            "🔍 Research",
            "🛡️ Fact-Check Report"
        ]
    )

    with article_tab:

        article_text = get_article_text(state)

        if article_text:
            st.markdown(article_text)
        else:
            st.info("No article is available yet.")

    with research_tab:

        st.markdown(
            state.get(
                "research",
                "No research available."
            )
        )

    with fact_tab:

        verified = state.get(
            "verified",
            False
        )

        if verified:
            st.success(
                "Automated verification passed."
            )
        else:
            st.warning(
                "Automated verification did not pass. "
                "Review the claims and evidence."
            )

        st.markdown(
            state.get(
                "fact_check",
                "No verification report available."
            )
        )

        st.caption(
            "Automated fact-checking can make mistakes. "
            "Human verification remains important."
        )

    st.divider()


# ============================================================
# 9. HUMAN APPROVAL
# ============================================================

if st.session_state.stage == "review":

    st.subheader("👤 Human Review")

    st.info(
        "Review the article, research sources, "
        "and verification report before approving."
    )

    state = st.session_state.result

    if not state.get("verified", False):
        st.warning(
            "The Fact-Checker did not verify all "
            "important claims. Approval should only "
            "follow manual verification."
        )

    feedback = st.text_area(
        "Revision Instructions",
        placeholder=(
            "For example: Make the article simpler, "
            "include examples, and improve the conclusion."
        ),
        height=120
    )

    approve_col, revise_col = st.columns(2)

    with approve_col:
        approve_clicked = st.button(
            "✅ Approve Article",
            type="primary",
            use_container_width=True
        )

    with revise_col:
        revise_clicked = st.button(
            "🔄 Request Changes",
            use_container_width=True
        )

    if approve_clicked:

        decision = {
            "approved": True,
            "feedback": ""
        }

        if execute_workflow(
            Command(resume=decision)
        ):
            st.rerun()

    if revise_clicked:

        if not feedback.strip():
            st.warning(
                "Please describe the changes "
                "you want the Writer to make."
            )

        else:
            decision = {
                "approved": False,
                "feedback": feedback.strip()
            }

            if execute_workflow(
                Command(resume=decision)
            ):
                st.rerun()


# ============================================================
# 10. APPROVED ARTICLE DOWNLOAD
# ============================================================

if st.session_state.stage == "completed":

    st.subheader("🎉 Article Completed")

    st.success(
        "Your article was approved and saved."
    )

    state = st.session_state.result

    output_path = state.get("output_path")

    if output_path:

        file_path = Path(output_path)

        if file_path.is_file():

            article_content = file_path.read_text(
                encoding="utf-8"
            )

            st.download_button(
                label="⬇️ Download Markdown Article",
                data=article_content,
                file_name=file_path.name,
                mime="text/markdown",
                use_container_width=True
            )

        else:
            # Fallback if local output file is missing
            st.warning(
                "The original output file was not found. "
                "You can download the saved article "
                "from the workflow checkpoint."
            )

            article_content = get_article_text(state)

            if article_content:

                st.download_button(
                    label="⬇️ Download Recovered Article",
                    data=article_content,
                    file_name="article.md",
                    mime="text/markdown",
                    use_container_width=True
                )


# ============================================================
# 11. WORKSPACE NAVIGATION
# ============================================================

if st.session_state.stage in (
    "review",
    "completed"
):

    st.divider()

    if st.button(
        "➕ Create Another Article",
        use_container_width=True
    ):

        reset_workspace()
        st.rerun()


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.caption(
    "Multi-Agent Content Pipeline | "
    "Gemini + Tavily + LangGraph + Streamlit"
)
