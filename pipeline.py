
import os
from pathlib import Path
from typing import TypedDict

import time
import random
from google.genai.errors import ServerError

from click import prompt
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tavily import TavilyClient

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
print("Selected Gemini model:", repr(MODEL))

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================================
# 2. SHARED STATE
# ==========================================

class ContentState(TypedDict, total=False):
    topic: str
    content_type: str
    research: str
    draft: str
    fact_check: str
    verified: bool
    revisions: int
    final_article: str
    approved: bool
    human_feedback: str
    output_path: str


# ==========================================
# 3. GEMINI HELPER
# ==========================================


def ask_gemini(prompt, use_search=False):

    max_attempts = 2

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"Gemini request attempt {attempt}...")

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            if not response.text:
                raise ValueError(
                    "Gemini returned an empty response."
                )

            return response.text

        except ServerError as error:

            if error.code != 503:
                raise

            if attempt == max_attempts:
                print(
                    "Gemini is still unavailable. "
                    "Please try again later."
                )
                raise

            wait_time = 15 + random.uniform(0, 5)

            print(
                f"Gemini server busy. "
                f"Waiting {wait_time:.1f} seconds..."
            )

            time.sleep(wait_time)


def search_web(query):

    print(f"Searching web: {query}")

    response = tavily.search(
        query=query,
        search_depth="basic",
        max_results=5
    )

    results = response.get("results", [])

    if not results:
        raise RuntimeError(
            "No search results returned. Cannot verify sources."
        )

    research = []

    for result in results:
        research.append(
            f"""
            Title: {result.get('title', '')}
            URL: {result.get('url', '')}
            Content: {result.get('content', '')}
            """
        )

    return "\n\n".join(research)


# ==========================================
# 4. RESEARCHER AGENT
# ==========================================


def researcher_agent(state: ContentState):

    print("\n[1] Researcher Agent working...")

    search_results = search_web(state["topic"])

    prompt = f"""
    You are an expert research assistant.

    Topic:
    {state['topic']}

    Web search results:
    {search_results}

    Your tasks:

    1. Summarize the key findings.
    2. Identify important facts.
    3. Organize the research.
    4. Include relevant source URLs.
    5. Highlight uncertainty.

    Only use information supported by the
    supplied search results.

    Do not invent citations.
    """

    research = ask_gemini(prompt)

    print("Research completed.")

    return {
        "research": research,
    }



# ==========================================
# 5. WRITER AGENT
# ==========================================

def writer_agent(state: ContentState):

    print("\n[2] Writer Agent working...")

    prompt = f"""
    You are a professional content writer.

    Topic:
    {state['topic']}

    Content type:
    {state['content_type']}

    Research:
    {state['research']}

    Previous draft:
    {state.get('draft', '')}

    Fact-check feedback:
    {state.get('fact_check', '')}

    Human feedback:
    {state.get('human_feedback', '')}

    Write an engaging article of approximately
    800 words.

    Requirements:

    - Attractive title
    - Clear introduction
    - Organized headings
    - Informative paragraphs
    - Strong conclusion
    - Sources for factual claims
    - Markdown formatting

    Address previous feedback when revising.
    Do not invent facts.
    """

    draft = ask_gemini(prompt)

    print("Draft generated.")

    return {
        "draft": draft,
        "final_article": ""
    }


# ==========================================
# 6. FACT-CHECKER AGENT
# ==========================================


def fact_checker_agent(state: ContentState):

    print("\n[3] Fact-Checker Agent working...")

    # Retrieve evidence independently.
    search_results = search_web(state["topic"])

    prompt = f"""
    You are a strict fact-checker.

    Article:
    {state['draft']}

    Research:
    {state['research']}

    Additional web evidence:
    {search_results}

    Verify important factual claims.

    Check:

    1. Statistics
    2. Dates and names
    3. Factual accuracy
    4. Source reliability
    5. Whether evidence supports the claims

    Use only the supplied evidence.

    Begin your answer with exactly:

    VERDICT: PASS

    OR

    VERDICT: FAIL

    Then explain the findings.

    If an important claim lacks evidence,
    return FAIL.

    Never invent sources.
    """

    report = ask_gemini(prompt)

    first_line = report.strip().splitlines()[0].strip()

    verified = first_line.upper() == "VERDICT: PASS"

    print("Fact-check completed.")
    print("Verified:", verified)

    return {
        "fact_check": report,
        "verified": verified,
        "revisions": state.get("revisions", 0) + 1
    }


# ==========================================
# 7. EDITOR AGENT
# ==========================================

def editor_agent(state: ContentState):

    print("\n[4] Editor Agent working...")

    prompt = f"""
    You are a senior content editor.

    Article:
    {state['draft']}

    Fact-check report:
    {state['fact_check']}

    Improve:

    - Grammar
    - Readability
    - Structure
    - Headings
    - Formatting
    - Professional tone

    Preserve verified facts and citations.
    Do not introduce new factual claims.

    Return the final Markdown article.
    """

    final_article = ask_gemini(prompt)

    print("Editing completed.")

    return {"final_article": final_article}


# ==========================================
# 8. FACT-CHECK ROUTING
# ==========================================

def route_fact_check(state: ContentState):

    if state["verified"]:
        return "editor"

    if state["revisions"] >= 3:
        return "human_review"

    return "writer"


# ==========================================
# 9. HUMAN APPROVAL
# ==========================================

def human_review(state: ContentState):

    decision = interrupt({
        "article": (
            state.get("final_article")
            or state["draft"]
        ),
        "fact_check": state["fact_check"],
        "verified": state["verified"],
        "message": "Review and approve this article."
    })

    approved = decision.get("approved") is True

    return Command(
        update={
            "approved": approved,
            "human_feedback": decision.get("feedback", "")
        },
        goto="save_article" if approved else "writer"
    )


# ==========================================
# 10. SAVE ARTICLE
# ==========================================

def save_article(state: ContentState):

    if not state.get("approved"):
        raise ValueError("Human approval required.")

    Path("output").mkdir(exist_ok=True)

    filename = f"article-{os.urandom(4).hex()}.md"
    filepath = Path("output") / filename

    article = (
        state.get("final_article")
        or state["draft"]
    )

    filepath.write_text(
        article,
        encoding="utf-8"
    )

    print("\nArticle saved successfully!")

    return {"output_path": str(filepath)}


# ==========================================
# 11. LANGGRAPH WORKFLOW
# ==========================================

builder = StateGraph(ContentState)

builder.add_node("researcher", researcher_agent)
builder.add_node("writer", writer_agent)
builder.add_node("fact_checker", fact_checker_agent)
builder.add_node("editor", editor_agent)
builder.add_node("human_review", human_review)
builder.add_node("save_article", save_article)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "fact_checker")

builder.add_conditional_edges(
    "fact_checker",
    route_fact_check,
    {
        "writer": "writer",
        "editor": "editor",
        "human_review": "human_review"
    }
)

builder.add_edge("editor", "human_review")
builder.add_edge("save_article", END)

memory = InMemorySaver()

workflow = builder.compile(
    checkpointer=memory
)
