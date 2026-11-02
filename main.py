import uuid

from langgraph.types import Command
from pipeline import workflow


def main():
    print("=" * 50)
    print("MULTI-AGENT CONTENT PIPELINE")
    print("=" * 50)

    topic = input("\nEnter article topic: ").strip()

    if not topic:
        print("Please provide a topic.")
        return

    content_type = input(
        "Content type (blog/newsletter): "
    ).strip() or "blog"

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        "recursion_limit": 100
    }

    initial_state = {
        "topic": topic,
        "content_type": content_type,
        "revisions": 0,
        "approved": False
    }

    result = workflow.invoke(initial_state, config)

    while result.get("__interrupt__"):
        state = workflow.get_state(config).values

        article = state.get(
            "final_article",
            state.get("draft", "")
        )

        print("\n" + "=" * 50)
        print("ARTICLE READY FOR REVIEW")
        print("=" * 50)

        print("\n", article)

        print("\nFACT-CHECK REPORT:")
        print(state.get("fact_check", ""))

        print("\nVerification passed:",
              state.get("verified", False))

        choice = input(
            "\nApprove article? (yes/no): "
        ).strip().lower()

        approved = choice in ("yes", "y")

        feedback = ""

        if not approved:
            feedback = input(
                "What changes are needed? "
            ).strip()

            if not feedback:
                feedback = "Improve the article and clarity."

        result = workflow.invoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback
                }
            ),
            config
        )

    print("\nWorkflow completed.")

    if result.get("output_path"):
        print("Saved to:", result["output_path"])


if __name__ == "__main__":
    main()