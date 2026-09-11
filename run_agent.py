from src.agent_graph import build_agent_graph
from src.tools.python_tool import clear_plot_artifacts


TURN_WIDTH = 70


def print_user_prompt():
    print("\n" + "═" * TURN_WIDTH)
    print("YOU")
    print("─" * TURN_WIDTH)


def print_terra_turn(text):
    print("\nTERRA")
    print("─" * TURN_WIDTH)
    print(text)
    print()


def main():
    clear_plot_artifacts()

    agent = build_agent_graph()

    messages = []

    print("Probability Course Agent")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        try:
            print_user_prompt()
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            turn_messages = [
                *messages,
                ("user", user_input),
            ]

            result = agent.invoke(
                {
                    "messages": turn_messages
                }
            )

            result_messages = result["messages"]

            # Only commit the new turn to conversation history
            # after the agent finishes successfully.
            messages = result_messages

            final_answer = ""

            for message in reversed(result_messages):
                if (
                    getattr(message, "type", None) == "ai"
                    and getattr(message, "content", None)
                    and not getattr(message, "tool_calls", None)
                ):
                    final_answer = message.content
                    break

            if final_answer:
                print_terra_turn(final_answer)
            else:
                print("\nTerra did not produce a final answer.\n")

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        except Exception as exc:
            print(
                f"\nAn error occurred: "
                f"{type(exc).__name__}: {exc}"
            )
            print("The conversation is still active. You can try again.\n")


if __name__ == "__main__":
    main()