from src.agent_graph import build_agent_graph


def main():
    agent = build_agent_graph()

    questions = [
        #"What is the central limit theorem?",
        #"How does my textbook explain Poissonization?",
        #"Give me a practice problem about expectation from the course exercises",
        #"Summarize chapter 6",
        #"Summarize chapter 6.1",
        #"Summarize the standard deviation chapter from my textbook",
        #"Explain maximum likelihood and then give me a practice problem",

        #"Hi, how are you?",
        #"Explain backpropagation",

        #"What is the latest stable verison of LangGraph",
        # "What is conformal prediction?",
        # "What are some recent applications of conformal prediction?",
        "Has conformal prediction become more widely used in machine learning over the last two years?",
        "Are there any recent concerns about calibration methods for large language models?",
        "What are researchers currently using probability theory for in generative AI?",

        # "How is probability used in modern machine learning?",
        # "Is the central limit theorem still important?",
        # "Why is the central limit theorem so important?",
    ]

    DEBUG_TOOL_RESULTS = False

    for question in questions:
        result = agent.invoke({
            "messages": [
                ("user", question)
            ]
        })

        print("\n" + "=" * 80)
        print("QUESTION:")
        print(question)

        for message in result["messages"]:
            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    print("\nAGENT TOOL CALL:")
                    print("Tool:", tool_call["name"])
                    print("Args:", tool_call["args"])

            elif message.type == "tool" and DEBUG_TOOL_RESULTS:
                print("\nTOOL RESULT:")
                print(message.content)

            elif message.type == "ai" and message.content:
                print("\nFINAL ANSWER:")
                print(message.content)


if __name__ == "__main__":
    main()