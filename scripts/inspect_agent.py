from src.agent_graph import build_agent_graph


def main():
    agent = build_agent_graph()

    general_questions = [
    "What is the central limit theorem?",
    "Hi, how are you?",
    "Explain backpropagation",
    # "What is conformal prediction?",
    "How is probability used in modern machine learning?",
    # "Is the central limit theorem still important?",
    "Why is the central limit theorem so important?",
    ]


    course_questions = [
        "How does my textbook explain Poissonization?",
        "Give me a practice problem about expectation from the course exercises",
        "Summarize chapter 7",
        "Summarize chapter 7.1",
        "Summarize the standard deviation chapter from my textbook",
        "Explain maximum likelihood and then give me a practice problem",
    ]


    web_questions = [
        "What is the latest stable version of LangGraph?",
        "What are some recent applications of conformal prediction?",
        "Has conformal prediction become more widely used in machine learning over the last two years?",
        "Are there any recent concerns about calibration methods for large language models?",
        "What are researchers currently using probability theory for in generative AI?",
    ]


    python_questions = [
        "What is 37 * 84?",
        "What is P(X >= 7) for X ~ Binomial(10, 0.6)?",
        "What is P(X >= 37) for X ~ Binomial(100, 0.31)?",
        "Compute the probability that a Poisson random variable with mean 18.7 is between 12 and 25 inclusive.",
        "Simulate 10000 rolls of two fair six-sided dice and estimate the probability that their sum is at least 10.",
        "Use simulation to estimate P(X >= 7) for X ~ Binomial(10, 0.6), and compare the simulation estimate to the exact probability.",
        "Compute the mean and variance of [3, 5, 7, 9, 11].",
        "Using Python, calculate P(X <= 4) for X ~ Binomial(12, 0.3).",
        "Use Python to compute the first 10 Fibonacci numbers.",
    ]

    plotting_questions = [
        "Plot the probability mass function of a Binomial(20, 0.3) random variable.",
        "Create two separate plots: one showing a Beta(2, 5) density and one showing a Gamma(2, 1) density.",
        "Plot a Binomial(20, 0.3) PMF, then create a separate plot of its cumulative distribution function.",
    ]

    mixed_questions = [
    "How does my textbook explain the binomial distribution, and plot a Binomial(20, 0.3) PMF?",
    "Explain conformal prediction and tell me whether there have been any important recent developments.",
    "Use my course materials to explain the central limit theorem, then simulate it.",
    "Find a practice problem about expectation from my course and use Python to verify the numerical answer after I solve it.",
    ]

    # Test for over-tooling, under-tooling, restraint in web searches
    boundary_questions = [ 
    "Explain the difference between a normal distribution and a gamma distribution, then plot an example of each.",
    "Use my textbook to explain the Gamma family from Chapter 18, then plot Gamma distributions with shape parameters 1, 2, and 5.",
    "Find me a matching-problem exercise from Chapter 5, but don't solve it or use Python yet.",
    "Use my course material on inclusion-exclusion to explain the method, then create a new practice problem that is not copied from the textbook.",
    "Use my textbook's treatment of sampling without replacement to explain the idea, then use Python to compute a small numerical example.",
    "Explain the chi-square distribution, then tell me about any important recent applications of it in machine learning.",
    "What are some recent uses of Gamma distributions in machine learning or generative modeling, and plot a few Gamma densities to illustrate how the shape changes?",
    "Give me a practice problem involving inclusion-exclusion.",
    ]

    questions = [
    (
        "Calculate the probability of getting exactly 4 aces when "
        "20 cards are drawn from a standard deck without replacement."
    ),
    ]

    # questions = boundary_questions

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