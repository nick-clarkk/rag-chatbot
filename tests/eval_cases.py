eval_cases = [

    # GENERAL KNOWLEDGE / NO TOOL
    {
        "id": "general_clt",
        "question": "What is the central limit theorem?",
        "expected_tools": [],
    },
    {
        "id": "general_backprop",
        "question": "Explain backpropagation.",
        "expected_tools": [],
    },
    {
        "id": "general_probability_ml",
        "question": "How is probability used in modern machine learning?",
        "expected_tools": [],
    },
    {
        "id": "general_simple_arithmetic",
        "question": "What is 37 * 84?",
        "expected_tools": [],
        "notes": (
            "Simple arithmetic should be answered directly rather than using Python."
        ),
    },


    # COURSE RETRIEVAL
    {
        "id": "course_poissonization",
        "question": "How does my textbook explain Poissonization?",
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "theory",
            }
        },
        "notes": (
            "The user explicitly asks how the textbook explains the topic, "
            "so course retrieval is required."
        ),
    },
    {
        "id": "course_chapter_7_summary",
        "question": "Summarize chapter 7.",
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "chapter": 7,
                "content_type": "theory",
            }
        },
        "notes": (
            "Explicit whole-chapter request should use structural retrieval "
            "with chapter=7."
        ),
    },
    {
        "id": "course_subchapter_7_1_summary",
        "question": "Summarize chapter 7.1.",
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "subchapter": "7.1",
                "content_type": "theory",
            }
        },
        "notes": (
            "Decimal course locations should be treated as subchapters, "
            "even if the user calls them a chapter."
        ),
    },
    {
        "id": "course_expectation_exercise",
        "question": (
            "Give me a practice problem about expectation "
            "from the course exercises."
        ),
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "exercises",
            }
        },
        "notes": (
            "Because the user explicitly asks for a course exercise, "
            "retrieve an existing exercise rather than generating a fresh one."
        ),
    },


    # FRESH VS COURSE-SOURCED PRACTICE
    {
        "id": "fresh_inclusion_exclusion_problem",
        "question": "Give me a practice problem involving inclusion-exclusion.",
        "expected_tools": [],
        "notes": (
            "A fresh generated practice problem is acceptable because the user "
            "did not ask for one from the course or textbook."
        ),
    },
    {
        "id": "course_matching_problem_restraint",
        "question": (
            "Find me a matching-problem exercise from Chapter 5, "
            "but don't solve it or use Python yet."
        ),
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "chapter": 5,
                "content_type": "exercises",
            }
        },
        "forbidden_tools": ["run_python"],
        "notes": (
            "Retrieve the requested course exercise, but respect the user's "
            "explicit instruction not to use Python or solve it."
        ),
    },


    # WEB
    {
        "id": "web_langgraph_version",
        "question": "What is the latest stable version of LangGraph?",
        "expected_tools": ["tavily_search"],
        "expect_sources": True,
        "notes": (
            "The answer depends on current software information, "
            "so web search is required."
        ),
    },
    {
        "id": "web_recent_conformal",
        "question": "What are some recent applications of conformal prediction?",
        "expected_tools": ["tavily_search"],
        "expect_sources": True,
        "notes": (
            "Recent applications require current external information rather "
            "than model knowledge alone."
        ),
    },
    {
        "id": "web_llm_calibration",
        "question": (
            "Are there any recent concerns about calibration methods "
            "for large language models?"
        ),
        "expected_tools": ["tavily_search"],
        "expect_sources": True,
        "notes": (
            "The question explicitly asks about recent research concerns, "
            "so web search should be used."
        ),
    },


    # PYTHON COMPUTATION
    {
        "id": "python_binomial_tail",
        "question": "What is P(X >= 37) for X ~ Binomial(100, 0.31)?",
        "expected_tools": ["run_python"],
        "notes": (
            "This is computationally involved enough that Python improves "
            "numerical reliability."
        ),
    },
    {
        "id": "python_simulation",
        "question": (
            "Simulate 10000 rolls of two fair six-sided dice and "
            "estimate the probability that their sum is at least 10."
        ),
        "expected_tools": ["run_python"],
        "notes": (
            "Simulation explicitly requires execution rather than direct "
            "model calculation."
        ),
    },
    {
        "id": "python_explicit_request",
        "question": "Using Python, calculate P(X <= 4) for X ~ Binomial(12, 0.3).",
        "expected_tools": ["run_python"],
        "notes": (
            "The user explicitly requests Python, so the tool should be used "
            "even though the calculation is relatively small."
        ),
    },


    # PLOTTING
    {
        "id": "plot_binomial_pmf",
        "question": (
            "Plot the probability mass function of a "
            "Binomial(20, 0.3) random variable."
        ),
        "expected_tools": ["run_python"],
        "expect_plot": True,
        "min_plots": 1,
        "notes": (
            "A visualization request should use Python and produce at least "
            "one saved plot artifact."
        ),
    },
    {
        "id": "plot_two_distributions",
        "question": (
            "Create two separate plots: one showing a Beta(2, 5) density "
            "and one showing a Gamma(2, 1) density."
        ),
        "expected_tools": ["run_python"],
        "expect_plot": True,
        "min_plots": 2,
        "notes": (
            "The user explicitly asks for two separate figures, so at least "
            "two plot artifacts should be created."
        ),
    },


    # MULTI-TOOL
    {
        "id": "mixed_gamma_course_plot",
        "question": (
            "Use my textbook to explain the Gamma family from Chapter 18, "
            "then plot Gamma distributions with shape parameters 1, 2, and 5."
        ),
        "expected_tools": [
            "search_course_materials",
            "run_python",
        ],
        "expected_tool_args": {
            "search_course_materials": {
                "chapter": 18,
                "content_type": "theory",
            }
        },
        "expect_plot": True,
        "min_plots": 1,
        "notes": (
            "This should combine structural course retrieval with Python "
            "visualization."
        ),
    },
    {
        "id": "mixed_sampling_python",
        "question": (
            "Use my textbook's treatment of sampling without replacement "
            "to explain the idea, then use Python to compute a small "
            "numerical example."
        ),
        "expected_tools": [
            "search_course_materials",
            "run_python",
        ],
        "notes": (
            "The explanation should be grounded in the textbook, while the "
            "numerical example should be executed with Python."
        ),
    },
    {
        "id": "mixed_inclusion_exclusion_fresh",
        "question": (
            "Use my course material on inclusion-exclusion to explain "
            "the method, then create a new practice problem that is not "
            "copied from the textbook."
        ),
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "theory",
            }
        },
        "notes": (
            "Retrieve theory for grounding, but generate a fresh practice "
            "problem rather than pulling an existing exercise."
        ),
    },
    {
        "id": "mixed_gamma_web_plot",
        "question": (
            "What are some recent uses of Gamma distributions in machine "
            "learning or generative modeling, and plot a few Gamma densities "
            "to illustrate how the shape changes?"
        ),
        "expected_tools": [
            "tavily_search",
            "run_python",
        ],
        "expect_sources": True,
        "expect_plot": True,
        "min_plots": 1,
        "notes": (
            "The recent-uses portion requires web search, while the density "
            "visualization requires Python."
        ),
    },

    # FRESH / HELD-OUT EVALUATION CASES
    {
        "id": "fresh_course_density_cdf",
        "question": (
            "Use my textbook to explain the relationship between a density "
            "function and a cumulative distribution function."
        ),
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "theory",
            }
        },
        "forbidden_tool_args": {
            "search_course_materials": ["chapter", "subchapter"],
        },
        "notes": (
            "The user asks about the textbook but does not provide a course "
            "location, so semantic theory retrieval should be used."
        ),
    },
    {
        "id": "fresh_course_density_subchapter",
        "question": "Summarize section 15.1 from my textbook.",
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "subchapter": "15.1",
                "content_type": "theory",
            }
        },
        "notes": (
            "An explicit decimal course location should route structurally "
            "through subchapter=15.1."
        ),
    },
    {
        "id": "fresh_course_convolution",
        "question": (
            "How does my textbook derive the distribution of the sum of "
            "two independent random variables?"
        ),
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "theory",
            }
        },
        "forbidden_tool_args": {
            "search_course_materials": ["chapter", "subchapter"],
        },
        "notes": (
            "This should use semantic textbook retrieval for convolution "
            "without inventing a chapter or subchapter."
        ),
    },
    {
        "id": "fresh_course_convolution_subchapter",
        "question": "Summarize section 19.1 from my textbook.",
        "expected_tools": ["search_course_materials"],
        "expected_tool_args": {
            "search_course_materials": {
                "subchapter": "19.1",
                "content_type": "theory",
            }
        },
        "notes": (
            "This checks structural routing to a previously unused subchapter."
        ),
    },


    # FRESH NON-COURSE CASES
    {
        "id": "fresh_web_recent_diffusion",
        "question": (
            "What are some important recent developments in diffusion model sampling?"
        ),
        "expected_tools": ["tavily_search"],
        "expect_sources": True,
        "notes": (
            "Recent research developments require current web search."
        ),
    },
    {
        "id": "fresh_python_hypergeometric",
        "question": (
            "Calculate the probability of getting exactly 4 aces when "
            "20 cards are drawn from a standard deck without replacement."
        ),
        "expected_tools": ["run_python"],
        "expected_numeric_answer": 0.0178963893,
        "numeric_tolerance": 1e-4,
        "notes": (
            "This is a nontrivial hypergeometric calculation. Python should be "
            "used to improve numerical reliability."
        ),
    },
    {
        "id": "fresh_python_birthday",
        "question": (
            "Assuming 365 equally likely birthdays and ignoring leap years, "
            "what is the probability that at least two people in a group of "
            "40 share a birthday?"
        ),
        "expected_tools": ["run_python"],
        "expected_numeric_answer": 0.8912318098,
        "numeric_tolerance": 1e-4,
        "notes": (
            "The complement involves a long product, so Python should be used "
            "for reliable numerical computation."
        ),
    },
    {
        "id": "fresh_python_binomial_tail",
        "question": (
            "If X ~ Binomial(50, 0.12), calculate P(X >= 10)."
        ),
        "expected_tools": ["run_python"],
        "expected_numeric_answer": 0.0707616000,
        "numeric_tolerance": 1e-4,
        "notes": (
            "This binomial tail requires summing several probabilities, so Python "
            "should be used for numerical reliability."
        ),
    },
    {
        "id": "fresh_plot_chi_square",
        "question": (
            "Plot chi-square densities with 1, 3, and 10 degrees of freedom "
            "on the same figure."
        ),
        "expected_tools": ["run_python"],
        "expect_plot": True,
        "min_plots": 1,
        "notes": (
            "This is a fresh visualization case and should be handled with Python."
        ),
    },
    {
        "id": "fresh_mixed_convolution_plot",
        "question": (
            "Use my textbook to explain why the sum of two independent "
            "Uniform(0, 1) random variables has a triangular density, "
            "then plot that density."
        ),
        "expected_tools": [
            "search_course_materials",
            "run_python",
        ],
        "expected_tool_args": {
            "search_course_materials": {
                "content_type": "theory",
            }
        },
        "expect_plot": True,
        "min_plots": 1,
        "notes": (
            "Course retrieval should ground the explanation, while Python "
            "should create the requested visualization."
        ),
    },
]