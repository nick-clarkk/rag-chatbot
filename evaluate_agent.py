from pathlib import Path
import re
import math

from src.agent_graph import build_agent_graph
from tests.eval_cases import eval_cases



# OUTPUT SETTINGS

# Keep routine evaluation output from clogging the terminal.
SHOW_PASS_DETAILS = False

# If True, show a short piece of the final answer when a case fails.
SHOW_FAILURE_ANSWER = False

# Prevent failure answers from clogging the terminal.
FAILURE_ANSWER_MAX_CHARS = 500



# RESULT EXTRACTION

def extract_tool_calls(messages):
    """
    Return every tool call made during an agent run.

    Example:
    [
        {
            "name": "search_course_materials",
            "args": {
                "query": "...",
                "content_type": "theory",
                "chapter": 7,
            },
        },
        {
            "name": "run_python",
            "args": {"code": "..."},
        },
    ]
    """
    tool_calls = []

    for message in messages:
        calls = getattr(message, "tool_calls", None)

        if not calls:
            continue

        for call in calls:
            tool_calls.append(
                {
                    "name": call["name"],
                    "args": call.get("args", {}),
                }
            )

    return tool_calls


def extract_final_answer(messages):
    """
    Return the final nonempty AI response after all tool calls.
    """
    for message in reversed(messages):
        if (
            getattr(message, "type", None) == "ai"
            and getattr(message, "content", None)
            and not getattr(message, "tool_calls", None)
        ):
            return message.content

    return ""


def extract_plot_paths(messages):
    """
    Find plot paths returned by tools or included in the final answer.
    """
    plot_pattern = re.compile(
        r"artifacts/python_plot_[A-Za-z0-9_-]+\.png"
    )

    paths = []

    for message in messages:
        content = getattr(message, "content", "")

        if not isinstance(content, str):
            continue

        for match in plot_pattern.findall(content):
            if match not in paths:
                paths.append(match)

    return paths



# INDIVIDUAL CHECKS

def check_expected_tools(case, tool_calls):
    """
    Required tools must appear at least once.

    Extra calls to a required tool are allowed. This avoids penalizing
    reasonable agent behavior such as performing two web searches.
    """
    expected = case.get("expected_tools", [])
    actual = [call["name"] for call in tool_calls]

    failures = []

    for tool in expected:
        if tool not in actual:
            failures.append(
                f"required tool '{tool}' was not called"
            )

    return failures


def check_unexpected_tools(case, tool_calls):
    """
    When expected_tools is empty, the case expects a direct answer.

    For cases with expected tools, additional tools are not automatically
    considered failures unless explicitly forbidden.
    """
    expected = case.get("expected_tools", [])
    actual = [call["name"] for call in tool_calls]

    failures = []

    if not expected and actual:
        failures.append(
            f"expected no tools, but called: {', '.join(actual)}"
        )

    return failures


def check_forbidden_tools(case, tool_calls):
    """
    Explicitly forbidden tools must never be called.
    """
    forbidden = case.get("forbidden_tools", [])
    actual = [call["name"] for call in tool_calls]

    failures = []

    for tool in forbidden:
        if tool in actual:
            failures.append(
                f"forbidden tool '{tool}' was called"
            )

    return failures


def check_expected_tool_args(case, tool_calls):
    """
    Check only the arguments explicitly specified by the eval case.

    Other arguments are allowed.

    Example:
        expected:
            chapter=7
            content_type="theory"

        actual:
            query="Chapter 7 summary"
            chapter=7
            content_type="theory"

    passes because query was not part of the expectation.
    """
    expectations = case.get("expected_tool_args", {})
    failures = []

    for tool_name, expected_args in expectations.items():
        matching_calls = [
            call for call in tool_calls
            if call["name"] == tool_name
        ]

        if not matching_calls:
            # Missing tool is already reported by expected-tools check.
            continue

        match_found = False

        for call in matching_calls:
            actual_args = call["args"]

            if all(
                actual_args.get(key) == expected_value
                for key, expected_value in expected_args.items()
            ):
                match_found = True
                break

        if not match_found:
            actual_arg_sets = [
                call["args"] for call in matching_calls
            ]

            failures.append(
                f"{tool_name} did not receive expected args "
                f"{expected_args}; got {actual_arg_sets}"
            )

    return failures


def check_forbidden_tool_args(case, tool_calls):
    """
    Optional support for cases where certain arguments should not be supplied.

    Example:
        "forbidden_tool_args": {
            "search_course_materials": ["chapter", "subchapter"]
        }

    None-valued arguments are treated as unset.
    """
    expectations = case.get("forbidden_tool_args", {})
    failures = []

    for tool_name, forbidden_args in expectations.items():
        matching_calls = [
            call for call in tool_calls
            if call["name"] == tool_name
        ]

        for call in matching_calls:
            actual_args = call["args"]

            for arg_name in forbidden_args:
                if (
                    arg_name in actual_args
                    and actual_args[arg_name] is not None
                ):
                    failures.append(
                        f"{tool_name} should not set '{arg_name}', "
                        f"but got {actual_args[arg_name]!r}"
                    )

    return failures


def check_final_answer(final_answer):
    """
    Every completed evaluation case should produce a final response.
    """
    if not final_answer.strip():
        return ["no final answer was produced"]

    return []


def check_sources(case, final_answer):
    """
    For web-search cases that require sources, verify that the response
    contains a Sources section.

    This checks presence only, not source quality.
    """
    if not case.get("expect_sources", False):
        return []

    if "sources" not in final_answer.lower():
        return ["expected a Sources section but none was found"]

    return []


def check_plots(case, plot_paths):
    """
    Check plot count and make sure the returned plot files exist.
    """
    if not case.get("expect_plot", False):
        return []

    failures = []

    min_plots = case.get("min_plots", 1)

    if len(plot_paths) < min_plots:
        failures.append(
            f"expected at least {min_plots} plot(s), "
            f"found {len(plot_paths)}"
        )

    for plot_path in plot_paths:
        if not Path(plot_path).exists():
            failures.append(
                f"plot path does not exist: {plot_path}"
            )

    return failures


def check_numeric_answer(case, final_answer):
    """
    For cases with an expected numeric result, verify that the final answer
    contains a number within the allowed tolerance.

    This is intentionally simple: it looks for numeric values in the final
    answer and passes if any one is close enough to the expected value.
    """
    if "expected_numeric_answer" not in case:
        return []

    expected = case["expected_numeric_answer"]
    tolerance = case.get("numeric_tolerance", 1e-4)

    number_pattern = re.compile(
        r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    )

    candidates = []

    for match in number_pattern.findall(final_answer):
        try:
            candidates.append(float(match))
        except ValueError:
            pass

    if not candidates:
        return [
            f"expected numeric answer near {expected}, "
            "but no numeric value was found"
        ]

    for value in candidates:
        if math.isclose(
            value,
            expected,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return []

    return [
        f"expected numeric answer near {expected} "
        f"(tolerance {tolerance}), but no matching value was found"
    ]



# CASE EVALUATION

def evaluate_case(agent, case):
    """
    Run one eval case and return its structured result.
    """
    result = agent.invoke(
        {
            "messages": [
                ("user", case["question"])
            ]
        }
    )

    messages = result["messages"]

    tool_calls = extract_tool_calls(messages)
    final_answer = extract_final_answer(messages)
    plot_paths = extract_plot_paths(messages)

    failures = []

    failures.extend(
        check_expected_tools(case, tool_calls)
    )
    failures.extend(
        check_unexpected_tools(case, tool_calls)
    )
    failures.extend(
        check_forbidden_tools(case, tool_calls)
    )
    failures.extend(
        check_expected_tool_args(case, tool_calls)
    )
    failures.extend(
        check_forbidden_tool_args(case, tool_calls)
    )
    failures.extend(
        check_final_answer(final_answer)
    )
    failures.extend(
        check_numeric_answer(case, final_answer)
    )
    failures.extend(
        check_sources(case, final_answer)
    )
    failures.extend(
        check_plots(case, plot_paths)
    )

    return {
        "id": case["id"],
        "passed": len(failures) == 0,
        "failures": failures,
        "tool_calls": tool_calls,
        "plot_paths": plot_paths,
        "final_answer": final_answer,
    }



# COMPACT TERMINAL OUTPUT

def print_case_result(result):
    """
    One line for passes. Failure details only when something went wrong.
    """
    case_id = result["id"]

    if result["passed"]:
        print(f"PASS  {case_id}")

        if SHOW_PASS_DETAILS:
            tools = [
                call["name"]
                for call in result["tool_calls"]
            ]

            if tools:
                print(f"      tools: {' -> '.join(tools)}")

            if result["plot_paths"]:
                print(
                    f"      plots: {len(result['plot_paths'])}"
                )

        return

    print(f"FAIL  {case_id}")

    for failure in result["failures"]:
        print(f"      - {failure}")

    tools = [
        call["name"]
        for call in result["tool_calls"]
    ]

    if tools:
        print(f"      tools: {' -> '.join(tools)}")

    if SHOW_FAILURE_ANSWER and result["final_answer"]:
        answer = result["final_answer"].replace("\n", " ")
        answer = answer[:FAILURE_ANSWER_MAX_CHARS]

        print(f"      answer: {answer}")


def print_summary(results):
    total = len(results)
    passed = sum(result["passed"] for result in results)
    failed = total - passed

    percentage = (
        (passed / total) * 100
        if total
        else 0.0
    )

    print()
    print("=" * 52)
    print(
        f"RESULT: {passed}/{total} passed "
        f"({percentage:.1f}%)"
    )

    if failed:
        print(f"FAILED: {failed}")
    else:
        print("All deterministic checks passed.")

    print("=" * 52)



# MAIN

def main():
    agent = build_agent_graph()

    results = []

    print(f"Running {len(eval_cases)} agent eval cases...\n")

    for case in eval_cases:
        try:
            result = evaluate_case(agent, case)

        except Exception as exc:
            result = {
                "id": case["id"],
                "passed": False,
                "failures": [
                    f"evaluation raised "
                    f"{type(exc).__name__}: {exc}"
                ],
                "tool_calls": [],
                "plot_paths": [],
                "final_answer": "",
            }

        results.append(result)
        print_case_result(result)

    print_summary(results)


if __name__ == "__main__":
    main()