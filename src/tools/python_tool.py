"""
Local bounded Python execution tool.

Some commands, imports, and functions are blocked as lightweight safeguards.
This is NOT intended to be a secure, production-grade sandbox.

It is intended for local use in this personal learning project.
"""


import ast
import io
import contextlib

from uuid import uuid4

from pathlib import Path

import matplotlib
matplotlib.use("Agg") # render images without a GUI
import matplotlib.pyplot as plt

from langchain_core.tools import tool

MAX_PLOTS = 20

PLOT_DIR = Path("artifacts")


BLOCKED_NAMES = {
    "eval",
    "exec",
    "open",
    "input",
    "compile",
    "globals",
    "locals",
    "vars",
    "__import__",
}

BLOCKED_MODULES = {
    "os",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "pathlib",
}

BLOCKED_PLOT_METHODS = { # stop LLM from deciding file paths or closing figures, run_python should handle that
    "savefig",
    "close",
    "show", # we are not using a GUI so this will stop error messages from clogging output
}


def validate_code(code: str) -> ast.Module:
    tree = ast.parse(code)

    for node in ast.walk(tree):

        # Block dangerous function calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_NAMES:
                raise ValueError(f"Use of '{node.func.id}' is not allowed.")

        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_MODULES:
                    raise ValueError(f"Import '{alias.name}' is not allowed.")

        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BLOCKED_MODULES:
                raise ValueError(f"Import from '{node.module}' is not allowed.")

        # Block access to dunder attributes such as __globals__
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Access to '{node.attr}' is not allowed.")

        # Plot persistence is managed by this tool
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_PLOT_METHODS:
                raise ValueError(
                    f"Plot method '{node.func.attr}' is not allowed. "
                    "Leave Matplotlib figures open; the tool saves them automatically."
                )

    return tree


# For clearing plots at the start of a new session
def clear_plot_artifacts() -> None:
    if not PLOT_DIR.exists():
        return

    for path in PLOT_DIR.glob("python_plot_*.png"):
        path.unlink()


# For deleting old plots after a certain threshold
def prune_old_plots() -> None:
    if not PLOT_DIR.exists():
        return

    plot_files = sorted(
        PLOT_DIR.glob("python_plot_*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    for path in plot_files[MAX_PLOTS:]:
        path.unlink()


@tool
def run_python(code: str) -> str:
    """
    Execute Python code for numerical calculations, probability computations,
    simulations, small computational tasks, and small visualizations.

    For visualizations, create Matplotlib figures but leave them open.
    Do not save or close figures; the tool handles plot persistence automatically.
    Use raw strings for Matplotlib text containing LaTeX backslashes.

    Return printed output, generated plots, or an error message.
    """

    plt.close("all") # close all plots before running code to avoid plots from earlier questions impacting current ones

    output = io.StringIO()

    try:
        tree = validate_code(code)

        namespace = {}

        with contextlib.redirect_stdout(output):

            # If the final statement is a bare expression, evaluate it
            # separately so its value is returned like in a Python REPL/Jupyter.
            if tree.body and isinstance(tree.body[-1], ast.Expr):

                statements = ast.Module(
                    body=tree.body[:-1],
                    type_ignores=[],
                )

                ast.fix_missing_locations(statements)

                exec(
                    compile(statements, "<string>", "exec"),
                    namespace,
                )

                expression = ast.Expression(
                    body=tree.body[-1].value,
                )

                ast.fix_missing_locations(expression)

                value = eval(
                    compile(expression, "<string>", "eval"),
                    namespace,
                )

                if value is not None:
                    print(repr(value))

            else:
                exec(
                    compile(tree, "<string>", "exec"),
                    namespace,
                )

        figure_numbers = plt.get_fignums()
        plot_paths = []

        if figure_numbers:
            PLOT_DIR.mkdir(parents=True, exist_ok=True)

            run_id = uuid4().hex[:8]

            for i, fig_num in enumerate(figure_numbers, start=1):
                fig = plt.figure(fig_num)

                path = PLOT_DIR / f"python_plot_{run_id}_{i}.png"

                fig.savefig(
                    path,
                    bbox_inches="tight",
                )

                plot_paths.append(path.as_posix())

            plt.close("all")

            prune_old_plots()

        result = output.getvalue().strip()

        if plot_paths:
            plot_text = "\n".join(
                f"Plot {i}: {path}"
                for i, path in enumerate(plot_paths, start=1)
            )

            if result:
                return f"{result}\n\nGenerated plots:\n{plot_text}"

            return f"Generated plots:\n{plot_text}"

        if result:
            return result

        return "Python executed successfully but produced no printed output."

    except Exception as e:
        return f"Python error: {type(e).__name__}: {e}"