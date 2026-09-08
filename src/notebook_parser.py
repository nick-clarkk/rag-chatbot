import json
from pathlib import Path
from langchain_core.documents import Document

def get_path_metadata(path):
    path = Path(path)

    chapter_folder = path.parents[1].name
    content_type = path.parent.name
    notebook_name = path.stem

    # Example: chapter_08_expectation
    chapter_parts = chapter_folder.split("_")

    chapter = int(chapter_parts[1])
    chapter_title = " ".join(chapter_parts[2:]).title()

    # Example: 01_Definition
    notebook_parts = notebook_name.split("_", 1)

    subchapter_number = int(notebook_parts[0])
    subchapter_title = notebook_parts[1].replace("_", " ").title()

    subchapter = f"{chapter}.{subchapter_number}"

    return {
        "chapter": chapter,
        "chapter_title": chapter_title,
        "subchapter": subchapter,
        "subchapter_title": subchapter_title,
        "content_type": content_type,
    }

def parse_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    extracted = []

    for i, cell in enumerate(notebook["cells"]):
        cell_type = cell["cell_type"]
        metadata = cell.get("metadata", {})
        tags = metadata.get("tags", [])

        # Skip cells explicitly marked by the textbook for removal
        if "remove-cell" in tags:
            continue

        source = "".join(cell.get("source", [])).strip()

        # Skip YAML-style frontmatter
        if cell_type == "markdown" and source.startswith("---") and source.endswith("---"):
            continue

        item = {
            "cell_index": i,
            "cell_type": cell_type,
            "source": source,
        }

        # Code cells may have useful displayed output, we won't parse the output in detail yet, but we will include it as a string for reference
        if cell_type == "code":
            outputs = []

            for output in cell.get("outputs", []):
                data = output.get("data", {})

                if "text/plain" in data:
                    text = "".join(data["text/plain"])
                    outputs.append(text)

                if "image/png" in data:
                    outputs.append("[IMAGE OUTPUT]")

            item["outputs"] = outputs

        extracted.append(item)

    return extracted

def create_documents(path):
    cells = parse_notebook(path)
    documents = []

    path_metadata = get_path_metadata(path)

    title = None
    current_section = None
    current_content = []

    for cell in cells:
        source = cell["source"]

        # Notebook title: starts the first section/document as "# Title"
        if cell["cell_type"] == "markdown" and source.startswith("# ") and not source.startswith("## "):
            # save previous section if it exists
            if current_section is not None and current_content:
                documents.append(
                    Document(
                        page_content="\n\n".join(current_content),
                        metadata={
                            "source": Path(path).name,
                            "section": current_section,
                            "title": title
                            **path_metadata,
                        },
                    )
                )
            # start a new section/document
            title = source.splitlines()[0][2:].strip()
            current_section = title
            current_content = [source]
            continue

        if cell["cell_type"] == "markdown" and source.startswith("## "):
            # save previous section if it exists
            if current_section is not None and current_content:
                documents.append(
                    Document(
                        page_content="\n\n".join(current_content),
                        metadata={
                            "source": Path(path).name,
                            "section": current_section,
                            "title": title,
                            **path_metadata,
                        },
                    )
                )

            # only keep the first line of the section as the section title
            current_section = source.splitlines()[0][3:].strip()
            current_content = [source]
            continue
        
        # general content, just cells between sections, add to current section
        if current_section is not None:
            current_content.append(source)
            # code outputs
            for output in cell.get("outputs", []):
                current_content.append(str(output))

    # save last section
    if current_section is not None and current_content:
        documents.append(
            Document(
                page_content="\n\n".join(current_content),
                metadata={
                    "source": Path(path).name,
                    "section": current_section,
                    "title": title,
                    **path_metadata,
                },
            )
        )

    return documents