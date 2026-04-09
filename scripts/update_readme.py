"""
Scan for lab directories containing lab.json, and regenerate the labs table
in the root README.md between the LABS_TABLE_START / LABS_TABLE_END markers.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
LABS_CONFIG_PATH = REPO_ROOT / "docs" / "labs-config.json"
START_MARKER = "<!-- LABS_TABLE_START -->"
END_MARKER = "<!-- LABS_TABLE_END -->"

REQUIRED_FIELDS = ["name", "industry", "description"]
GITHUB_REPO = "Azure-Samples/AzureML_industry_labs"
GITHUB_BRANCH = "main"


def discover_labs() -> list[dict]:
    """Find all lab directories that contain a lab.json file."""
    labs = []
    for lab_json in sorted(REPO_ROOT.glob("*/lab.json")):
        lab_dir = lab_json.parent.name
        try:
            with open(lab_json, encoding="utf-8-sig") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: {lab_dir}/lab.json could not be read: {e}")
            continue

        missing = [field for field in REQUIRED_FIELDS if field not in metadata]
        if missing:
            print(f"WARNING: {lab_dir}/lab.json missing fields: {missing}")
            continue

        metadata["directory"] = lab_dir
        labs.append(metadata)

    return labs


def build_table(labs: list[dict]) -> str:
    """Build a Markdown table from lab metadata."""
    lines = [
        "| # | Lab | Industry | Description |",
        "|---|-----|----------|-------------|",
    ]
    for i, lab in enumerate(labs, start=1):
        name = lab["name"]
        industry = lab["industry"]
        description = lab["description"]
        link = lab.get("githubPath", f"{lab['directory']}/")
        lines.append(
            f"| {i} | [{name}]({link}) | {industry} | {description} |"
        )
    return "\n".join(lines)


def update_readme(table: str) -> bool:
    """Replace the labs table in README.md. Returns True if content changed."""
    readme_text = README_PATH.read_text(encoding="utf-8")

    start_idx = readme_text.find(START_MARKER)
    end_idx = readme_text.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print("ERROR: Could not find LABS_TABLE markers in README.md")
        sys.exit(1)

    before = readme_text[: start_idx + len(START_MARKER)]
    after = readme_text[end_idx:]

    new_readme = f"{before}\n{table}\n{after}"

    if new_readme == readme_text:
        print("README.md is already up to date.")
        return False

    README_PATH.write_text(new_readme, encoding="utf-8")
    print("README.md updated with new labs table.")
    return True


def build_pages_config(labs: list[dict]) -> list[dict]:
    """Transform lab metadata into the schema expected by the GitHub Pages site."""
    config = []
    for lab in labs:
        default_path = f"https://github.com/{GITHUB_REPO}/tree/{GITHUB_BRANCH}/{lab['directory']}"
        github_path = lab.get("githubPath", default_path)
        entry = {
            "name": lab["name"],
            "industry": lab["industry"],
            "shortDescription": lab["description"],
            "detailedDescription": lab.get("detailedDescription", lab["description"]),
            "language": lab.get("language", []),
            "useCase": lab.get("useCase", []),
            "authors": lab.get("authors", []),
            "directory": lab["directory"],
            "githubPath": github_path,
            "external": github_path != default_path,
        }
        config.append(entry)
    return config


def update_labs_config(labs: list[dict]) -> bool:
    """Write labs-config.json for the GitHub Pages site. Returns True if changed."""
    pages_config = build_pages_config(labs)
    new_content = json.dumps(pages_config, indent=2, ensure_ascii=False) + "\n"

    if LABS_CONFIG_PATH.exists():
        existing = LABS_CONFIG_PATH.read_text(encoding="utf-8")
        if existing == new_content:
            print("docs/labs-config.json is already up to date.")
            return False

    LABS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABS_CONFIG_PATH.write_text(new_content, encoding="utf-8")
    print("docs/labs-config.json updated.")
    return True


def main():
    labs = discover_labs()
    if not labs:
        print("No labs found (no lab.json files detected).")
        sys.exit(0)

    print(f"Found {len(labs)} lab(s): {', '.join(lab['directory'] for lab in labs)}")
    table = build_table(labs)
    update_readme(table)
    update_labs_config(labs)


if __name__ == "__main__":
    main()
