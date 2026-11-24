"""Convert a local documentation HTML snapshot into structured JSON."""

import argparse
from collections import Counter
from collections.abc import Iterator
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit

import bs4
from bs4 import BeautifulSoup, NavigableString, Tag


HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")
PARSER = "html.parser"
CONTAINERS = {
    "article", "aside", "blockquote", "dd", "div", "dl", "dt",
    "figure", "figcaption", "header", "hgroup", "main", "p", "section",
}
BLOCK_TAGS = CONTAINERS.union(HEADINGS, ("pre", "ul", "ol", "table", "details"))


def _prose(text: str) -> str:
    return " ".join(text.split())


def _table(table: Tag) -> dict:
    caption = table.find("caption", recursive=False)
    rows = []
    for row in table.select(":scope > tr, :scope > :is(thead, tbody, tfoot) > tr"):
        cells = []
        for cell in row.find_all(["th", "td"], recursive=False):
            cells.append({
                "header": cell.name == "th",
                "blocks": list(_blocks(cell)),
                **{attr: cell[attr] for attr in ("rowspan", "colspan", "scope") if attr in cell.attrs},
            })
        rows.append({"group": row.parent.name if row.parent is not table else None, "cells": cells})
    return {"type": "table", "caption": _prose(caption.get_text()) if caption else None, "rows": rows}


def _block_elements(container: Tag) -> Iterator[str | Tag]:
    """Group inline text between direct block elements."""
    inline = []
    for node in container.children:
        if type(node) is NavigableString:
            inline.append(str(node))
        elif isinstance(node, Tag) and node.name in BLOCK_TAGS:
            text = _prose("".join(inline))
            if text:
                yield text
            inline.clear()
            yield node
        elif isinstance(node, Tag):
            inline.append(node.get_text())

    text = _prose("".join(inline))
    if text:
        yield text


def _content(container: Tag) -> Iterator[dict | Tag]:
    """Yield blocks, headings, and closing section markers in DOM order."""
    for element in _block_elements(container):
        if isinstance(element, str):
            yield {"type": "paragraph", "text": element}
        elif element.name == "pre":
            code = element.get_text()
            if code.strip():
                yield {"type": "code", "text": code}
        elif element.name in ("ul", "ol"):
            items = [list(_blocks(li)) for li in element.find_all("li", recursive=False)]
            if items:
                yield {"type": "list", "ordered": element.name == "ol", "items": items}
        elif element.name == "table":
            table = _table(element)
            if table["rows"] or table["caption"]:
                yield table
        elif element.name == "details":
            summary = element.find("summary", recursive=False)
            summary_text = _prose(summary.extract().get_text()) if summary else None
            yield {
                "type": "details",
                "summary": summary_text,
                "anchor": element.get("id"),
                "blocks": list(_blocks(element)),
            }
        elif element.name in HEADINGS:
            yield element
        else:
            yield from _content(element)
            if element.name == "section":
                yield element


def _blocks(container: Tag) -> Iterator[dict]:
    for entry in _content(container):
        if isinstance(entry, dict):
            yield entry
        elif entry.name != "section" and (text := _prose(entry.get_text())):
            yield {"type": "paragraph", "text": text}


def _anchors(heading: Tag, section_id: str | None) -> list[str]:
    link = heading.select_one(".headerlink[href^='#']")
    anchors = [unquote(link["href"][1:])] if link else []
    anchors.extend((heading.get("id"), section_id))
    # Sphinx can place alternate targets immediately before the heading.
    for sibling in heading.previous_siblings:
        if isinstance(sibling, NavigableString) and not sibling.strip():
            continue
        if not isinstance(sibling, Tag) or sibling.name not in ("span", "a") or sibling.get_text(strip=True):
            break
        anchors.extend((sibling.get("id"), sibling.get("name")))
    return list(dict.fromkeys(anchor for anchor in anchors if anchor))


def _sections(content: Tag, source_url: str) -> list[dict]:
    sections = []
    stack = []
    occurrences = Counter()

    def add_section(heading, level, anchors):
        parent_id = stack[-1][0]["id"] if stack else None
        identity = ("anchor", anchors[0]) if anchors else ("heading", parent_id, level, heading)
        occurrences[identity] += 1
        key = json.dumps([source_url, identity, occurrences[identity]], ensure_ascii=False)
        section = {
            "id": sha256(key.encode("utf-8")).hexdigest(),
            "parent_id": parent_id,
            "heading": heading,
            "level": level,
            "heading_path": [parent["heading"] for parent, _ in stack] + ([heading] if heading else []),
            "anchors": anchors,
            "blocks": [],
        }
        sections.append(section)
        return section

    untitled = None
    for entry in _content(content):
        if isinstance(entry, dict):
            if stack:
                stack[-1][0]["blocks"].append(entry)
            else:
                if untitled is None:
                    untitled = add_section(None, None, [])
                untitled["blocks"].append(entry)
        elif entry.name == "section":
            while stack and stack[-1][1] is entry:
                stack.pop()
        elif heading := _prose(entry.get_text()):
            level = int(entry.name[1])
            owner = entry.find_parent("section")
            first_in_section = owner is not None and (not stack or stack[-1][1] is not owner)
            section_id = owner.get("id") if first_in_section else None
            while stack and stack[-1][0]["level"] >= level:
                # An explicit enclosing section stays active until its closing marker.
                active_owner = stack[-1][1]
                if active_owner is not None and active_owner is not owner:
                    break
                stack.pop()
            section = add_section(heading, level, _anchors(entry, section_id))
            stack.append((section, owner))
    return sections


def _read_snapshot(html_path: Path, expected_sha256: str | None) -> tuple[str, str]:
    raw = html_path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("Expected SHA-256 must contain 64 hexadecimal characters.")
        if digest != expected_sha256.lower():
            raise ValueError(f"SHA-256 mismatch for {html_path}: expected {expected_sha256}, got {digest}.")
    try:
        html = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{html_path} is not valid UTF-8 HTML.") from error
    if not html.strip():
        raise ValueError(f"HTML input is empty: {html_path}")
    if "\x00" in html:
        raise ValueError(f"HTML input contains a NUL character: {html_path}")
    return html, digest


def _source_url(soup: BeautifulSoup, source_url: str | None) -> str:
    if source_url is None:
        canonical = soup.select_one("link[rel~=canonical][href]")
        source_url = canonical["href"] if canonical else ""
    source_url = source_url.strip()
    url = urlsplit(source_url)
    if url.scheme not in ("http", "https") or not url.hostname or "#" in source_url or re.search(r"\s", source_url):
        raise ValueError("Provide an absolute HTTP(S) source URL without a fragment, or include one in the HTML.")
    return source_url


def _clean_content(content: Tag) -> None:
    for element in content.select("script, style, nav, footer"):
        if not element.decomposed:
            element.decompose()
    # Keep permalink targets available for section provenance.
    for link in content.select(".headerlink"):
        link.clear()
    for element in content.find_all("br"):
        element.replace_with("\n")


def ingest_document(
    html_path: Path, *, source_url: str | None = None, expected_sha256: str | None = None
) -> dict:
    """Read a UTF-8 HTML snapshot and return its structured representation."""
    html, digest = _read_snapshot(html_path, expected_sha256)
    # shortcut: recoverable HTML errors are accepted, add a strict validator if the corpus requires it.
    soup = BeautifulSoup(html, PARSER)
    main = soup.find("main")
    content = (main.find("article") or main) if main else soup.find("article")
    if content is None:
        raise ValueError(f"No main documentation content found in {html_path}.")
    _clean_content(content)
    resolved_url = _source_url(soup, source_url)
    sections = _sections(content, resolved_url)
    if not sections:
        raise ValueError(f"The extracted document is empty: {html_path}")

    title = _prose(soup.title.get_text()) if soup.title else None
    version = re.search(r"— scikit-learn (\S+) documentation$", title or "")
    return {
        "schema_version": 1,
        "source": {
            "filename": html_path.name,
            "url": resolved_url,
            "title": title,
            "documentation_version": version.group(1) if version else None,
            "sha256": digest,
        },
        "processing": {"encoding": "UTF-8", "parser": PARSER, "beautifulsoup_version": bs4.__version__},
        "sections": sections,
    }


def _write_json(document: dict, output_path: Path) -> None:
    payload = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=output_path.parent, delete=False) as output:
            temporary_path = Path(output.name)
            output.write(payload)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-url", help="Override the canonical URL in the HTML.")
    parser.add_argument("--expected-sha256", help="Reject input bytes that do not match this hash.")
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".json" or args.output.resolve() == args.html.resolve():
            raise ValueError("Output must be a separate .json file.")
        document = ingest_document(
            args.html, source_url=args.source_url, expected_sha256=args.expected_sha256
        )
        _write_json(document, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ingestion: {error}\n")


if __name__ == "__main__":
    main()
