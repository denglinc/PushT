r"""
Renders the repository's docstring convention as reStructuredText for autodoc.

The modules write their docstrings as plain text:

    Summary line.
    Note: a remark, which may run over
    several lines.
    Args:
        - name: description, which may continue
          on deeper-indented lines
        - first, second: one description for two arguments
    Returns:
        - name: description            (or "- description" for an unnamed value)

Formulas are LaTeX in the notation of docs/guide/notation.md: inline as $\alpha_t z + \beta_t \epsilon$ (on one
line), and a display formula as a block between two lines that hold only $$. A docstring with a backslash is a raw
string (r-prefixed).

This extension turns Args into :param: fields (so autodoc adds the types from the signature), Returns into a
:returns: field, Note into a note admonition, inline formulas into the math role, $$ blocks into math directives,
other indented free text into literal blocks, and escapes the characters that reST would read as markup. Nothing in
the source files changes.
"""

from __future__ import annotations

import re
from typing import Any

from sphinx.application import Sphinx

SECTIONS = {"Args:", "Returns:"}
ITEM = re.compile(r"^(\s*)- (.*)$")
# "name: description" or "first, second: description", with plain identifiers before the colon.
NAMED = re.compile(r"^([A-Za-z_]\w*(?:, [A-Za-z_]\w*)*): (.*)$")
# A word's trailing underscore (reST's reference syntax) and lone markup characters.
TRAILING_UNDERSCORE = re.compile(r"(?<=\w)_(?=[\s.,;:)\]]|$)")
LEADING_UNDERSCORE = re.compile(r"(?<![\w\\])_(?=\w)")
MARKUP = re.compile(r"([*`|])")
# An inline formula, $...$ on one line; a line holding only $$ opens or closes a display formula.
INLINE_MATH = re.compile(r"\$([^$\n]+)\$")
DISPLAY_MATH = "$$"


def escape_plain(text: str) -> str:
    """
    Escapes reST markup in plain text without formulas.
    """
    text = text.replace("\\", "\\\\")
    text = MARKUP.sub(r"\\\1", text)
    text = TRAILING_UNDERSCORE.sub(r"\\_", text)
    return LEADING_UNDERSCORE.sub(r"\\_", text)


def escape(text: str) -> str:
    """
    Escapes reST markup in one line of docstring text and turns its inline formulas into the math role.
    Note: a role that touches a word character is set apart by an escaped space, which reST drops.
    """
    out, end = [], 0
    for match in INLINE_MATH.finditer(text):
        before = text[end : match.start()]
        out.append(escape_plain(before))
        if before[-1:].isalnum():
            out.append("\\ ")
        out.append(f":math:`{match.group(1)}`")
        end = match.end()
        if text[end : end + 1].isalnum():
            out.append("\\ ")
    out.append(escape_plain(text[end:]))
    return "".join(out)


def display_math(lines: list[str], i: int, indent: int) -> tuple[list[str], int]:
    """
    Writes a display formula as a math directive.
    Args:
        - lines: the docstring text
        - i: the line that opens the formula ($$)
        - indent: the indentation of the directive
    Returns:
        - the directive's lines
        - the first line after the closing $$
    """
    body = []
    i += 1
    while i < len(lines) and lines[i].strip() != DISPLAY_MATH:
        body.append(lines[i])
        i += 1
    depth = min((indent_of(b) for b in body if b.strip()), default=0)
    pad = " " * indent
    directive = [pad + ".. math::", ""]
    directive += [pad + "   " + b[depth:] if b.strip() else "" for b in body]
    return [*directive, ""], i + 1


def indent_of(line: str) -> int:
    """
    Returns the number of leading spaces of a line.
    """
    return len(line) - len(line.lstrip(" "))


def read_items(lines: list[str], i: int) -> tuple[list[str], int]:
    """
    Reads the "- ..." items of an Args or Returns section, each joined with its continuation lines.
    Args:
        - lines: the docstring
        - i: the first line after the section header
    Returns:
        - the items' texts
        - the first line after the section
    """
    items: list[str] = []
    item_indent = 0
    while i < len(lines) and lines[i].strip():
        match = ITEM.match(lines[i])
        if match and (not items or indent_of(lines[i]) <= item_indent):
            item_indent = indent_of(lines[i])
            items.append(match.group(2).strip())
        elif items and indent_of(lines[i]) > item_indent:
            items[-1] += " " + lines[i].strip()
        else:
            break
        i += 1
    return items, i


def section(header: str, items: list[str]) -> list[str]:
    """
    Writes an Args or Returns section as reST fields.
    """
    if header == "Args:":
        out = []
        for item in items:
            match = NAMED.match(item)
            if not match:
                out.append(f":param: {escape(item)}")
                continue
            names, text = match.groups()
            out += [f":param {name}: {escape(text)}" for name in names.split(", ")]
        return out
    parts = []
    for item in items:
        match = NAMED.match(item)
        parts.append(
            f"**{match.group(1)}** -- {escape(match.group(2))}"
            if match
            else escape(item)
        )
    if len(parts) == 1:
        return [f":returns: {parts[0]}"]
    return [":returns:", *(f"    * {part}" for part in parts)]


def free_text(lines: list[str]) -> list[str]:
    """
    Writes free text: escapes it, separates lists from the paragraph before them, and turns blocks indented deeper
    than the paragraph before them (examples, formulas, tables) into literal blocks.
    """
    out: list[str] = []
    # The indentation of the last paragraph line, and of the current list's dashes (None outside a list).
    base: int | None = None
    bullet: int | None = None
    i = 0
    while i < len(lines):
        stripped, depth = lines[i].strip(), indent_of(lines[i])
        if not stripped:
            out.append("")
            i += 1
            continue
        if stripped == DISPLAY_MATH:
            # A display formula: inside a list item at the item's text, elsewhere at the paragraph's indentation.
            in_item = bullet is not None and depth > bullet
            indent = bullet + 2 if in_item else (base or 0)
            if out and out[-1].strip():
                out.append("")
            block, i = display_math(lines, i, indent)
            out += block
            if not in_item:
                bullet = None
            continue
        if bullet is not None and depth > bullet and not stripped.startswith("- "):
            # The continuation of a list item, aligned with the item's text.
            out.append(" " * (bullet + 2) + escape(stripped))
            i += 1
            continue
        if stripped.startswith("- "):
            if bullet is None and out and out[-1].strip():
                out.append("")
            bullet = depth
            out.append(" " * depth + "- " + escape(stripped[2:]))
            i += 1
            continue
        if bullet is not None:
            # A paragraph after a list ends it.
            bullet = None
            if out[-1].strip():
                out.append("")
        if base is not None and depth > base:
            # A literal block: everything indented at least this deep (examples, formulas, tables).
            block = []
            while i < len(lines) and (
                not lines[i].strip() or indent_of(lines[i]) >= depth
            ):
                block.append(lines[i][depth:])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out += ["", "::", "", *("    " + b if b.strip() else "" for b in block), ""]
            continue
        base = depth
        out.append(" " * depth + escape(stripped))
        i += 1
    return out


def convert(lines: list[str]) -> list[str]:
    """
    Converts one docstring, given as lines without its common indentation.
    """
    out: list[str] = []
    text: list[str] = []

    def flush() -> None:
        """
        Writes the pending free text.
        """
        if text:
            out.extend(free_text(text))
            text.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if indent_of(line) == 0 and stripped in SECTIONS:
            flush()
            items, i = read_items(lines, i + 1)
            if out and out[-1].strip() and not out[-1].startswith(":"):
                out.append("")
            out += section(stripped, items)
            continue
        if indent_of(line) == 0 and stripped.startswith("Note:"):
            flush()
            body = [stripped[len("Note:") :].strip()] if stripped != "Note:" else []
            i += 1
            while i < len(lines) and lines[i].strip():
                if indent_of(lines[i]) == 0 and (
                    lines[i].strip() in SECTIONS or lines[i].startswith("Note:")
                ):
                    break
                if indent_of(lines[i]) == 0 and lines[i].startswith(
                    "Args and Returns:"
                ):
                    break
                body.append(lines[i])
                i += 1
            if out and out[-1].strip():
                out.append("")
            out += [".. note::", ""]
            out += ["   " + b if b.strip() else "" for b in free_text(body)]
            out.append("")
            continue
        if text and indent_of(line) == 0 and out and out[-1].startswith(":"):
            out.append("")
        text.append(line)
        i += 1
    flush()
    return out


def process_docstring(
    app: Sphinx, what: str, name: str, obj: Any, options: Any, lines: list[str]
) -> None:
    """
    Rewrites a docstring in place (the autodoc-process-docstring event).
    """
    lines[:] = convert(lines)


def setup(app: Sphinx) -> dict[str, Any]:
    """
    Registers the converter.
    """
    app.connect("autodoc-process-docstring", process_docstring)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
