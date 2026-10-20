"""
Sphinx configuration of the PushT documentation.

    uv run --only-group docs sphinx-build -W --keep-going -b html docs docs/_build/html      # what CI runs
    uv run --only-group docs sphinx-autobuild docs docs/_build/html --watch data.py --watch flow.py \
        --watch model.py --watch train.py --watch evaluation.py --watch logger.py --watch conf  # live preview

The guide (docs/guide) is written by hand; the API pages (docs/api) are generated from the docstrings. Every
cross-reference must resolve (nitpicky, and -W in CI), so renaming a function without updating the guide fails the
build.
"""

from __future__ import annotations

import sys
from pathlib import Path

DOCS = Path(__file__).parent
# The modules live at the repository root; the local extension next to this file.
sys.path.insert(0, str(DOCS.parent))
sys.path.insert(0, str(DOCS / "_ext"))

project = "PushT"
author = "Denglin Cheng"
copyright = "2026, denglinc"
html_title = "PushT"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "docstrings",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".md": "markdown"}

# MyST: $...$ and $$...$$ math, ::: fences for directives, anchors for ## and ### headings.
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
myst_heading_anchors = 3

# Theme: Furo, as rsl_rl and mjlab.
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/denglinc/PushT/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/denglinc/PushT",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17'
                ".55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52"
                "-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64"
                "-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
                "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 "
                "3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 "
                '8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# Autodoc, as rsl_rl: source order, types in the field lists, the heavy packages mocked so that a docs build needs
# neither a GPU stack nor the simulator. hydra, omegaconf and numpy are imported for real.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "gym_pusht",
    "gymnasium",
    "torch",
    "wandb",
    "wandb_gql",
    "wandb_workspaces",
    "zarr",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}

# Every cross-reference must resolve. Types from packages without an inventory (or mocked ones the inventory does
# not list) are the only exceptions.
nitpicky = True
nitpick_ignore_regex = [
    (r"py:.*", r"(omegaconf|hydra|wandb|gymnasium|gym_pusht|zarr)(\..*)?"),
    (
        r"py:.*",
        r"(DictConfig|Rollouts|Split|Frame|Padding|Conditioning|TimeOrder|Tangent|Reduction|Predict)",
    ),
]

copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# Names as the source spells them (unresolved annotations of dataclass fields and attributes) -> as the inventories do.
TYPE_ALIASES = {
    "np.ndarray": "numpy.ndarray",
    "Path": "pathlib.Path",
    "nn.Parameter": "torch.nn.parameter.Parameter",
    "torch.nn.Parameter": "torch.nn.parameter.Parameter",
}


def resolve_alias(app, env, node, contnode):
    """
    Retries an unresolved reference under the name its inventory uses (the missing-reference event).
    """
    target = TYPE_ALIASES.get(node.get("reftarget"))
    if target is None:
        return None
    from sphinx.ext.intersphinx import missing_reference

    node["reftarget"] = target
    return missing_reference(app, env, node, contnode)


def setup(app):
    """
    Resolves the aliases before intersphinx gives up on them.
    """
    app.connect("missing-reference", resolve_alias, priority=400)
