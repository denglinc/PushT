"""
Tests run on the CPU unless JAX_PLATFORMS says otherwise, so they never contend for a GPU that is training.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
