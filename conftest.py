"""Makes the repository root importable so `from transformer import ...` resolves.

pytest inserts the directory containing this file at the front of sys.path, which is
what lets the tests in tests/ import the package sitting alongside them. Without it,
plain `pytest` only sees tests/ on the path and raises ModuleNotFoundError.
"""
