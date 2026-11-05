"""
Python source code parser.

This module test the tree sitter is properly working

"""
from tree_sitter_language_pack import get_parser
"""Enabling support for three languages"""
languages = ["javascript", "typescript", "go"]

for language in languages:
    parser = get_parser(language)
    print(f"{language}: parser loaded successfully")
