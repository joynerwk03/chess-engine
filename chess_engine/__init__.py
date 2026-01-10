"""
Chess Engine Package

A modular chess engine with alpha-beta search, quiescence search,
transposition tables, and machine learning-tuned evaluation.
"""

from chess_engine.engine import Engine
from chess_engine.search import search
from chess_engine.eval_main import evaluate_board
from chess_engine.zobrist import compute_zobrist_hash

__all__ = ["Engine", "search", "evaluate_board", "compute_zobrist_hash"]
__version__ = "1.0.0"
