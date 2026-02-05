"""
Chess Engine Package

A modular chess engine with alpha-beta search, quiescence search,
transposition tables, and machine learning-tuned evaluation.
"""

from chess_engine.engine import Engine
from chess_engine.search import iterative_deepening_search, alpha_beta_search
from chess_engine.eval_main import evaluate_board
from chess_engine.zobrist import compute_zobrist_hash

__all__ = ["Engine", "evaluate_board", "compute_zobrist_hash", "iterative_deepening_search", "alpha_beta_search"]
__version__ = "1.0.0"
