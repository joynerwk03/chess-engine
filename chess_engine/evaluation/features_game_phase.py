import chess
from chess_engine.tuned_weights import (
    PIECE_VALUE_KNIGHT,
    PIECE_VALUE_BISHOP,
    PIECE_VALUE_ROOK,
    PIECE_VALUE_QUEEN,
)

# Sum of non-pawn, non-king material for normalization
_FULL_PHASE_MATERIAL = 2 * (
    2 * PIECE_VALUE_KNIGHT + 2 * PIECE_VALUE_BISHOP + 2 * PIECE_VALUE_ROOK + PIECE_VALUE_QUEEN
) + (
    2 * PIECE_VALUE_ROOK + 2 * PIECE_VALUE_QUEEN  # account for full set per side baseline
)
# Fallback if weights are unusual
if _FULL_PHASE_MATERIAL <= 0:
    _FULL_PHASE_MATERIAL = 1


def _current_phase_material(board: chess.Board) -> int:
    total = 0
    for _, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN or piece.piece_type == chess.KING:
            continue
        if piece.piece_type == chess.KNIGHT:
            total += PIECE_VALUE_KNIGHT
        elif piece.piece_type == chess.BISHOP:
            total += PIECE_VALUE_BISHOP
        elif piece.piece_type == chess.ROOK:
            total += PIECE_VALUE_ROOK
        elif piece.piece_type == chess.QUEEN:
            total += PIECE_VALUE_QUEEN
    return total


def calculate_game_phase(board: chess.Board) -> float:
    """
    Returns a scalar in [0.0, 1.0]. 1.0 ~ opening/middlegame, 0.0 ~ pure endgame
    based on remaining non-pawn, non-king material.
    """
    remaining = _current_phase_material(board)
    phase = min(1.0, max(0.0, remaining / _FULL_PHASE_MATERIAL))
    return phase
