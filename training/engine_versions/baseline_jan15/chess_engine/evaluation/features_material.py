import chess
from chess_engine.tuned_weights import (
    PIECE_VALUE_PAWN,
    PIECE_VALUE_KNIGHT,
    PIECE_VALUE_BISHOP,
    PIECE_VALUE_ROOK,
    PIECE_VALUE_QUEEN,
    PIECE_VALUE_KING,
)

PIECE_VALUE_MAP = {
    chess.PAWN: PIECE_VALUE_PAWN,
    chess.KNIGHT: PIECE_VALUE_KNIGHT,
    chess.BISHOP: PIECE_VALUE_BISHOP,
    chess.ROOK: PIECE_VALUE_ROOK,
    chess.QUEEN: PIECE_VALUE_QUEEN,
    chess.KING: PIECE_VALUE_KING,
}


def get_material_score(board: chess.Board) -> int:
    score = 0
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUE_MAP[piece.piece_type]
        score += val if piece.color == chess.WHITE else -val
    return score
