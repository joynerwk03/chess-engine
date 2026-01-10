import chess
from chess_engine.tuned_weights import (
    BISHOP_PAIR_BONUS,
    ROOK_ON_OPEN_FILE_BONUS,
    ROOK_ON_SEMI_OPEN_FILE_BONUS,
)


def get_bishop_pair_score(board: chess.Board) -> int:
    score = 0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS
    return score


essentially_all_files = [
    chess.BB_FILE_A, chess.BB_FILE_B, chess.BB_FILE_C, chess.BB_FILE_D,
    chess.BB_FILE_E, chess.BB_FILE_F, chess.BB_FILE_G, chess.BB_FILE_H,
]


def _file_mask(file_idx: int) -> int:
    # python-chess provides constants; use bitboard files
    return essentially_all_files[file_idx]


def get_rook_placement_score(board: chess.Board) -> int:
    score = 0
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    # White rooks
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        f = chess.square_file(sq)
        mask = _file_mask(f)
        has_white_pawn = bool(white_pawns & mask)
        has_black_pawn = bool(black_pawns & mask)
        if not has_white_pawn and not has_black_pawn:
            score += ROOK_ON_OPEN_FILE_BONUS
        elif not has_white_pawn:
            score += ROOK_ON_SEMI_OPEN_FILE_BONUS

    # Black rooks
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        f = chess.square_file(sq)
        mask = _file_mask(f)
        has_white_pawn = bool(white_pawns & mask)
        has_black_pawn = bool(black_pawns & mask)
        if not has_white_pawn and not has_black_pawn:
            score -= ROOK_ON_OPEN_FILE_BONUS
        elif not has_black_pawn:
            score -= ROOK_ON_SEMI_OPEN_FILE_BONUS

    return score
