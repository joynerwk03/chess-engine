import chess
from chess_engine.tuned_weights import (
    DOUBLED_PAWN_PENALTY,
    ISOLATED_PAWN_PENALTY,
    PASSED_PAWN_BONUS,
    OUTSIDE_PASSED_PAWN_BONUS,
)

# Use SquareSets for file masks to enable set ops and len()
FILE_MASKS = [
    chess.SquareSet(chess.BB_FILE_A),
    chess.SquareSet(chess.BB_FILE_B),
    chess.SquareSet(chess.BB_FILE_C),
    chess.SquareSet(chess.BB_FILE_D),
    chess.SquareSet(chess.BB_FILE_E),
    chess.SquareSet(chess.BB_FILE_F),
    chess.SquareSet(chess.BB_FILE_G),
    chess.SquareSet(chess.BB_FILE_H),
]


def _doubled_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    for f in range(8):
        mask = FILE_MASKS[f]
        wc = len(wp & mask)
        bc = len(bp & mask)
        if wc > 1:
            score += (wc - 1) * DOUBLED_PAWN_PENALTY
        if bc > 1:
            score -= (bc - 1) * DOUBLED_PAWN_PENALTY
    return score


def _isolated_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)

    # White isolated
    for sq in wp:
        f = chess.square_file(sq)
        left_mask = FILE_MASKS[f - 1] if f > 0 else chess.SquareSet(0)
        right_mask = FILE_MASKS[f + 1] if f < 7 else chess.SquareSet(0)
        if len(wp & (left_mask | right_mask)) == 0:
            score += ISOLATED_PAWN_PENALTY

    # Black isolated
    for sq in bp:
        f = chess.square_file(sq)
        left_mask = FILE_MASKS[f - 1] if f > 0 else chess.SquareSet(0)
        right_mask = FILE_MASKS[f + 1] if f < 7 else chess.SquareSet(0)
        if len(bp & (left_mask | right_mask)) == 0:
            score -= ISOLATED_PAWN_PENALTY

    return score


def _forward_mask_white(file_idx: int, rank_idx: int) -> chess.SquareSet:
    bb = 0
    for ff in (file_idx - 1, file_idx, file_idx + 1):
        if 0 <= ff <= 7:
            for r in range(rank_idx + 1, 8):
                bb |= 1 << chess.square(ff, r)
    return chess.SquareSet(bb)


def _forward_mask_black(file_idx: int, rank_idx: int) -> chess.SquareSet:
    bb = 0
    for ff in (file_idx - 1, file_idx, file_idx + 1):
        if 0 <= ff <= 7:
            for r in range(rank_idx - 1, -1, -1):
                bb |= 1 << chess.square(ff, r)
    return chess.SquareSet(bb)


def _passed_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)

    # White passed pawns
    for sq in wp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        mask = _forward_mask_white(f, r)
        if len(bp & mask) == 0:
            score += PASSED_PAWN_BONUS[r]
            if f in (0, 7):  # a- or h-file
                score += OUTSIDE_PASSED_PAWN_BONUS

    # Black passed pawns
    for sq in bp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        mask = _forward_mask_black(f, r)
        if len(wp & mask) == 0:
            score -= PASSED_PAWN_BONUS[7 - r]
            if f in (0, 7):
                score -= OUTSIDE_PASSED_PAWN_BONUS

    return score


def get_pawn_structure_score(board: chess.Board) -> int:
    """White-centric pawn structure evaluation.
    Negative penalties reduce the side's score; bonuses increase it.
    """
    score = 0
    score += _doubled_pawns_score(board)
    score += _isolated_pawns_score(board)
    score += _passed_pawns_score(board)
    return score
