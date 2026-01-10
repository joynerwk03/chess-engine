import chess
from chess_engine.tuned_weights import (
    KING_SHIELD_BONUS,
    KING_OPEN_FILE_PENALTY,
    KING_ATTACK_WEIGHT,
)

FILE_SETS = [
    chess.SquareSet(chess.BB_FILE_A),
    chess.SquareSet(chess.BB_FILE_B),
    chess.SquareSet(chess.BB_FILE_C),
    chess.SquareSet(chess.BB_FILE_D),
    chess.SquareSet(chess.BB_FILE_E),
    chess.SquareSet(chess.BB_FILE_F),
    chess.SquareSet(chess.BB_FILE_G),
    chess.SquareSet(chess.BB_FILE_H),
]

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _king_zone_squares(board: chess.Board, king_sq: int) -> chess.SquareSet:
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    bb = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = kf + df, kr + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                bb |= 1 << chess.square(nf, nr)
    return chess.SquareSet(bb)


def _pawn_shield_score(board: chess.Board, color: bool, king_sq: int) -> int:
    score = 0
    kf = chess.square_file(king_sq)
    files = [kf - 1, kf, kf + 1]
    pawns = board.pieces(chess.PAWN, color)
    if color == chess.WHITE:
        ranks = [1, 2]  # ranks 2-3 as a typical shield region
    else:
        ranks = [6, 5]  # ranks 7-6
    for f in files:
        if 0 <= f <= 7:
            for r in ranks:
                sq = chess.square(f, r)
                if sq in pawns:
                    score += KING_SHIELD_BONUS
    return score


def _open_files_near_king(board: chess.Board, color: bool, king_sq: int) -> int:
    penalty = 0
    kf = chess.square_file(king_sq)
    files = [kf - 1, kf, kf + 1]
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    for f in files:
        if 0 <= f <= 7:
            mask = FILE_SETS[f]
            has_white = len(wp & mask) > 0
            has_black = len(bp & mask) > 0
            if not has_white and not has_black:
                penalty += KING_OPEN_FILE_PENALTY
            else:
                # Semi-open for the side: no friendly pawns
                if color == chess.WHITE:
                    if not has_white:
                        penalty += KING_OPEN_FILE_PENALTY
                else:
                    if not has_black:
                        penalty += KING_OPEN_FILE_PENALTY
    return penalty


def _attacker_threat_score(board: chess.Board, color: bool, king_sq: int) -> int:
    # Enemy pieces attacking the king zone
    zone = _king_zone_squares(board, king_sq)
    enemy = not color
    total = 0
    for sq, piece in board.piece_map().items():
        if piece.color != enemy:
            continue
        if piece.piece_type not in PIECE_VALUES:
            continue
        attacks = board.attacks(sq)
        if len(attacks & zone) > 0:
            total += PIECE_VALUES[piece.piece_type]
    return total * KING_ATTACK_WEIGHT


def get_king_safety_score(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        king_sq = board.king(color)
        if king_sq is None:
            continue
        score_part = 0
        score_part += _pawn_shield_score(board, color, king_sq)
        score_part += _open_files_near_king(board, color, king_sq)
        score_part -= _attacker_threat_score(board, color, king_sq)
        score += score_part if color == chess.WHITE else -score_part
    return score
