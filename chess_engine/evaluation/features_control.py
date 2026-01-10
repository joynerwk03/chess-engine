import chess
from chess_engine.tuned_weights import UNCASTLED_KING_PENALTY, CENTER_CONTROL_BONUS

CENTER_SQUARES = [
    chess.D4, chess.E4, chess.D5, chess.E5
]


def get_uncastled_king_penalty(board: chess.Board, phase: float) -> int:
    """
    Penalize sides that still have castling rights (i.e., have not castled yet) in middlegame.
    phase in [0,1]: apply only if phase > 0.5.
    """
    if phase <= 0.5:
        return 0
    score = 0
    # If a side has any castling right, count as uncastled
    if board.has_castling_rights(chess.WHITE):
        score += UNCASTLED_KING_PENALTY
    if board.has_castling_rights(chess.BLACK):
        score -= UNCASTLED_KING_PENALTY
    return score


def get_center_control_score(board: chess.Board) -> int:
    score = 0
    for sq in CENTER_SQUARES:
        w_attackers = len(board.attackers(chess.WHITE, sq))
        b_attackers = len(board.attackers(chess.BLACK, sq))
        score += (w_attackers - b_attackers) * CENTER_CONTROL_BONUS
    return score
