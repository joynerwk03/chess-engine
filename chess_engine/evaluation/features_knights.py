import chess
from chess_engine.tuned_weights import KNIGHT_OUTPOST_BONUS


def _enemy_pawn_attacks_square(board: chess.Board, sq: int, enemy_color: bool) -> bool:
    # Any enemy pawn that attacks sq?
    attackers = board.attackers(enemy_color, sq)
    pawns = board.pieces(chess.PAWN, enemy_color)
    return len(attackers & pawns) > 0


def _supported_by_friendly_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    # Any friendly pawn that attacks sq?
    attackers = board.attackers(color, sq)
    pawns = board.pieces(chess.PAWN, color)
    return len(attackers & pawns) > 0


def get_knight_outpost_score(board: chess.Board) -> int:
    score = 0
    for color in (chess.WHITE, chess.BLACK):
        knights = board.pieces(chess.KNIGHT, color)
        for sq in knights:
            rank = chess.square_rank(sq)
            # Deep squares: White on ranks 4-7 (idx 3..6), Black on ranks 1-4 (idx 1..4)
            if color == chess.WHITE:
                deep = 3 <= rank <= 6
            else:
                deep = 1 <= rank <= 4
            if not deep:
                continue
            if not _supported_by_friendly_pawn(board, sq, color):
                continue
            if _enemy_pawn_attacks_square(board, sq, not color):
                continue
            score += KNIGHT_OUTPOST_BONUS if color == chess.WHITE else -KNIGHT_OUTPOST_BONUS
    return score
