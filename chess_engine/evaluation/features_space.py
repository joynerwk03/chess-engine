import chess
from chess_engine.tuned_weights import MOBILITY_BONUS, SPACE_BONUS

CENTER_RANKS = {2, 3, 4, 5}  # ranks 3-6 (0-indexed)


def get_mobility_score(board: chess.Board) -> int:
    # Count legal moves for major/minor pieces only
    def count_side(color: bool) -> int:
        total = 0
        for sq, piece in board.piece_map().items():
            if piece.color != color:
                continue
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                total += len(board.attacks(sq))
        return total

    white = count_side(chess.WHITE)
    black = count_side(chess.BLACK)
    return (white - black) * MOBILITY_BONUS


def get_space_score(board: chess.Board) -> int:
    # Count pawn-controlled squares in central ranks
    def pawn_control(color: bool) -> int:
        pawns = board.pieces(chess.PAWN, color)
        count = 0
        for sq in pawns:
            attacks = board.attacks(sq)
            for a in attacks:
                if chess.square_rank(a) in CENTER_RANKS:
                    count += 1
        return count

    white = pawn_control(chess.WHITE)
    black = pawn_control(chess.BLACK)
    return (white - black) * SPACE_BONUS
