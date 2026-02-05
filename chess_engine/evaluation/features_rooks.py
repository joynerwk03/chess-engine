"""
Rook-specific evaluation features:
- Rook on 7th rank
- Doubled rooks on 7th
- Connected rooks
"""
import chess
from chess_engine.tuned_weights import (
    ROOK_ON_7TH_BONUS,
    DOUBLED_ROOKS_ON_7TH_BONUS,
    CONNECTED_ROOKS_BONUS,
)


def get_rook_on_7th_score(board: chess.Board) -> int:
    """
    Bonus for rooks on the 7th rank (2nd rank for Black).
    Even bigger bonus for doubled rooks on 7th.
    """
    score = 0
    
    # White rooks on 7th rank (rank index 6)
    white_rooks = board.pieces(chess.ROOK, chess.WHITE)
    white_rooks_on_7th = [sq for sq in white_rooks if chess.square_rank(sq) == 6]
    
    if len(white_rooks_on_7th) >= 2:
        score += DOUBLED_ROOKS_ON_7TH_BONUS
    elif len(white_rooks_on_7th) == 1:
        score += ROOK_ON_7TH_BONUS
    
    # Black rooks on 2nd rank (rank index 1)
    black_rooks = board.pieces(chess.ROOK, chess.BLACK)
    black_rooks_on_2nd = [sq for sq in black_rooks if chess.square_rank(sq) == 1]
    
    if len(black_rooks_on_2nd) >= 2:
        score -= DOUBLED_ROOKS_ON_7TH_BONUS
    elif len(black_rooks_on_2nd) == 1:
        score -= ROOK_ON_7TH_BONUS
    
    return score


def get_connected_rooks_score(board: chess.Board) -> int:
    """
    Bonus for connected rooks (rooks that defend each other on the same rank or file).
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        rooks = list(board.pieces(chess.ROOK, color))
        
        if len(rooks) >= 2:
            rook1, rook2 = rooks[0], rooks[1]
            r1_file, r1_rank = chess.square_file(rook1), chess.square_rank(rook1)
            r2_file, r2_rank = chess.square_file(rook2), chess.square_rank(rook2)
            
            connected = False
            
            # Check if on same rank
            if r1_rank == r2_rank:
                # Check if no pieces between them
                min_file, max_file = min(r1_file, r2_file), max(r1_file, r2_file)
                blocked = False
                for f in range(min_file + 1, max_file):
                    if board.piece_at(chess.square(f, r1_rank)):
                        blocked = True
                        break
                if not blocked:
                    connected = True
            
            # Check if on same file
            elif r1_file == r2_file:
                # Check if no pieces between them
                min_rank, max_rank = min(r1_rank, r2_rank), max(r1_rank, r2_rank)
                blocked = False
                for r in range(min_rank + 1, max_rank):
                    if board.piece_at(chess.square(r1_file, r)):
                        blocked = True
                        break
                if not blocked:
                    connected = True
            
            if connected:
                score += sign * CONNECTED_ROOKS_BONUS
    
    return score


def get_rook_features_score(board: chess.Board) -> int:
    """Combined rook features evaluation."""
    score = 0
    score += get_rook_on_7th_score(board)
    score += get_connected_rooks_score(board)
    return score
