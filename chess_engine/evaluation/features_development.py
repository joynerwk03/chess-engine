"""
Development evaluation features:
- Undeveloped pieces penalty
- Early queen development penalty
"""
import chess
from chess_engine.tuned_weights import (
    UNDEVELOPED_MINOR_PENALTY,
    EARLY_QUEEN_PENALTY,
)


def get_development_score(board: chess.Board) -> int:
    """
    Evaluate piece development.
    Penalize undeveloped minor pieces and early queen moves.
    Only applies in the opening phase (first ~15 moves).
    """
    # Rough move counter based on piece positions
    # This is approximate since we don't track actual moves
    total_pieces = len(board.piece_map())
    if total_pieces < 26:  # Not opening anymore if pieces traded
        return 0
    
    score = 0
    
    # Check White development
    score += _development_for_color(board, chess.WHITE)
    score -= _development_for_color(board, chess.BLACK)
    
    return score


def _development_for_color(board: chess.Board, color: bool) -> int:
    """Calculate development score for one color."""
    score = 0
    
    if color == chess.WHITE:
        back_rank = 0
        queen_start = chess.D1
        knight_starts = [chess.B1, chess.G1]
        bishop_starts = [chess.C1, chess.F1]
    else:
        back_rank = 7
        queen_start = chess.D8
        knight_starts = [chess.B8, chess.G8]
        bishop_starts = [chess.C8, chess.F8]
    
    # Count undeveloped minor pieces
    undeveloped_minors = 0
    
    for knight_sq in knight_starts:
        if board.piece_at(knight_sq) == chess.Piece(chess.KNIGHT, color):
            undeveloped_minors += 1
    
    for bishop_sq in bishop_starts:
        if board.piece_at(bishop_sq) == chess.Piece(chess.BISHOP, color):
            undeveloped_minors += 1
    
    score += undeveloped_minors * UNDEVELOPED_MINOR_PENALTY
    
    # Check for early queen development
    queen = board.pieces(chess.QUEEN, color)
    if queen:
        queen_sq = list(queen)[0]
        
        # If queen is not on start square and minors are undeveloped
        if queen_sq != queen_start and undeveloped_minors >= 2:
            score += EARLY_QUEEN_PENALTY
    
    return score


def get_piece_development_count(board: chess.Board, color: bool) -> int:
    """Count how many minor pieces are developed (not on starting squares)."""
    if color == chess.WHITE:
        knight_starts = [chess.B1, chess.G1]
        bishop_starts = [chess.C1, chess.F1]
    else:
        knight_starts = [chess.B8, chess.G8]
        bishop_starts = [chess.C8, chess.F8]
    
    developed = 0
    
    # Check knights
    for sq in board.pieces(chess.KNIGHT, color):
        if sq not in knight_starts:
            developed += 1
    
    # Check bishops  
    for sq in board.pieces(chess.BISHOP, color):
        if sq not in bishop_starts:
            developed += 1
    
    return developed
