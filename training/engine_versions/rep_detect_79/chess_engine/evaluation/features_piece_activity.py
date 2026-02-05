"""
Piece activity evaluation features:
- Trapped pieces
- Knight on rim
- Minor piece coordination (knights vs bishops based on position)
"""
import chess
from chess_engine.tuned_weights import (
    TRAPPED_PIECE_PENALTY,
    KNIGHT_ON_RIM_PENALTY,
    KNIGHT_PAWN_BONUS,
    BISHOP_OPEN_BONUS,
)

# Rim squares (a-file and h-file, ranks 1 and 8)
RIM_SQUARES = (
    chess.SquareSet(chess.BB_FILE_A) | 
    chess.SquareSet(chess.BB_FILE_H)
)


def get_trapped_pieces_score(board: chess.Board) -> int:
    """
    Penalize pieces with very limited mobility (trapped).
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        
        # Check knights
        for sq in board.pieces(chess.KNIGHT, color):
            # Count legal moves for this knight
            moves = 0
            for attack_sq in board.attacks(sq):
                # Can move there if empty or enemy piece
                piece_at_target = board.piece_at(attack_sq)
                if piece_at_target is None or piece_at_target.color != color:
                    moves += 1
            
            if moves <= 1:
                score += sign * TRAPPED_PIECE_PENALTY
            
            # Knight on rim penalty
            if sq in RIM_SQUARES:
                score += sign * KNIGHT_ON_RIM_PENALTY
        
        # Check bishops
        for sq in board.pieces(chess.BISHOP, color):
            moves = 0
            for attack_sq in board.attacks(sq):
                piece_at_target = board.piece_at(attack_sq)
                if piece_at_target is None or piece_at_target.color != color:
                    moves += 1
            
            if moves <= 1:
                score += sign * TRAPPED_PIECE_PENALTY
        
        # Check rooks (trapped rooks are very bad)
        for sq in board.pieces(chess.ROOK, color):
            moves = 0
            for attack_sq in board.attacks(sq):
                piece_at_target = board.piece_at(attack_sq)
                if piece_at_target is None or piece_at_target.color != color:
                    moves += 1
            
            if moves <= 2:
                score += sign * TRAPPED_PIECE_PENALTY
    
    return score


def get_minor_piece_coordination_score(board: chess.Board) -> int:
    """
    Evaluate knights vs bishops based on pawn count.
    Knights are better in closed positions (more pawns).
    Bishops are better in open positions (fewer pawns).
    """
    score = 0
    
    # Count total pawns
    total_pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
    pawns_missing = 16 - total_pawns  # How many pawns have been captured
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        
        # Knights get bonus based on number of pawns on board
        num_knights = len(board.pieces(chess.KNIGHT, color))
        knight_bonus = num_knights * total_pawns * KNIGHT_PAWN_BONUS // 8  # Normalize
        score += sign * knight_bonus
        
        # Bishops get bonus based on openness (missing pawns)
        num_bishops = len(board.pieces(chess.BISHOP, color))
        bishop_bonus = num_bishops * pawns_missing * BISHOP_OPEN_BONUS // 8  # Normalize
        score += sign * bishop_bonus
    
    return score


def get_piece_activity_score(board: chess.Board) -> int:
    """Combined piece activity evaluation."""
    score = 0
    score += get_trapped_pieces_score(board)
    score += get_minor_piece_coordination_score(board)
    return score
