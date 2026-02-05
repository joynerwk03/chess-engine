"""
Bishop-specific evaluation features:
- Bad bishop detection
- Fianchetto bonus
"""
import chess
from chess_engine.tuned_weights import (
    BAD_BISHOP_PENALTY,
    VERY_BAD_BISHOP_PENALTY,
    FIANCHETTO_BONUS,
)

# Light squares and dark squares
LIGHT_SQUARES = chess.SquareSet(chess.BB_LIGHT_SQUARES)
DARK_SQUARES = chess.SquareSet(chess.BB_DARK_SQUARES)


def is_light_square(square: int) -> bool:
    """Check if a square is a light square."""
    return square in LIGHT_SQUARES


def get_bad_bishop_score(board: chess.Board) -> int:
    """
    Penalize bishops blocked by own pawns on the same color squares.
    A bishop is "bad" when many of its own pawns are on the same color squares.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        bishops = board.pieces(chess.BISHOP, color)
        pawns = board.pieces(chess.PAWN, color)
        
        for bishop_sq in bishops:
            bishop_on_light = is_light_square(bishop_sq)
            
            # Count own pawns on same color squares as bishop
            blocking_pawns = 0
            for pawn_sq in pawns:
                pawn_on_light = is_light_square(pawn_sq)
                if pawn_on_light == bishop_on_light:
                    # Extra penalty for center pawns
                    pawn_file = chess.square_file(pawn_sq)
                    if pawn_file in [3, 4]:  # d and e files
                        blocking_pawns += 2
                    else:
                        blocking_pawns += 1
            
            if blocking_pawns >= 4:
                score += sign * VERY_BAD_BISHOP_PENALTY
            elif blocking_pawns >= 2:
                score += sign * BAD_BISHOP_PENALTY * (blocking_pawns // 2)
    
    return score


def get_fianchetto_score(board: chess.Board) -> int:
    """
    Bonus for fianchettoed bishops (bishop on g2/b2/g7/b7 with appropriate structure).
    """
    score = 0
    
    # White fianchetto on kingside (Bg2 with pawns on f2/g3/h2)
    if board.piece_at(chess.G2) == chess.Piece(chess.BISHOP, chess.WHITE):
        # Check for supporting pawn structure
        has_g3_pawn = board.piece_at(chess.G3) == chess.Piece(chess.PAWN, chess.WHITE)
        has_f2_pawn = board.piece_at(chess.F2) == chess.Piece(chess.PAWN, chess.WHITE)
        has_h2_pawn = board.piece_at(chess.H2) == chess.Piece(chess.PAWN, chess.WHITE)
        
        if has_g3_pawn and (has_f2_pawn or has_h2_pawn):
            score += FIANCHETTO_BONUS
    
    # White fianchetto on queenside (Bb2)
    if board.piece_at(chess.B2) == chess.Piece(chess.BISHOP, chess.WHITE):
        has_b3_pawn = board.piece_at(chess.B3) == chess.Piece(chess.PAWN, chess.WHITE)
        has_a2_pawn = board.piece_at(chess.A2) == chess.Piece(chess.PAWN, chess.WHITE)
        has_c2_pawn = board.piece_at(chess.C2) == chess.Piece(chess.PAWN, chess.WHITE)
        
        if has_b3_pawn and (has_a2_pawn or has_c2_pawn):
            score += FIANCHETTO_BONUS
    
    # Black fianchetto on kingside (Bg7)
    if board.piece_at(chess.G7) == chess.Piece(chess.BISHOP, chess.BLACK):
        has_g6_pawn = board.piece_at(chess.G6) == chess.Piece(chess.PAWN, chess.BLACK)
        has_f7_pawn = board.piece_at(chess.F7) == chess.Piece(chess.PAWN, chess.BLACK)
        has_h7_pawn = board.piece_at(chess.H7) == chess.Piece(chess.PAWN, chess.BLACK)
        
        if has_g6_pawn and (has_f7_pawn or has_h7_pawn):
            score -= FIANCHETTO_BONUS
    
    # Black fianchetto on queenside (Bb7)
    if board.piece_at(chess.B7) == chess.Piece(chess.BISHOP, chess.BLACK):
        has_b6_pawn = board.piece_at(chess.B6) == chess.Piece(chess.PAWN, chess.BLACK)
        has_a7_pawn = board.piece_at(chess.A7) == chess.Piece(chess.PAWN, chess.BLACK)
        has_c7_pawn = board.piece_at(chess.C7) == chess.Piece(chess.PAWN, chess.BLACK)
        
        if has_b6_pawn and (has_a7_pawn or has_c7_pawn):
            score -= FIANCHETTO_BONUS
    
    return score


def get_bishop_features_score(board: chess.Board) -> int:
    """Combined bishop features evaluation."""
    score = 0
    score += get_bad_bishop_score(board)
    score += get_fianchetto_score(board)
    return score
