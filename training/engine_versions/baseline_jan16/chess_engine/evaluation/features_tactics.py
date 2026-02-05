"""
Tactical evaluation features:
- Hanging (undefended) pieces
- Threats and attacks
- Pins and skewers
"""
import chess
from chess_engine.tuned_weights import (
    HANGING_PAWN_PENALTY,
    HANGING_MINOR_PENALTY,
    HANGING_ROOK_PENALTY,
    HANGING_QUEEN_PENALTY,
    ATTACKED_BY_LESSER_PENALTY,
    THREAT_BONUS,
    PIN_BONUS,
    PINNED_PIECE_PENALTY,
    PIECE_VALUE_PAWN,
    PIECE_VALUE_KNIGHT,
    PIECE_VALUE_BISHOP,
    PIECE_VALUE_ROOK,
    PIECE_VALUE_QUEEN,
)

PIECE_VALUES = {
    chess.PAWN: PIECE_VALUE_PAWN,
    chess.KNIGHT: PIECE_VALUE_KNIGHT,
    chess.BISHOP: PIECE_VALUE_BISHOP,
    chess.ROOK: PIECE_VALUE_ROOK,
    chess.QUEEN: PIECE_VALUE_QUEEN,
    chess.KING: 20000,
}

HANGING_PENALTIES = {
    chess.PAWN: HANGING_PAWN_PENALTY,
    chess.KNIGHT: HANGING_MINOR_PENALTY,
    chess.BISHOP: HANGING_MINOR_PENALTY,
    chess.ROOK: HANGING_ROOK_PENALTY,
    chess.QUEEN: HANGING_QUEEN_PENALTY,
}


def get_attackers_defenders(board: chess.Board, square: int, color: bool) -> tuple[chess.SquareSet, chess.SquareSet]:
    """Get attackers and defenders of a square for a given piece color."""
    attackers = board.attackers(not color, square)
    defenders = board.attackers(color, square)
    return attackers, defenders


def get_hanging_pieces_score(board: chess.Board) -> int:
    """
    Evaluate hanging (undefended) pieces.
    A piece is hanging if it's attacked and not adequately defended.
    Only penalize if the piece is truly in danger.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        enemy_to_move = (board.turn != color)
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                attackers, defenders = get_attackers_defenders(board, square, color)
                
                if not attackers:
                    continue
                
                if not defenders:
                    # Completely undefended and attacked - very bad if enemy to move
                    penalty = HANGING_PENALTIES.get(piece_type, 0)
                    if enemy_to_move:
                        score += sign * penalty
                    else:
                        # We can move it, half penalty
                        score += sign * (penalty // 2)
                else:
                    # Has defenders - only worry if attackers > defenders and attacked by lesser
                    attacker_values = []
                    for sq in attackers:
                        pt = board.piece_type_at(sq)
                        if pt is not None:
                            attacker_values.append(PIECE_VALUES.get(pt, 20000))
                    
                    defender_values = []
                    for sq in defenders:
                        pt = board.piece_type_at(sq)
                        if pt is not None:
                            defender_values.append(PIECE_VALUES.get(pt, 20000))
                    
                    if attacker_values:
                        min_attacker_value = min(attacker_values)
                        piece_value = PIECE_VALUES.get(piece_type, 0)
                        
                        # Only penalize if attacked by lesser AND not well defended
                        # Well defended means defenders >= attackers
                        if min_attacker_value < piece_value and len(attackers) > len(defenders):
                            score += sign * ATTACKED_BY_LESSER_PENALTY
    
    return score


def get_threats_score(board: chess.Board) -> int:
    """
    Evaluate threats - bonus for attacking enemy pieces.
    Only count threats where:
    1. The piece is undefended OR
    2. The attacker is worth less than the target
    This avoids counting "queen attacks 6 pieces" as a huge advantage
    when none are actually capturable.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        enemy_color = not color
        
        # Count meaningful attacks on enemy pieces
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            piece_value = PIECE_VALUES.get(piece_type, 0)
            
            for square in board.pieces(piece_type, enemy_color):
                our_attackers = board.attackers(color, square)
                defenders = board.attackers(enemy_color, square)
                
                if not our_attackers:
                    continue
                
                # Get the value of our least valuable attacker
                min_attacker_value = float('inf')
                for attacker_sq in our_attackers:
                    attacker_type = board.piece_type_at(attacker_sq)
                    if attacker_type and attacker_type != chess.KING:
                        val = PIECE_VALUES.get(attacker_type, 0)
                        min_attacker_value = min(min_attacker_value, val)
                
                if min_attacker_value == float('inf'):
                    continue  # Only king attacks, not a real threat
                
                # Only count as a threat if:
                # 1. Undefended, OR
                # 2. Attacked by a piece worth less than the target
                if not defenders:
                    # Undefended piece under attack - significant threat
                    score += sign * THREAT_BONUS
                elif min_attacker_value < piece_value:
                    # Attacked by lesser piece - even if defended, this is tension
                    score += sign * (THREAT_BONUS // 2)
    
    return score


def _is_pinned_to_king(board: chess.Board, square: int, color: bool) -> bool:
    """Check if a piece is pinned to its king."""
    king_square = board.king(color)
    if king_square is None:
        return False
    
    # Get the piece at the square
    piece = board.piece_at(square)
    if piece is None or piece.color != color:
        return False
    
    # Check if removing this piece would expose the king to attack
    # by a sliding piece (bishop, rook, queen)
    enemy_color = not color
    
    # Direction from king to piece
    king_file, king_rank = chess.square_file(king_square), chess.square_rank(king_square)
    piece_file, piece_rank = chess.square_file(square), chess.square_rank(square)
    
    df = piece_file - king_file
    dr = piece_rank - king_rank
    
    if df == 0 and dr == 0:
        return False
    
    # Normalize direction
    if df != 0:
        df = df // abs(df)
    if dr != 0:
        dr = dr // abs(dr)
    
    # Check if there's an enemy sliding piece in this direction beyond our piece
    current_file, current_rank = piece_file + df, piece_rank + dr
    
    while 0 <= current_file <= 7 and 0 <= current_rank <= 7:
        check_square = chess.square(current_file, current_rank)
        check_piece = board.piece_at(check_square)
        
        if check_piece is not None:
            if check_piece.color == enemy_color:
                # Is it a sliding piece that can attack along this direction?
                if df == 0 or dr == 0:  # Horizontal/vertical
                    if check_piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        return True
                else:  # Diagonal
                    if check_piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                        return True
            break
        
        current_file += df
        current_rank += dr
    
    return False


def get_pins_score(board: chess.Board) -> int:
    """
    Evaluate pins - bonus for creating pins, penalty for having pinned pieces.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        
        # Check each of our pieces for pins
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, color):
                if _is_pinned_to_king(board, square, color):
                    score += sign * PINNED_PIECE_PENALTY
        
        # Check if we're pinning enemy pieces (inverse bonus)
        enemy_color = not color
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            for square in board.pieces(piece_type, enemy_color):
                if _is_pinned_to_king(board, square, enemy_color):
                    score += sign * PIN_BONUS
    
    return score


def get_check_threats_score(board: chess.Board) -> int:
    """
    Evaluate check threats - the side to move gets extra bonus.
    This is slightly asymmetric because the side to move can execute threats immediately.
    """
    score = 0
    stm = board.turn
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        enemy_color = not color
        enemy_king = board.king(enemy_color)
        
        if enemy_king is None:
            continue
        
        # Count how many of our pieces could give check in one move
        check_potential = 0
        
        # Squares around enemy king
        king_ring = chess.SquareSet(chess.BB_KING_ATTACKS[enemy_king])
        
        # For each of our pieces, check if they attack king ring squares
        for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            for sq in board.pieces(piece_type, color):
                attacks = board.attacks(sq)
                if attacks & king_ring:
                    if piece_type == chess.QUEEN:
                        check_potential += 20
                    elif piece_type == chess.ROOK:
                        check_potential += 15
                    else:
                        check_potential += 10
        
        # Bonus multiplier for side to move (can execute threats immediately)
        multiplier = 1.5 if color == stm else 1.0
        score += sign * int(min(check_potential, 50) * multiplier)
    
    return score
    
    return score


def get_knight_fork_threats_score(board: chess.Board) -> int:
    """
    Specifically evaluate knight fork threats.
    Knights are the most dangerous forking pieces.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        enemy_color = not color
        
        # Get enemy king and queen positions
        enemy_king = board.king(enemy_color)
        enemy_queens = board.pieces(chess.QUEEN, enemy_color)
        enemy_rooks = board.pieces(chess.ROOK, enemy_color)
        
        if enemy_king is None:
            continue
        
        # For each of our knights
        for knight_sq in board.pieces(chess.KNIGHT, color):
            # Check each square the knight can move to
            knight_moves = board.attacks(knight_sq)
            
            for target_sq in knight_moves:
                # Skip if occupied by our piece
                if board.color_at(target_sq) == color:
                    continue
                
                # What would this square attack?
                # Knights attack in L-shape from target square
                potential_attacks = chess.SquareSet(chess.BB_KNIGHT_ATTACKS[target_sq])
                
                attacks_king = enemy_king in potential_attacks
                attacks_queen = bool(enemy_queens & potential_attacks)
                attacks_rook = bool(enemy_rooks & potential_attacks)
                
                # Royal fork (king + queen)
                if attacks_king and attacks_queen:
                    score += sign * 100  # Huge threat
                # King + rook fork
                elif attacks_king and attacks_rook:
                    score += sign * 60
                # Queen + rook fork
                elif attacks_queen and attacks_rook:
                    score += sign * 40
    
    return score


def get_pawn_fork_score(board: chess.Board) -> int:
    """
    Detect pawn forks - when an enemy pawn attacks two of our pieces.
    This is critical because one piece WILL be lost (or a bad trade forced).
    
    We penalize the side whose pieces are forked heavily, as material loss is imminent.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        # We check if THIS color's pawns are forking enemy pieces
        sign = 1 if color == chess.WHITE else -1
        enemy_color = not color
        
        for pawn_sq in board.pieces(chess.PAWN, color):
            # Get squares this pawn attacks
            pawn_attacks = board.attacks(pawn_sq)
            
            # Find enemy pieces being attacked (not pawns - those aren't real forks)
            attacked_pieces = []
            for target_sq in pawn_attacks:
                piece = board.piece_at(target_sq)
                if piece and piece.color == enemy_color and piece.piece_type != chess.PAWN:
                    attacked_pieces.append((target_sq, piece.piece_type))
            
            # If pawn attacks 2+ pieces, it's a fork
            if len(attacked_pieces) >= 2:
                # Calculate the value of the pieces being attacked
                values = [PIECE_VALUES.get(pt, 0) for (sq, pt) in attacked_pieces]
                values.sort(reverse=True)  # Highest value first
                
                # The defender can save at most one piece
                # So they will likely lose material equal to the second-most valuable piece
                # This is a SEVERE penalty - essentially losing the piece value
                second_value = values[1] if len(values) > 1 else 0
                
                # Fork penalty is close to the full value of the piece being lost
                # since one piece WILL be lost (minus pawn value if they capture)
                if second_value > PIECE_VALUE_PAWN:
                    # Give 80% of the piece value as penalty (conservative)
                    fork_bonus = int(second_value * 0.8)
                    score += sign * fork_bonus
    
    return score


def get_tactical_score(board: chess.Board) -> int:
    """Combined tactical evaluation."""
    score = 0
    score += get_hanging_pieces_score(board)
    score += get_threats_score(board)
    score += get_pins_score(board)
    score += get_pawn_fork_score(board)  # Detect pawn forks
    return score
