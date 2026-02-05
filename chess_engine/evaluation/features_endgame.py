"""
Endgame-specific evaluation features:
- King centralization in endgame
- King opposition
- Rule of the square for passed pawns
- Pawn race calculation
- King activity and aggression
- Mating patterns
"""
import chess
from chess_engine.tuned_weights import (
    KING_CENTRALIZATION_BONUS,
    KING_OPPOSITION_BONUS,
    RULE_OF_SQUARE_BONUS,
)


# Center squares for king centralization (most valuable in endgame)
CENTER_SQUARES = {
    chess.D4, chess.E4, chess.D5, chess.E5,
}

INNER_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}

EXTENDED_CENTER = {
    chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2,
    chess.B3, chess.G3,
    chess.B4, chess.G4,
    chess.B5, chess.G5,
    chess.B6, chess.G6,
    chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7,
}

# Distance tables for king proximity to corners/edges
CORNER_DISTANCE = []
for sq in range(64):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    dist_corners = min(
        max(f, r),              # a1
        max(7-f, r),            # h1  
        max(f, 7-r),            # a8
        max(7-f, 7-r)           # h8
    )
    CORNER_DISTANCE.append(dist_corners)

CENTER_DISTANCE = []
for sq in range(64):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    # Manhattan distance to center
    dist = abs(f - 3.5) + abs(r - 3.5)
    CENTER_DISTANCE.append(int(dist))


def is_endgame(board: chess.Board) -> bool:
    """
    Determine if the position is an endgame.
    Endgame = no queens OR each side has at most queen + minor piece.
    """
    white_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    black_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    if white_queens == 0 and black_queens == 0:
        return True
    
    # Count material
    white_minors = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.WHITE))
    black_minors = len(board.pieces(chess.KNIGHT, chess.BLACK)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    white_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    black_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    
    white_material = white_queens * 9 + white_rooks * 5 + white_minors * 3
    black_material = black_queens * 9 + black_rooks * 5 + black_minors * 3
    
    # Endgame if total material is low
    return (white_material + black_material) <= 13


def get_endgame_score(board: chess.Board, phase: float) -> int:
    """
    Comprehensive endgame-specific evaluation.
    More aggressive scaling - even mid-late game positions benefit from endgame knowledge.
    """
    # Apply endgame knowledge earlier (phase < 0.7 instead of 0.5)
    if phase > 0.7:
        return 0
    
    score = 0
    
    # Scale by how much we're in the endgame (more aggressive scaling)
    endgame_factor = (0.7 - phase) / 0.7  # 0 at phase=0.7, 1 at phase=0
    
    # Core endgame features
    score += int(_king_centralization_score(board) * endgame_factor * 1.5)
    score += int(_king_proximity_score(board) * endgame_factor)
    score += int(_king_opposition_score(board) * endgame_factor)
    score += int(_rule_of_square_score(board) * endgame_factor * 2.5)  # Critical
    score += int(_passed_pawn_endgame_score(board) * endgame_factor * 2.0)  # Critical
    score += int(_mating_potential_score(board) * endgame_factor)
    score += int(_connected_passed_pawns_score(board) * endgame_factor * 1.5)
    score += int(_king_pawn_coordination_score(board) * endgame_factor)
    
    return score


def _connected_passed_pawns_score(board: chess.Board) -> int:
    """
    Connected passed pawns are extremely powerful in endgames.
    Two pawns supporting each other are often unstoppable.
    """
    score = 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Find white passed pawns
    white_passed = []
    for sq in white_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for check_r in range(r + 1, 8):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in black_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            white_passed.append((sq, f, r))
    
    # Check if any white passed pawns are connected (adjacent files)
    for i, (sq1, f1, r1) in enumerate(white_passed):
        for sq2, f2, r2 in white_passed[i+1:]:
            if abs(f1 - f2) == 1:  # Adjacent files
                # Bonus based on how advanced they are
                avg_rank = (r1 + r2) / 2
                score += int(30 + avg_rank * 15)
    
    # Find black passed pawns
    black_passed = []
    for sq in black_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for check_r in range(r - 1, -1, -1):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in white_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        if is_passed:
            black_passed.append((sq, f, r))
    
    # Check if any black passed pawns are connected
    for i, (sq1, f1, r1) in enumerate(black_passed):
        for sq2, f2, r2 in black_passed[i+1:]:
            if abs(f1 - f2) == 1:
                avg_rank = (r1 + r2) / 2
                score -= int(30 + (7 - avg_rank) * 15)
    
    return score


def _king_pawn_coordination_score(board: chess.Board) -> int:
    """
    Evaluate how well the king supports its pawns in the endgame.
    King should be in front of or beside passed pawns.
    """
    score = 0
    
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
    bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
    
    # White king supporting white pawns
    for sq in white_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # King distance to pawn
        dist = max(abs(wk_file - f), abs(wk_rank - r))
        
        # Bonus if king is close and in front of or beside the pawn
        if dist <= 2:
            if wk_rank >= r:  # King in front or same rank
                score += 15
            else:
                score += 8
    
    # Black king supporting black pawns
    for sq in black_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        dist = max(abs(bk_file - f), abs(bk_rank - r))
        
        if dist <= 2:
            if bk_rank <= r:  # King in front or same rank (for black, lower rank is forward)
                score -= 15
            else:
                score -= 8
    
    return score


def _king_centralization_score(board: chess.Board) -> int:
    """
    In endgames, kings should be centralized and active.
    More granular scoring based on exact position.
    """
    score = 0
    
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        
        if king_sq is None:
            continue
        
        # Bonus for center proximity
        if king_sq in CENTER_SQUARES:
            score += sign * KING_CENTRALIZATION_BONUS * 2
        elif king_sq in INNER_CENTER:
            score += sign * KING_CENTRALIZATION_BONUS
        elif king_sq in EXTENDED_CENTER:
            score += sign * (KING_CENTRALIZATION_BONUS // 2)
        
        # Additional penalty for being on edge/corner
        center_dist = CENTER_DISTANCE[king_sq]
        score -= sign * center_dist * 3
    
    return score


def _king_proximity_score(board: chess.Board) -> int:
    """
    In endgames, the winning side's king should approach the enemy king
    to help deliver mate or support pawns.
    """
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 0
    
    # Calculate king distance
    wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
    bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
    king_distance = max(abs(wk_file - bk_file), abs(wk_rank - bk_rank))
    
    # Material advantage determines who benefits from king proximity
    material_balance = _get_material_balance(board)
    
    if abs(material_balance) < 100:
        return 0  # Balanced - no bonus
    
    # Bonus for the winning side to have kings closer (helps mating)
    proximity_bonus = (8 - king_distance) * 5
    
    if material_balance > 0:
        return proximity_bonus  # White ahead, benefit from closeness
    else:
        return -proximity_bonus  # Black ahead


def _get_material_balance(board: chess.Board) -> int:
    """Get material balance (positive = white ahead)."""
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, 
              chess.ROOK: 500, chess.QUEEN: 900}
    balance = 0
    for pt, val in values.items():
        balance += val * len(board.pieces(pt, chess.WHITE))
        balance -= val * len(board.pieces(pt, chess.BLACK))
    return balance


def _king_opposition_score(board: chess.Board) -> int:
    """
    In king and pawn endgames, having the opposition is valuable.
    Opposition = kings are on same file or rank with one square between them,
    and the other side has to move.
    """
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 0
    
    wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
    bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
    
    # Check if kings are in opposition
    file_diff = abs(wk_file - bk_file)
    rank_diff = abs(wk_rank - bk_rank)
    
    # Direct opposition: same file/rank, 2 squares apart
    # Or diagonal opposition: 2 squares apart diagonally
    in_opposition = (
        (file_diff == 0 and rank_diff == 2) or  # Vertical opposition
        (file_diff == 2 and rank_diff == 0) or  # Horizontal opposition
        (file_diff == 2 and rank_diff == 2)     # Diagonal opposition
    )
    
    if in_opposition:
        # The side NOT to move has the opposition
        if board.turn == chess.WHITE:
            return -KING_OPPOSITION_BONUS  # Black has opposition
        else:
            return KING_OPPOSITION_BONUS   # White has opposition
    
    return 0


def _rule_of_square_score(board: chess.Board) -> int:
    """
    Rule of the square: can the enemy king catch a passed pawn?
    If not, the passed pawn is very dangerous.
    """
    score = 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    black_king = board.king(chess.BLACK)
    white_king = board.king(chess.WHITE)
    
    if black_king is None or white_king is None:
        return 0
    
    bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
    wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
    
    # Check white passed pawns
    for sq in white_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # Check if it's a passed pawn
        is_passed = True
        for check_r in range(r + 1, 8):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in black_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            # Distance pawn needs to travel to promote
            pawn_distance = 7 - r
            
            # Distance black king needs to travel to intercept
            # King needs to reach the promotion square or a square in front of pawn
            king_distance = max(abs(bk_file - f), abs(bk_rank - 7))
            
            # Adjust for whose turn it is
            if board.turn == chess.BLACK:
                king_distance -= 1  # Black can move first
            
            if pawn_distance < king_distance:
                # Pawn will promote!
                score += RULE_OF_SQUARE_BONUS
    
    # Check black passed pawns
    for sq in black_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        is_passed = True
        for check_r in range(r - 1, -1, -1):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in white_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            pawn_distance = r  # Distance to rank 0
            king_distance = max(abs(wk_file - f), abs(wk_rank - 0))
            
            if board.turn == chess.WHITE:
                king_distance -= 1
            
            if pawn_distance < king_distance:
                score -= RULE_OF_SQUARE_BONUS
    
    return score


def _passed_pawn_endgame_score(board: chess.Board) -> int:
    """
    Additional endgame-specific passed pawn bonuses.
    - Bonus for advanced passed pawns
    - Bonus for king supporting the pawn
    - Bonus for unstoppable pawns
    """
    score = 0
    
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 0
    
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)
    
    # Evaluate white passed pawns
    for sq in white_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # Check if passed
        is_passed = True
        for check_r in range(r + 1, 8):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in black_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            # Exponential bonus for advanced pawns
            rank_bonus = [0, 10, 15, 25, 45, 80, 150, 0][r]
            score += rank_bonus
            
            # Bonus for own king being close to pawn
            wk_dist = max(abs(chess.square_file(white_king) - f), 
                         abs(chess.square_rank(white_king) - r))
            if wk_dist <= 2:
                score += 20
            
            # Penalty for enemy king being close
            bk_dist = max(abs(chess.square_file(black_king) - f),
                         abs(chess.square_rank(black_king) - r))
            if bk_dist > 3:
                score += 15  # Enemy king too far
    
    # Evaluate black passed pawns
    for sq in black_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        is_passed = True
        for check_r in range(r - 1, -1, -1):
            for check_f in [f - 1, f, f + 1]:
                if 0 <= check_f <= 7:
                    if chess.square(check_f, check_r) in white_pawns:
                        is_passed = False
                        break
            if not is_passed:
                break
        
        if is_passed:
            rank_bonus = [0, 150, 80, 45, 25, 15, 10, 0][r]
            score -= rank_bonus
            
            bk_dist = max(abs(chess.square_file(black_king) - f),
                         abs(chess.square_rank(black_king) - r))
            if bk_dist <= 2:
                score -= 20
            
            wk_dist = max(abs(chess.square_file(white_king) - f),
                         abs(chess.square_rank(white_king) - r))
            if wk_dist > 3:
                score -= 15
    
    return score


def _mating_potential_score(board: chess.Board) -> int:
    """
    Evaluate positions with mating potential.
    - Drive enemy king to corner when up major material
    - Penalize positions where our king is in corner when down material
    """
    score = 0
    
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    if white_king is None or black_king is None:
        return 0
    
    material_balance = _get_material_balance(board)
    
    # If significantly ahead, drive enemy king to corner
    if material_balance > 300:  # White significantly ahead
        # Penalize black king being in center (we want it in corner)
        bk_corner_dist = CORNER_DISTANCE[black_king]
        score += (4 - bk_corner_dist) * 15  # Up to 60 cp for corner
        
        # Bonus for our king being close to enemy king
        wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
        bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
        king_dist = max(abs(wk_file - bk_file), abs(wk_rank - bk_rank))
        score += (8 - king_dist) * 8
        
    elif material_balance < -300:  # Black significantly ahead
        wk_corner_dist = CORNER_DISTANCE[white_king]
        score -= (4 - wk_corner_dist) * 15
        
        wk_file, wk_rank = chess.square_file(white_king), chess.square_rank(white_king)
        bk_file, bk_rank = chess.square_file(black_king), chess.square_rank(black_king)
        king_dist = max(abs(wk_file - bk_file), abs(wk_rank - bk_rank))
        score -= (8 - king_dist) * 8
    
    return score
