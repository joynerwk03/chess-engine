import chess
from chess_engine.tuned_weights import (
    KING_SHIELD_BONUS,
    KING_OPEN_FILE_PENALTY,
    KING_ATTACK_WEIGHT,
    PAWN_STORM_BONUS,
    QUEEN_TROPISM_BONUS,
    ROOK_TROPISM_BONUS,
    MINOR_TROPISM_BONUS,
    KING_ESCAPE_SQUARES_BONUS,
    KING_NO_ESCAPE_PENALTY,
)

# Back rank weakness penalty (when enemy can check on back rank)
BACK_RANK_WEAKNESS_PENALTY = 150

# Penalty for advancing kingside pawns before castling
# This creates permanent weaknesses
KINGSIDE_PAWN_ADVANCE_PENALTY = 30

# Penalty for king in center without castling rights (exposed king)
KING_IN_CENTER_NO_CASTLING_PENALTY = 80

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
        score_part += _pawn_storm_score(board, color, king_sq)
        score_part += _king_tropism_score(board, color)
        score_part += _virtual_king_mobility_score(board, color, king_sq)
        score_part -= _back_rank_weakness(board, color, king_sq)
        score_part -= _kingside_pawn_advance_penalty(board, color)
        score_part -= _king_in_center_penalty(board, color, king_sq)
        score += score_part if color == chess.WHITE else -score_part
    return score


def _king_in_center_penalty(board: chess.Board, color: bool, king_sq: int) -> int:
    """
    Penalize having the king in or near the center (d/e files) when:
    1. No castling rights remain, AND
    2. It's still the middlegame (queens/major pieces on board)
    
    This discourages moving the king early and losing castling options.
    """
    king_file = chess.square_file(king_sq)
    home_rank = 0 if color == chess.WHITE else 7
    king_rank = chess.square_rank(king_sq)
    
    # Only penalize if king is in the center or nearby (c-f files)
    if king_file < 2 or king_file > 5:
        return 0  # King is on a/b or g/h files, likely castled
    
    # Check if we have castling rights
    if color == chess.WHITE:
        has_castling = board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE)
    else:
        has_castling = board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)
    
    if has_castling:
        return 0  # Still have castling options, no penalty
    
    # Check if it's still middlegame (queens and rooks present)
    enemy = not color
    enemy_queens = len(board.pieces(chess.QUEEN, enemy))
    enemy_rooks = len(board.pieces(chess.ROOK, enemy))
    
    if enemy_queens == 0 and enemy_rooks <= 1:
        return 0  # Endgame, center king is OK
    
    # King is in center with no castling rights and enemy has attacking pieces
    penalty = KING_IN_CENTER_NO_CASTLING_PENALTY
    
    # Extra penalty if king has moved off home rank (truly exposed)
    if king_rank != home_rank:
        penalty += KING_IN_CENTER_NO_CASTLING_PENALTY // 2
    
    return penalty


def _back_rank_weakness(board: chess.Board, color: bool, king_sq: int) -> int:
    """
    Detect back rank weakness: when king is on back rank and
    enemy queen/rook can give check or invade the back rank.
    """
    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    back_rank = 0 if color == chess.WHITE else 7
    
    # Only applies if king is on back rank
    if king_rank != back_rank:
        return 0
    
    enemy = not color
    back_rank_bb = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    
    # Check if enemy heavy pieces (queen/rook) attack our back rank
    threat = 0
    for piece_type in [chess.QUEEN, chess.ROOK]:
        for sq in board.pieces(piece_type, enemy):
            attacks = board.attacks(sq)
            # If they attack any square on our back rank near the king
            for target_sq in attacks:
                if chess.square_rank(target_sq) == back_rank:
                    # Extra dangerous if adjacent to king
                    target_file = chess.square_file(target_sq)
                    if abs(target_file - king_file) <= 1:
                        # Direct threat to king area
                        if piece_type == chess.QUEEN:
                            threat += BACK_RANK_WEAKNESS_PENALTY
                        else:
                            threat += BACK_RANK_WEAKNESS_PENALTY // 2
                        break  # Only count once per piece
    
    return threat


def _pawn_storm_score(board: chess.Board, color: bool, enemy_king_sq: int) -> int:
    """
    Bonus for advancing pawns toward the enemy king (pawn storm).
    """
    enemy_king_file = chess.square_file(enemy_king_sq)
    enemy_king_rank = chess.square_rank(enemy_king_sq)
    
    score = 0
    our_pawns = board.pieces(chess.PAWN, color)
    
    # Only consider pawns near the enemy king's file
    storm_files = [enemy_king_file - 1, enemy_king_file, enemy_king_file + 1]
    
    for sq in our_pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        if f in storm_files and 0 <= f <= 7:
            if color == chess.WHITE:
                # Pawn advancing toward black king (higher rank = closer)
                if r >= 4:  # Pawn is advanced
                    advancement = r - 3  # How far past rank 4
                    score += PAWN_STORM_BONUS * advancement
            else:
                # Pawn advancing toward white king (lower rank = closer)
                if r <= 3:
                    advancement = 4 - r
                    score += PAWN_STORM_BONUS * advancement
    
    return score


def _king_tropism_score(board: chess.Board, color: bool) -> int:
    """
    Bonus for pieces close to the enemy king.
    More aggressive pieces (queen, rooks) get higher bonuses.
    """
    enemy_color = not color
    enemy_king_sq = board.king(enemy_color)
    if enemy_king_sq is None:
        return 0
    
    enemy_king_file = chess.square_file(enemy_king_sq)
    enemy_king_rank = chess.square_rank(enemy_king_sq)
    
    score = 0
    
    # Queens
    for sq in board.pieces(chess.QUEEN, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        distance = max(abs(f - enemy_king_file), abs(r - enemy_king_rank))
        # Closer = better (distance 1-7, we want low distance to give high bonus)
        bonus = (8 - distance) * QUEEN_TROPISM_BONUS
        score += bonus
    
    # Rooks
    for sq in board.pieces(chess.ROOK, color):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        distance = max(abs(f - enemy_king_file), abs(r - enemy_king_rank))
        bonus = (8 - distance) * ROOK_TROPISM_BONUS
        score += bonus
    
    # Minor pieces (knights, bishops)
    for piece_type in [chess.KNIGHT, chess.BISHOP]:
        for sq in board.pieces(piece_type, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            distance = max(abs(f - enemy_king_file), abs(r - enemy_king_rank))
            bonus = (8 - distance) * MINOR_TROPISM_BONUS
            score += bonus
    
    return score


def _virtual_king_mobility_score(board: chess.Board, color: bool, king_sq: int) -> int:
    """
    Evaluate how many safe escape squares the king has.
    Penalize kings with no escape (potential mating nets).
    """
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    enemy_color = not color
    
    safe_squares = 0
    
    # Check all adjacent squares
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            
            nf, nr = king_file + df, king_rank + dr
            if not (0 <= nf <= 7 and 0 <= nr <= 7):
                continue
            
            target_sq = chess.square(nf, nr)
            
            # Check if square is occupied by own piece
            piece_at_target = board.piece_at(target_sq)
            if piece_at_target and piece_at_target.color == color:
                continue
            
            # Check if square is attacked by enemy
            if board.is_attacked_by(enemy_color, target_sq):
                continue
            
            safe_squares += 1
    
    if safe_squares <= 1:
        return KING_NO_ESCAPE_PENALTY
    else:
        return safe_squares * KING_ESCAPE_SQUARES_BONUS


def _kingside_pawn_advance_penalty(board: chess.Board, color: bool) -> int:
    """
    Penalize advancing kingside pawns (f, g, h) when:
    1. The king hasn't castled yet, OR
    2. The king is on the kingside (after castling)
    
    This discourages weakening the kingside pawn shield before 
    it's needed or when it IS needed.
    """
    penalty = 0
    home_rank = 0 if color == chess.WHITE else 7
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    
    # Determine if king has castled kingside
    # King on g1/g8 with rook on f1/f8 is typical kingside castle
    kingside_castled = (king_file >= 5 and king_rank == home_rank)
    
    # Check if king is still in center (hasn't castled anywhere)
    king_in_center = (3 <= king_file <= 4 and king_rank == home_rank)
    
    pawns = board.pieces(chess.PAWN, color)
    pawn_home_rank = 1 if color == chess.WHITE else 6
    
    # Check f, g, h pawns
    for file_idx, file_name in [(5, 'f'), (6, 'g'), (7, 'h')]:
        home_sq = chess.square(file_idx, pawn_home_rank)
        
        # Find our pawn on this file (if any)
        pawn_on_file = None
        for sq in pawns:
            if chess.square_file(sq) == file_idx:
                pawn_on_file = sq
                break
        
        if pawn_on_file is None:
            # Pawn was captured or never existed - not our concern here
            continue
        
        pawn_rank = chess.square_rank(pawn_on_file)
        
        # Has this pawn advanced from its home rank?
        if color == chess.WHITE:
            advanced_squares = pawn_rank - pawn_home_rank
        else:
            advanced_squares = pawn_home_rank - pawn_rank
        
        if advanced_squares <= 0:
            continue  # Pawn hasn't advanced
        
        # Penalize based on situation:
        # 1. King in center - advancing these pawns weakens future castling
        # 2. King on kingside - advancing weakens the pawn shield
        if king_in_center or kingside_castled:
            # g-pawn and h-pawn advances are especially bad
            if file_idx == 6:  # g-pawn
                penalty += KINGSIDE_PAWN_ADVANCE_PENALTY * advanced_squares * 2
            elif file_idx == 7:  # h-pawn
                penalty += KINGSIDE_PAWN_ADVANCE_PENALTY * advanced_squares
            else:  # f-pawn
                penalty += KINGSIDE_PAWN_ADVANCE_PENALTY * advanced_squares
    
    return penalty
