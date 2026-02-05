import chess
from chess_engine.tuned_weights import (
    DOUBLED_PAWN_PENALTY,
    ISOLATED_PAWN_PENALTY,
    PASSED_PAWN_BONUS,
    OUTSIDE_PASSED_PAWN_BONUS,
    BACKWARD_PAWN_PENALTY,
    PAWN_CHAIN_BONUS,
    PAWN_MAJORITY_BONUS,
    PASSED_PAWN_BLOCKER_BONUS,
    UNBLOCKED_PASSED_PAWN_PENALTY,
)

# Use SquareSets for file masks to enable set ops and len()
FILE_MASKS = [
    chess.SquareSet(chess.BB_FILE_A),
    chess.SquareSet(chess.BB_FILE_B),
    chess.SquareSet(chess.BB_FILE_C),
    chess.SquareSet(chess.BB_FILE_D),
    chess.SquareSet(chess.BB_FILE_E),
    chess.SquareSet(chess.BB_FILE_F),
    chess.SquareSet(chess.BB_FILE_G),
    chess.SquareSet(chess.BB_FILE_H),
]


def _doubled_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    for f in range(8):
        mask = FILE_MASKS[f]
        wc = len(wp & mask)
        bc = len(bp & mask)
        if wc > 1:
            score += (wc - 1) * DOUBLED_PAWN_PENALTY
        if bc > 1:
            score -= (bc - 1) * DOUBLED_PAWN_PENALTY
    return score


def _isolated_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)

    # White isolated
    for sq in wp:
        f = chess.square_file(sq)
        left_mask = FILE_MASKS[f - 1] if f > 0 else chess.SquareSet(0)
        right_mask = FILE_MASKS[f + 1] if f < 7 else chess.SquareSet(0)
        if len(wp & (left_mask | right_mask)) == 0:
            score += ISOLATED_PAWN_PENALTY

    # Black isolated
    for sq in bp:
        f = chess.square_file(sq)
        left_mask = FILE_MASKS[f - 1] if f > 0 else chess.SquareSet(0)
        right_mask = FILE_MASKS[f + 1] if f < 7 else chess.SquareSet(0)
        if len(bp & (left_mask | right_mask)) == 0:
            score -= ISOLATED_PAWN_PENALTY

    return score


def _forward_mask_white(file_idx: int, rank_idx: int) -> chess.SquareSet:
    bb = 0
    for ff in (file_idx - 1, file_idx, file_idx + 1):
        if 0 <= ff <= 7:
            for r in range(rank_idx + 1, 8):
                bb |= 1 << chess.square(ff, r)
    return chess.SquareSet(bb)


def _forward_mask_black(file_idx: int, rank_idx: int) -> chess.SquareSet:
    bb = 0
    for ff in (file_idx - 1, file_idx, file_idx + 1):
        if 0 <= ff <= 7:
            for r in range(rank_idx - 1, -1, -1):
                bb |= 1 << chess.square(ff, r)
    return chess.SquareSet(bb)


def _passed_pawns_score(board: chess.Board) -> int:
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)

    # White passed pawns
    for sq in wp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        mask = _forward_mask_white(f, r)
        if len(bp & mask) == 0:
            score += PASSED_PAWN_BONUS[r]
            if f in (0, 7):  # a- or h-file
                score += OUTSIDE_PASSED_PAWN_BONUS

    # Black passed pawns
    for sq in bp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        mask = _forward_mask_black(f, r)
        if len(wp & mask) == 0:
            score -= PASSED_PAWN_BONUS[7 - r]
            if f in (0, 7):
                score -= OUTSIDE_PASSED_PAWN_BONUS

    return score


def get_pawn_structure_score(board: chess.Board) -> int:
    """White-centric pawn structure evaluation.
    Negative penalties reduce the side's score; bonuses increase it.
    """
    score = 0
    score += _doubled_pawns_score(board)
    score += _isolated_pawns_score(board)
    score += _passed_pawns_score(board)
    score += _backward_pawns_score(board)
    score += _pawn_chain_score(board)
    score += _pawn_majority_score(board)
    score += _passed_pawn_blocker_score(board)
    return score


def _backward_pawns_score(board: chess.Board) -> int:
    """
    Penalize backward pawns - pawns that cannot be defended by other pawns
    and are on semi-open files (can be attacked by enemy rooks).
    """
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    
    # White backward pawns
    for sq in wp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # Check if any friendly pawns can defend this pawn
        can_be_defended = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                # Check if there's a friendly pawn behind that could advance to defend
                for check_r in range(r - 1, -1, -1):
                    if chess.square(adj_f, check_r) in wp:
                        can_be_defended = True
                        break
            if can_be_defended:
                break
        
        if not can_be_defended:
            # Check if on semi-open file (no enemy pawn in front)
            is_semi_open = True
            for check_r in range(r + 1, 8):
                if chess.square(f, check_r) in bp:
                    is_semi_open = False
                    break
            
            if is_semi_open:
                score += BACKWARD_PAWN_PENALTY
    
    # Black backward pawns
    for sq in bp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        can_be_defended = False
        for adj_f in [f - 1, f + 1]:
            if 0 <= adj_f <= 7:
                for check_r in range(r + 1, 8):
                    if chess.square(adj_f, check_r) in bp:
                        can_be_defended = True
                        break
            if can_be_defended:
                break
        
        if not can_be_defended:
            is_semi_open = True
            for check_r in range(r - 1, -1, -1):
                if chess.square(f, check_r) in wp:
                    is_semi_open = False
                    break
            
            if is_semi_open:
                score -= BACKWARD_PAWN_PENALTY
    
    return score


def _pawn_chain_score(board: chess.Board) -> int:
    """
    Bonus for pawns in a chain (pawns that defend each other diagonally).
    """
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    
    # White pawn chains
    for sq in wp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        # Check if this pawn is defended by another pawn
        if r > 0:
            if f > 0 and chess.square(f - 1, r - 1) in wp:
                score += PAWN_CHAIN_BONUS
            if f < 7 and chess.square(f + 1, r - 1) in wp:
                score += PAWN_CHAIN_BONUS
    
    # Black pawn chains
    for sq in bp:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        if r < 7:
            if f > 0 and chess.square(f - 1, r + 1) in bp:
                score -= PAWN_CHAIN_BONUS
            if f < 7 and chess.square(f + 1, r + 1) in bp:
                score -= PAWN_CHAIN_BONUS
    
    return score


def _pawn_majority_score(board: chess.Board) -> int:
    """
    Bonus for having a pawn majority on one side of the board.
    Especially valuable for creating passed pawns.
    """
    score = 0
    wp = board.pieces(chess.PAWN, chess.WHITE)
    bp = board.pieces(chess.PAWN, chess.BLACK)
    
    # Queenside (files a-c)
    white_qs = sum(1 for sq in wp if chess.square_file(sq) <= 2)
    black_qs = sum(1 for sq in bp if chess.square_file(sq) <= 2)
    
    # Kingside (files f-h)
    white_ks = sum(1 for sq in wp if chess.square_file(sq) >= 5)
    black_ks = sum(1 for sq in bp if chess.square_file(sq) >= 5)
    
    # Award majority bonus
    if white_qs > black_qs:
        score += PAWN_MAJORITY_BONUS
    elif black_qs > white_qs:
        score -= PAWN_MAJORITY_BONUS
    
    if white_ks > black_ks:
        score += PAWN_MAJORITY_BONUS
    elif black_ks > white_ks:
        score -= PAWN_MAJORITY_BONUS
    
    return score


def _get_passed_pawns(board: chess.Board, color: bool) -> list[int]:
    """Get list of passed pawn squares for a color."""
    passed = []
    pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        
        if color == chess.WHITE:
            mask = _forward_mask_white(f, r)
        else:
            mask = _forward_mask_black(f, r)
        
        if len(enemy_pawns & mask) == 0:
            passed.append(sq)
    
    return passed


def _passed_pawn_blocker_score(board: chess.Board) -> int:
    """
    Bonus for blocking enemy passed pawns.
    Penalty for having unblocked passed pawns coming toward you.
    """
    score = 0
    
    # Find white's passed pawns and check if blocked
    white_passers = _get_passed_pawns(board, chess.WHITE)
    for passer_sq in white_passers:
        f = chess.square_file(passer_sq)
        r = chess.square_rank(passer_sq)
        
        # Check the square directly in front
        if r < 7:
            blocker_sq = chess.square(f, r + 1)
            blocker = board.piece_at(blocker_sq)
            
            if blocker and blocker.color == chess.BLACK:
                score -= PASSED_PAWN_BLOCKER_BONUS  # Black gets credit for blocking
            elif not blocker:
                score -= UNBLOCKED_PASSED_PAWN_PENALTY  # Black penalized for not blocking
    
    # Find black's passed pawns and check if blocked
    black_passers = _get_passed_pawns(board, chess.BLACK)
    for passer_sq in black_passers:
        f = chess.square_file(passer_sq)
        r = chess.square_rank(passer_sq)
        
        if r > 0:
            blocker_sq = chess.square(f, r - 1)
            blocker = board.piece_at(blocker_sq)
            
            if blocker and blocker.color == chess.WHITE:
                score += PASSED_PAWN_BLOCKER_BONUS  # White gets credit for blocking
            elif not blocker:
                score += UNBLOCKED_PASSED_PAWN_PENALTY  # White penalized for not blocking
    
    return score
