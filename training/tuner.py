#!/usr/bin/env python3
"""
Tuning pipeline for the chess engine evaluation function.

Stages:
1) Feature extraction from FEN positions (raw, unweighted counts).
2) Build design matrices from .npz datasets.
3) Ridge regression hyperparameter sweep on validation set.
4) Save learned weights to tuned_weights.py (feature -> weight mapping).

Note: This script expects train_data.npz, val_data.npz (and optionally test_data.npz)
created by generate_data.py with arrays: fens (object), scores (int16/int32).
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

try:
    import chess
except Exception as e:
    print("Missing dependency. Please install python-chess, numpy, tqdm, scikit-learn.", file=sys.stderr)
    raise

# sklearn imports
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
except Exception as e:
    print("Missing scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    raise

# =============================
# Configuration
# =============================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.npz")
VAL_PATH = os.path.join(DATA_DIR, "val_data.npz")
TEST_PATH = os.path.join(DATA_DIR, "test_data.npz")  # optional; evaluated if present
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "learned_weights.py")

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

# =============================
# Feature Extraction
# =============================

def _game_phase(board: chess.Board) -> float:
    """Approximate phase in [0,1]: 1.0 = full middlegame, 0.0 = pure endgame.
    Uses remaining non-pawn, non-king piece count normalized by initial count (14).
    """
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(pt, chess.WHITE))
        count += len(board.pieces(pt, chess.BLACK))
    return max(0.0, min(1.0, count / 14.0))


def _piece_letter(pt: int) -> str:
    return {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}[pt]


def _file_index(sq: int) -> int:
    return chess.square_file(sq)


def _rank_index(sq: int) -> int:
    return chess.square_rank(sq)


def _count_doubled_pawns(board: chess.Board, color: bool) -> int:
    total = 0
    pawns = board.pieces(chess.PAWN, color)
    files = [0] * 8
    for sq in pawns:
        files[_file_index(sq)] += 1
    for n in files:
        if n > 1:
            total += (n - 1)
    return total


def _is_isolated_pawn(board: chess.Board, color: bool, sq: int) -> bool:
    f = _file_index(sq)
    pawns = board.pieces(chess.PAWN, color)
    left_file = {p for p in pawns if _file_index(p) == f - 1}
    right_file = {p for p in pawns if _file_index(p) == f + 1}
    return len(left_file) == 0 and len(right_file) == 0


def _count_isolated_pawns(board: chess.Board, color: bool) -> int:
    cnt = 0
    for sq in board.pieces(chess.PAWN, color):
        if _is_isolated_pawn(board, color, sq):
            cnt += 1
    return cnt


def _is_passed_pawn(board: chess.Board, color: bool, sq: int) -> bool:
    f = _file_index(sq)
    r = _rank_index(sq)
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    enemy_pawns = board.pieces(chess.PAWN, enemy)
    # Squares ahead on same or adjacent files
    if color == chess.WHITE:
        ahead_ranks = range(r + 1, 8)
    else:
        ahead_ranks = range(r - 1, -1, -1)
    for fr in (f - 1, f, f + 1):
        if fr < 0 or fr > 7:
            continue
        for rr in ahead_ranks:
            sq2 = chess.square(fr, rr)
            if sq2 in enemy_pawns:
                return False
    return True


def _count_passed_pawns(board: chess.Board, color: bool) -> Tuple[int, int]:
    """Returns (passed_count, outside_on_a_or_h_count)."""
    passed = 0
    outside = 0
    for sq in board.pieces(chess.PAWN, color):
        if _is_passed_pawn(board, color, sq):
            passed += 1
            if _file_index(sq) in (0, 7):
                outside += 1
    return passed, outside


def _knight_outpost(board: chess.Board, color: bool, sq: int) -> bool:
    # Supported by own pawn and not attackable by enemy pawn
    own_pawn_support = bool(board.attackers(color, sq) & board.pieces(chess.PAWN, color))
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    enemy_pawn_attack = bool(board.attackers(enemy, sq) & board.pieces(chess.PAWN, enemy))
    return own_pawn_support and not enemy_pawn_attack


def _count_knight_outposts(board: chess.Board, color: bool) -> int:
    cnt = 0
    for sq in board.pieces(chess.KNIGHT, color):
        if _knight_outpost(board, color, sq):
            cnt += 1
    return cnt


def _king_square(board: chess.Board, color: bool) -> Optional[int]:
    k = board.king(color)
    return k


def _pawn_shield_count(board: chess.Board, color: bool) -> int:
    ksq = _king_square(board, color)
    if ksq is None:
        return 0
    f = _file_index(ksq)
    # Shield squares: files f-1..f+1, ranks in front (1-2 ranks ahead from king home side)
    targets: List[int] = []
    if color == chess.WHITE:
        ranks = [1, 2]  # relative ranks ahead from rank 0 (2 = 3rd rank)
        base_rank = 1  # from white perspective, pawns on 2nd and 3rd rank form a shield
        for df in (-1, 0, 1):
            ff = f + df
            if 0 <= ff < 8:
                for rr in (1, 2):
                    sq = chess.square(ff, rr)
                    targets.append(sq)
    else:
        for df in (-1, 0, 1):
            ff = f + df
            if 0 <= ff < 8:
                for rr in (6, 5):  # 7th and 6th ranks for black
                    sq = chess.square(ff, rr)
                    targets.append(sq)
    pawns = board.pieces(chess.PAWN, color)
    return sum(1 for t in targets if t in pawns)


def _file_has_pawn(board: chess.Board, color: Optional[bool], file_idx: int) -> bool:
    pawns_white = board.pieces(chess.PAWN, chess.WHITE)
    pawns_black = board.pieces(chess.PAWN, chess.BLACK)
    if color is None:
        # any color
        for r in range(8):
            sq = chess.square(file_idx, r)
            if sq in pawns_white or sq in pawns_black:
                return True
        return False
    else:
        pawns = pawns_white if color == chess.WHITE else pawns_black
        for r in range(8):
            if chess.square(file_idx, r) in pawns:
                return True
        return False


def _rook_open_file_counts(board: chess.Board, color: bool) -> Tuple[int, int]:
    open_cnt = 0
    semi_cnt = 0
    for sq in board.pieces(chess.ROOK, color):
        f = _file_index(sq)
        any_pawn = _file_has_pawn(board, None, f)
        own_pawn = _file_has_pawn(board, color, f)
        if not any_pawn:
            open_cnt += 1
        if not own_pawn:
            semi_cnt += 1
    return open_cnt, semi_cnt


def _king_file_open_semi(board: chess.Board, color: bool) -> Tuple[int, int]:
    ksq = _king_square(board, color)
    if ksq is None:
        return 0, 0
    f = _file_index(ksq)
    any_pawn = _file_has_pawn(board, None, f)
    own_pawn = _file_has_pawn(board, color, f)
    open_file = 0 if any_pawn else 1
    semi_open = 0 if own_pawn else 1
    return open_file, semi_open


def _king_ring_attackers(board: chess.Board, color: bool) -> int:
    ksq = _king_square(board, color)
    if ksq is None:
        return 0
    # King ring: 3x3 around king
    ring = {ksq}
    for d_sq in [
        chess.square(x, y)
        for x in range(max(0, _file_index(ksq) - 1), min(7, _file_index(ksq) + 1) + 1)
        for y in range(max(0, _rank_index(ksq) - 1), min(7, _rank_index(ksq) + 1) + 1)
    ]:
        ring.add(d_sq)
    enemy = chess.BLACK if color == chess.WHITE else chess.WHITE
    attackers = 0
    for sq in ring:
        attackers += len(board.attackers(enemy, sq))
    return attackers


def _mobility_attacks(board: chess.Board, color: bool) -> int:
    # Count controlled squares by N,B,R,Q
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            count += len(board.attacks(sq))
    return count


def _space_pawn_control(board: chess.Board, color: bool) -> int:
    # Count pawn-controlled squares in forward 3 ranks
    cnt = 0
    pawns = board.pieces(chess.PAWN, color)
    for sq in pawns:
        for target in board.attacks(sq):
            # White pawns attack up (higher rank); black down
            if color == chess.WHITE:
                if _rank_index(target) in (2, 3, 4):
                    cnt += 1
            else:
                if _rank_index(target) in (5, 4, 3):
                    cnt += 1
    return cnt


def _center_control(board: chess.Board) -> int:
    centers = [chess.D4, chess.E4, chess.D5, chess.E5]
    white = sum(len(board.attackers(chess.WHITE, sq)) for sq in centers)
    black = sum(len(board.attackers(chess.BLACK, sq)) for sq in centers)
    return white - black


def _uncastled(board: chess.Board) -> int:
    w = 1 if (board.has_kingside_castling_rights(chess.WHITE) or board.has_queenside_castling_rights(chess.WHITE)) else 0
    b = 1 if (board.has_kingside_castling_rights(chess.BLACK) or board.has_queenside_castling_rights(chess.BLACK)) else 0
    return w - b


def _tempo(board: chess.Board) -> int:
    return 1 if board.turn == chess.WHITE else -1


# =============================
# NEW FEATURE HELPERS
# =============================

def _is_hanging(board: chess.Board, sq: int, color: bool) -> bool:
    """Check if a piece is hanging (attacked but not defended)."""
    enemy = not color
    attackers = board.attackers(enemy, sq)
    defenders = board.attackers(color, sq)
    return len(attackers) > 0 and len(defenders) == 0


def _count_hanging_pieces(board: chess.Board, color: bool) -> int:
    """Count hanging pieces for a color."""
    count = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if _is_hanging(board, sq, color):
                count += 1
    return count


def _count_threats(board: chess.Board, color: bool) -> int:
    """Count pieces attacked by lower-value pieces (threats)."""
    enemy = not color
    piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    threat_count = 0
    
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            for target in board.attacks(sq):
                target_piece = board.piece_at(target)
                if target_piece and target_piece.color == enemy:
                    if piece_values.get(target_piece.piece_type, 0) > piece_values.get(pt, 0):
                        threat_count += 1
    return threat_count


def _is_rook_on_7th(board: chess.Board, sq: int, color: bool) -> bool:
    """Check if rook is on 7th rank."""
    rank = _rank_index(sq)
    return (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1)


def _count_rooks_on_7th(board: chess.Board, color: bool) -> int:
    """Count rooks on 7th rank."""
    return sum(1 for sq in board.pieces(chess.ROOK, color) if _is_rook_on_7th(board, sq, color))


def _are_rooks_connected(board: chess.Board, color: bool) -> bool:
    """Check if two rooks are connected (can see each other)."""
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return False
    r1, r2 = rooks[0], rooks[1]
    # Check if one rook attacks the other's square
    return r2 in board.attacks(r1)


def _is_bad_bishop(board: chess.Board, sq: int, color: bool) -> int:
    """Check if bishop is blocked by own pawns. Returns penalty level (0, 1, or 2)."""
    # Determine bishop square color
    file_idx = _file_index(sq)
    rank_idx = _rank_index(sq)
    is_light_square = (file_idx + rank_idx) % 2 == 1
    
    # Count own pawns on same color squares
    pawns_blocking = 0
    for pawn_sq in board.pieces(chess.PAWN, color):
        pf, pr = _file_index(pawn_sq), _rank_index(pawn_sq)
        pawn_is_light = (pf + pr) % 2 == 1
        if pawn_is_light == is_light_square:
            pawns_blocking += 1
    
    if pawns_blocking >= 4:
        return 2  # Severe bad bishop
    elif pawns_blocking >= 3:
        return 1  # Moderate bad bishop
    return 0


def _count_bad_bishops(board: chess.Board, color: bool) -> Tuple[int, int]:
    """Returns (severe_count, moderate_count) of bad bishops."""
    severe, moderate = 0, 0
    for sq in board.pieces(chess.BISHOP, color):
        level = _is_bad_bishop(board, sq, color)
        if level == 2:
            severe += 1
        elif level == 1:
            moderate += 1
    return severe, moderate


def _is_fianchetto(board: chess.Board, sq: int, color: bool) -> bool:
    """Check if bishop is fianchettoed (b2/g2 for White, b7/g7 for Black)."""
    if color == chess.WHITE:
        return sq in (chess.B2, chess.G2)
    else:
        return sq in (chess.B7, chess.G7)


def _count_fianchetto(board: chess.Board, color: bool) -> int:
    """Count fianchettoed bishops."""
    return sum(1 for sq in board.pieces(chess.BISHOP, color) if _is_fianchetto(board, sq, color))


def _is_trapped_piece(board: chess.Board, sq: int, piece_type: int, color: bool) -> bool:
    """Check if a piece has very limited mobility (potentially trapped)."""
    attacks = board.attacks(sq)
    safe_squares = 0
    enemy = not color
    
    for target in attacks:
        # Check if square is safe (not attacked or defended)
        if not board.attackers(enemy, target):
            safe_squares += 1
        elif board.attackers(color, target):
            safe_squares += 1
    
    # Piece is trapped if it has 0-1 safe squares
    return safe_squares <= 1


def _count_trapped_pieces(board: chess.Board, color: bool) -> int:
    """Count trapped minor pieces and rooks."""
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        for sq in board.pieces(pt, color):
            if _is_trapped_piece(board, sq, pt, color):
                count += 1
    return count


def _is_knight_on_rim(board: chess.Board, sq: int) -> bool:
    """Check if knight is on rim (a or h file, or rank 1 or 8)."""
    f, r = _file_index(sq), _rank_index(sq)
    return f == 0 or f == 7 or r == 0 or r == 7


def _count_knights_on_rim(board: chess.Board, color: bool) -> int:
    """Count knights on rim."""
    return sum(1 for sq in board.pieces(chess.KNIGHT, color) if _is_knight_on_rim(board, sq))


def _is_backward_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    """Check if pawn is backward (no friendly pawns to support its advance)."""
    f = _file_index(sq)
    r = _rank_index(sq)
    
    # Check adjacent files for friendly pawns that could support
    for adj_f in (f - 1, f + 1):
        if adj_f < 0 or adj_f > 7:
            continue
        for pawn_sq in board.pieces(chess.PAWN, color):
            pf, pr = _file_index(pawn_sq), _rank_index(pawn_sq)
            if pf == adj_f:
                # For white, supporting pawns should be on same or lower rank
                if color == chess.WHITE and pr <= r:
                    return False
                elif color == chess.BLACK and pr >= r:
                    return False
    return True


def _count_backward_pawns(board: chess.Board, color: bool) -> int:
    """Count backward pawns."""
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        if _is_backward_pawn(board, sq, color):
            count += 1
    return count


def _count_pawn_chains(board: chess.Board, color: bool) -> int:
    """Count pawns that are part of chains (protected by other pawns)."""
    count = 0
    for sq in board.pieces(chess.PAWN, color):
        # Check if protected by another pawn
        defenders = board.attackers(color, sq)
        if defenders & board.pieces(chess.PAWN, color):
            count += 1
    return count


def _count_undeveloped_pieces(board: chess.Board, color: bool) -> int:
    """Count minor pieces still on starting squares."""
    count = 0
    if color == chess.WHITE:
        start_squares = {chess.B1, chess.G1, chess.C1, chess.F1}
    else:
        start_squares = {chess.B8, chess.G8, chess.C8, chess.F8}
    
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if sq in start_squares:
                count += 1
    return count


def _is_early_queen_out(board: chess.Board, color: bool, phase: float) -> bool:
    """Check if queen is out early while minor pieces undeveloped."""
    if phase < 0.8:  # Not early game
        return False
    
    undeveloped = _count_undeveloped_pieces(board, color)
    if undeveloped < 2:
        return False
    
    queen_sqs = board.pieces(chess.QUEEN, color)
    if not queen_sqs:
        return False
    
    queen_sq = list(queen_sqs)[0]
    start_sq = chess.D1 if color == chess.WHITE else chess.D8
    
    return queen_sq != start_sq


def _king_centralization(board: chess.Board, color: bool) -> int:
    """Measure how centralized the king is (for endgame). Higher = more central."""
    ksq = _king_square(board, color)
    if ksq is None:
        return 0
    f, r = _file_index(ksq), _rank_index(ksq)
    # Distance from center (3.5, 3.5)
    center_dist = abs(f - 3.5) + abs(r - 3.5)
    # Invert so higher = better (max 7, min 0 at center)
    return int(7 - center_dist)


def _count_pawn_storm(board: chess.Board, color: bool) -> int:
    """Count advanced pawns near enemy king (pawn storm)."""
    enemy = not color
    enemy_king_sq = _king_square(board, enemy)
    if enemy_king_sq is None:
        return 0
    
    ek_file = _file_index(enemy_king_sq)
    count = 0
    
    for sq in board.pieces(chess.PAWN, color):
        pf, pr = _file_index(sq), _rank_index(sq)
        # Check if pawn is near enemy king's file
        if abs(pf - ek_file) <= 1:
            # For white, advanced means higher rank; for black, lower
            if color == chess.WHITE and pr >= 4:
                count += 1
            elif color == chess.BLACK and pr <= 3:
                count += 1
    return count


def _king_tropism(board: chess.Board, color: bool) -> int:
    """Sum of how close attacking pieces are to enemy king."""
    enemy = not color
    enemy_king_sq = _king_square(board, enemy)
    if enemy_king_sq is None:
        return 0
    
    ek_f, ek_r = _file_index(enemy_king_sq), _rank_index(enemy_king_sq)
    tropism = 0
    
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            pf, pr = _file_index(sq), _rank_index(sq)
            dist = max(abs(pf - ek_f), abs(pr - ek_r))  # Chebyshev distance
            tropism += max(0, 7 - dist)  # Closer = higher score
    
    return tropism


def extract_features(board: chess.Board) -> Dict[str, float]:
    """Extract raw feature counts from a position.

    - Material counts (piece count differences).
    - PST indicators per piece-square for MG and EG, scaled by phase and (1-phase).
    - Heuristic counts: bishop pair, rook open/semi-open, pawn structure, knights outpost,
      king safety components, mobility, space, center control, uncastled, tempo.
    """
    feats: Dict[str, float] = {}

    phase = _game_phase(board)  # MG weight
    eg_phase = 1.0 - phase      # EG weight

    # Material counts
    for pt, name in [
        (chess.PAWN, "PAWN"), (chess.KNIGHT, "KNIGHT"), (chess.BISHOP, "BISHOP"),
        (chess.ROOK, "ROOK"), (chess.QUEEN, "QUEEN"), (chess.KING, "KING")
    ]:
        diff = len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        feats[f"material_{name}"] = float(diff)

    # PST indicators (mirrored for black to match typical PST usage)
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        pletter = _piece_letter(pt)
        if piece.color == chess.WHITE:
            idx = sq
            sgn = 1.0
        else:
            idx = chess.square_mirror(sq)
            sgn = -1.0
        sq_name = chess.square_name(idx).upper()
        feats[f"PST_MG_{pletter}_{sq_name}"] = feats.get(f"PST_MG_{pletter}_{sq_name}", 0.0) + sgn * phase
        feats[f"PST_EG_{pletter}_{sq_name}"] = feats.get(f"PST_EG_{pletter}_{sq_name}", 0.0) + sgn * eg_phase

    # Bishop pair
    bp = (1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0) - (
        1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0
    )
    feats["bishop_pair"] = float(bp)

    # Rook placement
    w_open, w_semi = _rook_open_file_counts(board, chess.WHITE)
    b_open, b_semi = _rook_open_file_counts(board, chess.BLACK)
    feats["rooks_open_file"] = float(w_open - b_open)
    feats["rooks_semi_open_file"] = float(w_semi - b_semi)

    # Pawn structure
    w_doubled = _count_doubled_pawns(board, chess.WHITE)
    b_doubled = _count_doubled_pawns(board, chess.BLACK)
    feats["doubled_pawns"] = float(w_doubled - b_doubled)

    w_isolated = _count_isolated_pawns(board, chess.WHITE)
    b_isolated = _count_isolated_pawns(board, chess.BLACK)
    feats["isolated_pawns"] = float(w_isolated - b_isolated)

    w_passed, w_outside = _count_passed_pawns(board, chess.WHITE)
    b_passed, b_outside = _count_passed_pawns(board, chess.BLACK)
    feats["passed_pawns"] = float(w_passed - b_passed)
    feats["outside_passed_pawns"] = float(w_outside - b_outside)

    # Knight outposts
    feats["knight_outposts"] = float(_count_knight_outposts(board, chess.WHITE) - _count_knight_outposts(board, chess.BLACK))

    # King safety components
    feats["king_pawn_shield"] = float(_pawn_shield_count(board, chess.WHITE) - _pawn_shield_count(board, chess.BLACK))
    w_open_file, w_semi_file = _king_file_open_semi(board, chess.WHITE)
    b_open_file, b_semi_file = _king_file_open_semi(board, chess.BLACK)
    feats["king_open_file"] = float(w_open_file - b_open_file)
    feats["king_semi_open_file"] = float(w_semi_file - b_semi_file)
    feats["king_ring_attackers"] = float(_king_ring_attackers(board, chess.WHITE) - _king_ring_attackers(board, chess.BLACK))

    # Mobility and Space
    feats["mobility"] = float(_mobility_attacks(board, chess.WHITE) - _mobility_attacks(board, chess.BLACK))
    feats["space"] = float(_space_pawn_control(board, chess.WHITE) - _space_pawn_control(board, chess.BLACK))

    # Center control
    feats["center_control"] = float(_center_control(board))

    # Uncastled and Tempo
    feats["uncastled"] = float(_uncastled(board))
    feats["tempo"] = float(_tempo(board))

    # =============================
    # NEW FEATURES
    # =============================
    
    # Tactical features
    feats["hanging_pieces"] = float(_count_hanging_pieces(board, chess.WHITE) - _count_hanging_pieces(board, chess.BLACK))
    feats["threats"] = float(_count_threats(board, chess.WHITE) - _count_threats(board, chess.BLACK))
    
    # Rook features
    feats["rooks_on_7th"] = float(_count_rooks_on_7th(board, chess.WHITE) - _count_rooks_on_7th(board, chess.BLACK))
    feats["connected_rooks"] = float(
        (1 if _are_rooks_connected(board, chess.WHITE) else 0) - 
        (1 if _are_rooks_connected(board, chess.BLACK) else 0)
    )
    
    # Bishop features
    w_severe, w_moderate = _count_bad_bishops(board, chess.WHITE)
    b_severe, b_moderate = _count_bad_bishops(board, chess.BLACK)
    feats["bad_bishop_severe"] = float(w_severe - b_severe)
    feats["bad_bishop_moderate"] = float(w_moderate - b_moderate)
    feats["fianchetto"] = float(_count_fianchetto(board, chess.WHITE) - _count_fianchetto(board, chess.BLACK))
    
    # Piece activity
    feats["trapped_pieces"] = float(_count_trapped_pieces(board, chess.WHITE) - _count_trapped_pieces(board, chess.BLACK))
    feats["knights_on_rim"] = float(_count_knights_on_rim(board, chess.WHITE) - _count_knights_on_rim(board, chess.BLACK))
    
    # Enhanced pawn features
    feats["backward_pawns"] = float(_count_backward_pawns(board, chess.WHITE) - _count_backward_pawns(board, chess.BLACK))
    feats["pawn_chains"] = float(_count_pawn_chains(board, chess.WHITE) - _count_pawn_chains(board, chess.BLACK))
    
    # Development (scaled by phase - only relevant in opening)
    feats["undeveloped_pieces"] = float(
        (_count_undeveloped_pieces(board, chess.WHITE) - _count_undeveloped_pieces(board, chess.BLACK)) * phase
    )
    feats["early_queen_out"] = float(
        ((1 if _is_early_queen_out(board, chess.WHITE, phase) else 0) - 
         (1 if _is_early_queen_out(board, chess.BLACK, phase) else 0)) * phase
    )
    
    # King safety extensions
    feats["pawn_storm"] = float(_count_pawn_storm(board, chess.WHITE) - _count_pawn_storm(board, chess.BLACK))
    feats["king_tropism"] = float(_king_tropism(board, chess.WHITE) - _king_tropism(board, chess.BLACK))
    
    # Endgame features (scaled by eg_phase - only relevant in endgame)
    feats["king_centralization"] = float(
        (_king_centralization(board, chess.WHITE) - _king_centralization(board, chess.BLACK)) * eg_phase
    )

    return feats


# =============================
# Dataset -> Design Matrix
# =============================

def _dict_to_vector(d: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    v = np.zeros(len(feature_names), dtype=np.float64)
    for i, name in enumerate(feature_names):
        v[i] = d.get(name, 0.0)
    return v


def create_training_matrix(npz_path: str, feature_names: Optional[List[str]] = None, allow_extend: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a .npz file and build X, y.

    If allow_extend is True and feature_names is provided, unseen features will be appended
    to the list and returned. If False, unknown features are ignored and only the provided
    feature_names are used to construct X.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    fens = data["fens"].astype(object)
    scores = data["scores"].astype(np.float64)

    # Build/extend feature name list
    if feature_names is None:
        feature_names = []

    feature_dicts: List[Dict[str, float]] = []

    for fen in tqdm(fens, desc=f"Extracting features from {os.path.basename(npz_path)}", unit="pos"):
        board = chess.Board(fen)
        feats = extract_features(board)
        if allow_extend:
            # Extend feature_names with any unseen features
            for k in feats.keys():
                if k not in feature_names:
                    feature_names.append(k)
        feature_dicts.append(feats)

    # Build matrix strictly in the order of feature_names
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
    for i, feats in enumerate(feature_dicts):
        X[i, :] = _dict_to_vector(feats, feature_names)

    y = scores
    return X, y, feature_names


# =============================
# Tuning
# =============================

def tune_weights(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Ridge, float]:
    best_model: Optional[Ridge] = None
    best_mse = float("inf")
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mse = mean_squared_error(y_val, pred)
        print(f"alpha={alpha}: val MSE={mse:.2f}")
        if mse < best_mse:
            best_mse = mse
            best_model = model
    assert best_model is not None
    print(f"Best alpha selected. Val MSE={best_mse:.2f}")
    return best_model, best_mse


# =============================
# Saving Results
# =============================

def save_weights(model: Ridge, feature_names: List[str], out_path: str = "temp_weights.py") -> None:
    """Write learned weights to a Python module mapping each feature to a weight.

    WARNING: This will overwrite the file at out_path if it exists.
    """
    coef = model.coef_.astype(float)
    intercept = float(model.intercept_)

    lines: List[str] = []
    lines.append("# Auto-generated by tuner.py\n")
    lines.append("# Learned linear model weights for evaluation features\n\n")
    lines.append(f"INTERCEPT = {intercept:.8f}\n\n")
    lines.append("FEATURE_WEIGHTS = {\n")
    for name, w in zip(feature_names, coef.tolist()):
        # Escape quotes in feature names if any
        safe_name = name.replace('"', '\\"').replace("'", "\\'")
        lines.append(f"    '{safe_name}': {w:.8f},\n")
    lines.append("}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved learned weights to {out_path}")


# =============================
# Orchestrator
# =============================

def main() -> None:
    print("Loading training data...")
    X_train, y_train, feat_names = create_training_matrix(TRAIN_PATH, allow_extend=True)

    print("Loading validation data...")
    X_val, y_val, feat_names = create_training_matrix(VAL_PATH, feature_names=feat_names, allow_extend=True)

    # Validation may introduce new features; rebuild training matrix to align columns
    if X_train.shape[1] != len(feat_names):
        print(
            f"Feature set expanded to {len(feat_names)} after validation. Rebuilding training matrix to align..."
        )
        X_train, y_train, feat_names = create_training_matrix(TRAIN_PATH, feature_names=feat_names, allow_extend=False)

    print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")

    model, best_mse = tune_weights(X_train, y_train, X_val, y_val)

    # Optional test evaluation
    if os.path.isfile(TEST_PATH):
        print("Loading test data...")
        # Do not allow test to alter the learned feature space
        X_test, y_test, _ = create_training_matrix(TEST_PATH, feature_names=feat_names, allow_extend=False)
        test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_pred)
        print(f"Test MSE = {test_mse:.2f}")

    # Save learned weights
    save_weights(model, feat_names, out_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
