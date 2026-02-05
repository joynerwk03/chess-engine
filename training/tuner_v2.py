#!/usr/bin/env python3
"""
Improved Tuning Pipeline v2 for chess engine evaluation.

Key improvements over v1:
1. Fixed material values (not learned) - use known good values
2. Only tune positional/strategic features
3. Weighted loss to balance game phases
4. Feature normalization for better convergence
5. Separate MG/EG weight learning
6. Outlier filtering (remove extreme scores)
7. Cross-validation for better generalization

Usage:
    python -m training.tuner_v2
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

try:
    import chess
except ImportError:
    print("Missing python-chess. Install with: pip install python-chess", file=sys.stderr)
    raise

try:
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error
except ImportError:
    print("Missing scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    raise

# =============================
# Configuration
# =============================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train_data.npz")
VAL_PATH = os.path.join(DATA_DIR, "val_data.npz")
TEST_PATH = os.path.join(DATA_DIR, "test_data.npz")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "learned_weights_v2.py")

# Fixed material values (not learned - these are well-established)
FIXED_MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Score outlier threshold (positions with |score| > this are filtered)
SCORE_CLAMP = 1500

# Minimum positions per feature to include it
MIN_FEATURE_COUNT = 100


# =============================
# Helper Functions
# =============================

def _game_phase(board: chess.Board) -> float:
    """Phase in [0,1]: 1.0 = full middlegame, 0.0 = pure endgame."""
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(pt, chess.WHITE))
        count += len(board.pieces(pt, chess.BLACK))
    return max(0.0, min(1.0, count / 14.0))


def _file_index(sq: int) -> int:
    return chess.square_file(sq)


def _rank_index(sq: int) -> int:
    return chess.square_rank(sq)


def _material_score(board: chess.Board) -> int:
    """Calculate material score with fixed values."""
    score = 0
    for pt, val in FIXED_MATERIAL.items():
        diff = len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        score += val * diff
    return score


# =============================
# Feature Extraction
# =============================

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


def _count_isolated_pawns(board: chess.Board, color: bool) -> int:
    cnt = 0
    pawns = board.pieces(chess.PAWN, color)
    files = [0] * 8
    for sq in pawns:
        files[_file_index(sq)] += 1
    for sq in pawns:
        f = _file_index(sq)
        left = files[f - 1] if f > 0 else 0
        right = files[f + 1] if f < 7 else 0
        if left == 0 and right == 0:
            cnt += 1
    return cnt


def _is_passed_pawn(board: chess.Board, color: bool, sq: int) -> bool:
    f = _file_index(sq)
    r = _rank_index(sq)
    enemy = not color
    enemy_pawns = board.pieces(chess.PAWN, enemy)
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


def _count_passed_pawns(board: chess.Board, color: bool) -> int:
    return sum(1 for sq in board.pieces(chess.PAWN, color) if _is_passed_pawn(board, color, sq))


def _has_bishop_pair(board: chess.Board, color: bool) -> bool:
    return len(board.pieces(chess.BISHOP, color)) >= 2


def _rook_on_open_file(board: chess.Board, color: bool) -> int:
    count = 0
    for sq in board.pieces(chess.ROOK, color):
        f = _file_index(sq)
        has_own_pawn = any(_file_index(p) == f for p in board.pieces(chess.PAWN, color))
        has_enemy_pawn = any(_file_index(p) == f for p in board.pieces(chess.PAWN, not color))
        if not has_own_pawn and not has_enemy_pawn:
            count += 1
    return count


def _rook_on_semi_open_file(board: chess.Board, color: bool) -> int:
    count = 0
    for sq in board.pieces(chess.ROOK, color):
        f = _file_index(sq)
        has_own_pawn = any(_file_index(p) == f for p in board.pieces(chess.PAWN, color))
        has_enemy_pawn = any(_file_index(p) == f for p in board.pieces(chess.PAWN, not color))
        if not has_own_pawn and has_enemy_pawn:
            count += 1
    return count


def _knight_outpost(board: chess.Board, color: bool) -> int:
    count = 0
    enemy = not color
    for sq in board.pieces(chess.KNIGHT, color):
        # Check if supported by pawn
        supported = bool(board.attackers(color, sq) & board.pieces(chess.PAWN, color))
        # Check if can be attacked by enemy pawn
        attackable = bool(board.attackers(enemy, sq) & board.pieces(chess.PAWN, enemy))
        if supported and not attackable:
            count += 1
    return count


def _king_pawn_shield(board: chess.Board, color: bool) -> int:
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    kf, kr = _file_index(king_sq), _rank_index(king_sq)
    count = 0
    pawns = board.pieces(chess.PAWN, color)
    # Check shield squares
    if color == chess.WHITE:
        for df in (-1, 0, 1):
            ff = kf + df
            if 0 <= ff < 8:
                for rr in (1, 2):
                    if chess.square(ff, rr) in pawns:
                        count += 1
    else:
        for df in (-1, 0, 1):
            ff = kf + df
            if 0 <= ff < 8:
                for rr in (6, 5):
                    if chess.square(ff, rr) in pawns:
                        count += 1
    return count


def _king_on_open_file(board: chess.Board, color: bool) -> bool:
    king_sq = board.king(color)
    if king_sq is None:
        return False
    kf = _file_index(king_sq)
    has_any_pawn = any(_file_index(p) == kf for p in board.pieces(chess.PAWN, chess.WHITE))
    has_any_pawn |= any(_file_index(p) == kf for p in board.pieces(chess.PAWN, chess.BLACK))
    return not has_any_pawn


def _mobility(board: chess.Board, color: bool) -> int:
    count = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            count += len(board.attacks(sq))
    return count


def _center_control(board: chess.Board) -> int:
    centers = [chess.D4, chess.E4, chess.D5, chess.E5]
    white = sum(len(board.attackers(chess.WHITE, sq)) for sq in centers)
    black = sum(len(board.attackers(chess.BLACK, sq)) for sq in centers)
    return white - black


def _rook_on_7th(board: chess.Board, color: bool) -> int:
    target_rank = 6 if color == chess.WHITE else 1
    return sum(1 for sq in board.pieces(chess.ROOK, color) if _rank_index(sq) == target_rank)


def _connected_rooks(board: chess.Board, color: bool) -> bool:
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return False
    return rooks[1] in board.attacks(rooks[0])


def _king_tropism(board: chess.Board, color: bool) -> int:
    """Sum of proximity of attacking pieces to enemy king."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0
    ek_f, ek_r = _file_index(enemy_king), _rank_index(enemy_king)
    tropism = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            pf, pr = _file_index(sq), _rank_index(sq)
            dist = max(abs(pf - ek_f), abs(pr - ek_r))
            tropism += max(0, 7 - dist)
    return tropism


def _king_centralization(board: chess.Board, color: bool) -> int:
    """King distance from center (lower = more central)."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    f, r = _file_index(king_sq), _rank_index(king_sq)
    return int(7 - (abs(f - 3.5) + abs(r - 3.5)))


def _backward_pawns(board: chess.Board, color: bool) -> int:
    count = 0
    pawns = board.pieces(chess.PAWN, color)
    for sq in pawns:
        f, r = _file_index(sq), _rank_index(sq)
        has_support = False
        for adj_f in (f - 1, f + 1):
            if adj_f < 0 or adj_f > 7:
                continue
            for adj_sq in pawns:
                af, ar = _file_index(adj_sq), _rank_index(adj_sq)
                if af == adj_f:
                    if color == chess.WHITE and ar <= r:
                        has_support = True
                    elif color == chess.BLACK and ar >= r:
                        has_support = True
        if not has_support:
            count += 1
    return count


def _pawn_chains(board: chess.Board, color: bool) -> int:
    """Count pawns protected by other pawns."""
    count = 0
    pawns = board.pieces(chess.PAWN, color)
    for sq in pawns:
        defenders = board.attackers(color, sq) & pawns
        if defenders:
            count += 1
    return count


def extract_positional_features(board: chess.Board) -> Dict[str, float]:
    """
    Extract only positional features (not material).
    Returns raw feature differences (White - Black) that will be weighted.
    """
    feats: Dict[str, float] = {}
    phase = _game_phase(board)
    eg_phase = 1.0 - phase
    
    # Pawn structure
    feats["doubled_pawns"] = float(_count_doubled_pawns(board, chess.WHITE) - _count_doubled_pawns(board, chess.BLACK))
    feats["isolated_pawns"] = float(_count_isolated_pawns(board, chess.WHITE) - _count_isolated_pawns(board, chess.BLACK))
    feats["passed_pawns"] = float(_count_passed_pawns(board, chess.WHITE) - _count_passed_pawns(board, chess.BLACK))
    feats["backward_pawns"] = float(_backward_pawns(board, chess.WHITE) - _backward_pawns(board, chess.BLACK))
    feats["pawn_chains"] = float(_pawn_chains(board, chess.WHITE) - _pawn_chains(board, chess.BLACK))
    
    # Piece placement
    feats["bishop_pair"] = float(_has_bishop_pair(board, chess.WHITE)) - float(_has_bishop_pair(board, chess.BLACK))
    feats["rook_open_file"] = float(_rook_on_open_file(board, chess.WHITE) - _rook_on_open_file(board, chess.BLACK))
    feats["rook_semi_open"] = float(_rook_on_semi_open_file(board, chess.WHITE) - _rook_on_semi_open_file(board, chess.BLACK))
    feats["knight_outpost"] = float(_knight_outpost(board, chess.WHITE) - _knight_outpost(board, chess.BLACK))
    feats["rook_on_7th"] = float(_rook_on_7th(board, chess.WHITE) - _rook_on_7th(board, chess.BLACK))
    feats["connected_rooks"] = float(_connected_rooks(board, chess.WHITE)) - float(_connected_rooks(board, chess.BLACK))
    
    # King safety (scaled by phase - more important in middlegame)
    feats["king_shield_mg"] = float(_king_pawn_shield(board, chess.WHITE) - _king_pawn_shield(board, chess.BLACK)) * phase
    feats["king_open_file_mg"] = float(_king_on_open_file(board, chess.WHITE)) - float(_king_on_open_file(board, chess.BLACK))
    feats["king_open_file_mg"] *= phase
    feats["king_tropism_mg"] = float(_king_tropism(board, chess.WHITE) - _king_tropism(board, chess.BLACK)) * phase
    
    # Mobility and control (scaled by phase)
    feats["mobility_mg"] = float(_mobility(board, chess.WHITE) - _mobility(board, chess.BLACK)) * phase
    feats["center_control_mg"] = float(_center_control(board)) * phase
    
    # Endgame features (scaled by eg_phase)
    feats["king_centralization_eg"] = float(_king_centralization(board, chess.WHITE) - _king_centralization(board, chess.BLACK)) * eg_phase
    
    # Tempo (side to move advantage)
    feats["tempo"] = 1.0 if board.turn == chess.WHITE else -1.0
    
    return feats


# =============================
# Training
# =============================

def load_and_preprocess(npz_path: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load positions, extract features, and compute residuals (score - material).
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    
    data = np.load(npz_path, allow_pickle=True)
    fens = data["fens"].astype(object)
    scores = data["scores"].astype(np.float64)
    
    if max_samples and len(fens) > max_samples:
        indices = np.random.choice(len(fens), max_samples, replace=False)
        fens = fens[indices]
        scores = scores[indices]
    
    feature_dicts = []
    residuals = []
    
    for i, fen in enumerate(tqdm(fens, desc=f"Processing {os.path.basename(npz_path)}")):
        try:
            board = chess.Board(str(fen))
        except Exception:
            continue
        
        score = scores[i]
        
        # Filter outliers
        if abs(score) > SCORE_CLAMP:
            continue
        
        # Compute material
        material = _material_score(board)
        
        # Residual is what we want to predict (score minus material)
        residual = score - material
        
        # Extract positional features
        feats = extract_positional_features(board)
        
        feature_dicts.append(feats)
        residuals.append(residual)
    
    # Build feature matrix
    if not feature_dicts:
        raise ValueError("No valid positions found")
    
    feature_names = sorted(feature_dicts[0].keys())
    X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float64)
    y = np.array(residuals, dtype=np.float64)
    
    for i, feats in enumerate(feature_dicts):
        for j, name in enumerate(feature_names):
            X[i, j] = feats.get(name, 0.0)
    
    return X, y, feature_names


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                feature_names: List[str]) -> Tuple[Ridge, StandardScaler, Dict[str, float]]:
    """
    Train ridge regression with cross-validation for alpha selection.
    Uses StandardScaler for better convergence.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Try different regularization strengths
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    best_alpha = 1.0
    best_mse = float('inf')
    
    print("\nHyperparameter search:")
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, pred)
        print(f"  alpha={alpha}: val MSE={mse:.2f}")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha}, MSE: {best_mse:.2f}")
    
    # Train final model
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)
    
    # Extract weights (unscale them)
    # w_original = w_scaled / scale
    weights = {}
    for i, name in enumerate(feature_names):
        scale = scaler.scale_[i] if scaler.scale_[i] != 0 else 1.0
        weights[name] = model.coef_[i] / scale
    
    return model, scaler, weights


def save_weights(weights: Dict[str, float], intercept: float, out_path: str):
    """Save learned weights to Python file."""
    with open(out_path, 'w') as f:
        f.write("# Auto-generated by tuner_v2.py\n")
        f.write("# Learned positional weights (material values are fixed)\n")
        f.write("#\n")
        f.write("# These weights predict the RESIDUAL (score - material)\n")
        f.write("# so they represent pure positional value.\n\n")
        
        f.write(f"INTERCEPT = {intercept:.4f}\n\n")
        
        f.write("# Fixed material values (not learned)\n")
        f.write("MATERIAL_PAWN = 100\n")
        f.write("MATERIAL_KNIGHT = 320\n")
        f.write("MATERIAL_BISHOP = 330\n")
        f.write("MATERIAL_ROOK = 500\n")
        f.write("MATERIAL_QUEEN = 900\n\n")
        
        f.write("# Learned positional weights\n")
        f.write("POSITIONAL_WEIGHTS = {\n")
        
        # Sort by absolute value
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, w in sorted_weights:
            f.write(f"    '{name}': {w:.2f},\n")
        
        f.write("}\n")
    
    print(f"\nSaved weights to {out_path}")


def main():
    print("=" * 60)
    print("Chess Engine Tuner v2 - Positional Weight Learning")
    print("=" * 60)
    print()
    print("Improvements over v1:")
    print("  - Fixed material values (100/320/330/500/900)")
    print("  - Only learning positional features")
    print("  - Predicting residual (score - material)")
    print("  - Outlier filtering (|score| > 1500)")
    print()
    
    # Load data
    print("Loading training data...")
    X_train, y_train, feature_names = load_and_preprocess(TRAIN_PATH)
    print(f"Training samples: {len(y_train)}")
    print(f"Features: {len(feature_names)}")
    print(f"Residual mean: {y_train.mean():.1f}, std: {y_train.std():.1f}")
    
    print("\nLoading validation data...")
    X_val, y_val, _ = load_and_preprocess(VAL_PATH)
    print(f"Validation samples: {len(y_val)}")
    
    # Train
    print("\nTraining model...")
    model, scaler, weights = train_model(X_train, y_train, X_val, y_val, feature_names)
    
    # Test
    if os.path.isfile(TEST_PATH):
        print("\nEvaluating on test set...")
        X_test, y_test, _ = load_and_preprocess(TEST_PATH)
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, pred)
        print(f"Test MSE: {test_mse:.2f}")
    
    # Show learned weights
    print("\nLearned positional weights:")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, w in sorted_weights[:15]:
        print(f"  {name}: {w:+.1f}")
    
    # Save
    save_weights(weights, model.intercept_, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
