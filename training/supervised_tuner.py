#!/usr/bin/env python3
"""
Supervised Weight Tuner using Stockfish Evaluations

Generates training data from Stockfish, then uses linear regression
to learn optimal evaluation weights. Tests multiple ablations:
- Clipping strategies
- Position phase weighting
- Target transformations
- Regularization methods

Much faster than playing games - regression on 10k positions takes seconds.
"""

import os
import sys
import json
import time
import random
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'tuning_data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'tuning_results')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_training_positions(n_positions: int = 10000, 
                                 sf_depth: int = 12,
                                 output_file: Optional[str] = None) -> List[dict]:
    """
    Generate diverse positions with Stockfish evaluations.
    Returns list of {fen, sf_eval, phase, features}.
    """
    print(f"Generating {n_positions} positions with SF depth {sf_depth}...")
    
    if output_file and os.path.exists(output_file):
        print(f"Loading cached data from {output_file}")
        with open(output_file, 'rb') as f:
            return pickle.load(f)
    
    positions = []
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    start_time = time.time()
    attempts = 0
    max_attempts = n_positions * 3
    
    while len(positions) < n_positions and attempts < max_attempts:
        attempts += 1
        
        # Generate position from random game
        board = chess.Board()
        n_moves = random.randint(4, 50)
        
        for _ in range(n_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            
            # Weighted random - prefer captures and checks slightly
            weights = []
            for m in moves:
                w = 1.0
                if board.is_capture(m):
                    w = 2.0
                if board.gives_check(m):
                    w = 1.5
                weights.append(w)
            
            total = sum(weights)
            weights = [w/total for w in weights]
            move = random.choices(moves, weights=weights, k=1)[0]
            board.push(move)
        
        if board.is_game_over():
            continue
        
        # Get SF evaluation
        try:
            info = sf.analyse(board, chess.engine.Limit(depth=sf_depth))
            score_obj = info.get('score')
            if score_obj is None:
                continue
            
            pov_score = score_obj.white()
            
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                if mate_in is None:
                    continue
                # Convert mate to large but not infinite score
                sf_eval = 10000 - abs(mate_in) * 100
                if mate_in < 0:
                    sf_eval = -sf_eval
            else:
                sf_eval = pov_score.score()
                if sf_eval is None:
                    continue
            
            # Extract features
            features = extract_features(board)
            
            positions.append({
                'fen': board.fen(),
                'sf_eval': sf_eval,
                'phase': features['phase'],
                'features': features,
            })
            
            if len(positions) % 500 == 0:
                elapsed = time.time() - start_time
                rate = len(positions) / elapsed
                remaining = (n_positions - len(positions)) / rate
                print(f"  {len(positions)}/{n_positions} ({rate:.1f}/sec, ~{remaining:.0f}s remaining)")
                
        except Exception as e:
            continue
    
    sf.quit()
    
    elapsed = time.time() - start_time
    print(f"Generated {len(positions)} positions in {elapsed:.1f}s")
    
    if output_file:
        with open(output_file, 'wb') as f:
            pickle.dump(positions, f)
        print(f"Saved to {output_file}")
    
    return positions


def extract_features(board: chess.Board) -> Dict[str, Any]:
    """Extract individual evaluation features from a position."""
    from chess_engine.evaluation.features_material import get_material_score
    from chess_engine.evaluation.features_pst import get_pst_score
    from chess_engine.evaluation.features_special import get_bishop_pair_score, get_rook_placement_score
    from chess_engine.evaluation.features_pawn import get_pawn_structure_score
    from chess_engine.evaluation.features_knights import get_knight_outpost_score
    from chess_engine.evaluation.features_king_safety import get_king_safety_score
    from chess_engine.evaluation.features_space import get_mobility_score, get_space_score
    from chess_engine.evaluation.features_control import get_center_control_score
    from chess_engine.evaluation.features_tactics import get_tactical_score
    from chess_engine.evaluation.features_rooks import get_rook_features_score
    from chess_engine.evaluation.features_game_phase import calculate_game_phase
    
    phase = calculate_game_phase(board)
    
    # PST returns tuple (mg_score, eg_score) - blend based on phase
    pst_mg, pst_eg = get_pst_score(board)
    pst_blended = int(pst_mg * phase + pst_eg * (1 - phase))
    
    return {
        'material': get_material_score(board),
        'pst': pst_blended,
        'bishop_pair': get_bishop_pair_score(board),
        'rook_placement': get_rook_placement_score(board),
        'pawn_structure': get_pawn_structure_score(board),
        'knight_outpost': get_knight_outpost_score(board),
        'king_safety': get_king_safety_score(board),
        'mobility': get_mobility_score(board),
        'space': get_space_score(board),
        'center_control': get_center_control_score(board),
        'tactical': get_tactical_score(board),
        'rook_features': get_rook_features_score(board),
        'phase': phase,
    }


# ============================================================================
# TUNING CONFIGURATIONS
# ============================================================================

@dataclass
class TuningConfig:
    """Configuration for a tuning experiment."""
    name: str
    
    # Clipping strategy
    clip_min: float = -10000
    clip_max: float = 10000
    use_tanh: bool = False
    tanh_scale: float = 500
    
    # Position selection
    phase_min: float = 0.0  # Include positions with phase >= this
    phase_max: float = 1.0  # Include positions with phase <= this
    
    # Target transformation
    learn_residual: bool = False  # If True, learn residual from material
    use_win_prob: bool = False    # If True, convert to win probability
    
    # Regularization
    ridge_alpha: float = 0.0  # L2 regularization strength
    
    def __str__(self):
        return self.name


def get_ablation_configs() -> List[TuningConfig]:
    """Define all ablation configurations to test."""
    configs = []
    
    # 1. Clipping ablations
    configs.append(TuningConfig(name="clip_none", clip_min=-10000, clip_max=10000))
    configs.append(TuningConfig(name="clip_300", clip_min=-300, clip_max=300))
    configs.append(TuningConfig(name="clip_500", clip_min=-500, clip_max=500))
    configs.append(TuningConfig(name="clip_800", clip_min=-800, clip_max=800))
    configs.append(TuningConfig(name="clip_1500", clip_min=-1500, clip_max=1500))
    configs.append(TuningConfig(name="tanh_400", use_tanh=True, tanh_scale=400))
    configs.append(TuningConfig(name="tanh_600", use_tanh=True, tanh_scale=600))
    
    # 2. Phase ablations (with reasonable clipping)
    configs.append(TuningConfig(name="phase_opening", clip_min=-500, clip_max=500, 
                                phase_min=0.6, phase_max=1.0))
    configs.append(TuningConfig(name="phase_middle", clip_min=-500, clip_max=500,
                                phase_min=0.25, phase_max=0.75))
    configs.append(TuningConfig(name="phase_endgame", clip_min=-500, clip_max=500,
                                phase_min=0.0, phase_max=0.4))
    configs.append(TuningConfig(name="phase_balanced", clip_min=-500, clip_max=500))
    
    # 3. Target transformation ablations
    configs.append(TuningConfig(name="residual_500", clip_min=-500, clip_max=500,
                                learn_residual=True))
    configs.append(TuningConfig(name="winprob", use_win_prob=True))
    
    # 4. Regularization ablations
    configs.append(TuningConfig(name="ridge_light", clip_min=-500, clip_max=500,
                                ridge_alpha=0.01))
    configs.append(TuningConfig(name="ridge_heavy", clip_min=-500, clip_max=500,
                                ridge_alpha=0.1))
    
    return configs


# ============================================================================
# REGRESSION
# ============================================================================

def apply_target_transform(sf_eval: float, config: TuningConfig) -> float:
    """Apply target transformation based on config."""
    if config.use_win_prob:
        # Sigmoid to win probability
        return 1.0 / (1.0 + np.exp(-sf_eval / 400))
    
    if config.use_tanh:
        # Soft clipping
        return config.tanh_scale * np.tanh(sf_eval / config.tanh_scale)
    
    # Hard clipping
    return max(config.clip_min, min(config.clip_max, sf_eval))


def build_regression_data(positions: List[dict], config: TuningConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build feature matrix X and target vector y for regression."""
    
    # Filter by phase
    filtered = [p for p in positions 
                if config.phase_min <= p['phase'] <= config.phase_max]
    
    if len(filtered) < 100:
        print(f"  Warning: only {len(filtered)} positions after phase filtering, using all")
        filtered = positions
    
    # Feature names (in order)
    feature_names = [
        'pst', 'bishop_pair', 'rook_placement', 'pawn_structure',
        'knight_outpost', 'king_safety', 'mobility', 'space',
        'center_control', 'tactical', 'rook_features'
    ]
    
    X = []
    y = []
    
    for pos in filtered:
        features = pos['features']
        sf_eval = pos['sf_eval']
        
        # Build feature vector
        fv = [features[name] for name in feature_names]
        
        # Transform target
        target = apply_target_transform(sf_eval, config)
        
        if config.learn_residual:
            target = target - features['material']
        
        X.append(fv)
        y.append(target)
    
    return np.array(X), np.array(y), feature_names


def train_weights(positions: List[dict], config: TuningConfig) -> Tuple[Dict[str, float], float]:
    """Train weights using linear regression."""
    
    X, y, feature_names = build_regression_data(positions, config)
    
    # Convert to float64 for numerical stability
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    n_samples, n_features = X.shape
    
    # Ridge regression: (X'X + αI)^(-1) X'y
    XtX = X.T @ X
    if config.ridge_alpha > 0:
        XtX += config.ridge_alpha * np.eye(n_features)
    
    Xty = X.T @ y
    
    try:
        weights = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    # Compute training error
    predictions = X @ weights
    
    if config.use_win_prob:
        # For win prob, use cross-entropy or just MSE on probabilities
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
    else:
        mse = np.mean((predictions - y) ** 2)
        rmse = np.sqrt(mse)
    
    # Create weight dictionary
    learned = {name: float(w) for name, w in zip(feature_names, weights)}
    
    return learned, rmse


# ============================================================================
# WEIGHT APPLICATION
# ============================================================================

def apply_learned_weights(learned: Dict[str, float], blend_ratio: float = 1.0):
    """
    Apply learned weights to tuned_weights.py.
    blend_ratio: 1.0 = fully learned, 0.0 = keep original
    
    Maps learned feature weights to actual weight constants.
    """
    from chess_engine import tuned_weights as tw
    
    # Mapping from learned features to actual weight names and scaling
    # Since features are aggregates, we scale the constituent weights
    weight_mapping = {
        'bishop_pair': [('BISHOP_PAIR_BONUS', 1.0)],
        'rook_placement': [('ROOK_ON_OPEN_FILE_BONUS', 0.5), ('ROOK_ON_SEMI_OPEN_FILE_BONUS', 0.3)],
        'mobility': [('MOBILITY_BONUS', 1.0)],
        'king_safety': [('KING_ATTACK_WEIGHT', 0.1), ('KING_SHIELD_BONUS', 0.3)],
        'tactical': [('THREAT_BONUS', 0.3), ('HANGING_MINOR_PENALTY', -0.3)],
        'center_control': [('CENTER_CONTROL_BONUS', 1.0)],
        'knight_outpost': [('KNIGHT_OUTPOST_BONUS', 1.0)],
        'rook_features': [('ROOK_ON_7TH_BONUS', 0.5), ('CONNECTED_ROOKS_BONUS', 0.2)],
    }
    
    applied = {}
    for feature_name, learned_weight in learned.items():
        if feature_name not in weight_mapping:
            continue
        
        for weight_name, scale in weight_mapping[feature_name]:
            if hasattr(tw, weight_name):
                original = getattr(tw, weight_name)
                # Compute new value
                adjustment = learned_weight * scale * blend_ratio
                # For penalties (negative), adjustment should keep sign sensible
                if original < 0:
                    new_val = int(original + adjustment)
                else:
                    new_val = int(original * (1 - blend_ratio) + (original + adjustment) * blend_ratio)
                
                setattr(tw, weight_name, new_val)
                applied[weight_name] = new_val
    
    return applied


def save_weights_to_file(learned: Dict[str, float], config_name: str):
    """Save learned weights to a JSON file."""
    output_file = os.path.join(RESULTS_DIR, f"weights_{config_name}.json")
    with open(output_file, 'w') as f:
        json.dump(learned, f, indent=2)
    return output_file


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_weights_vs_stockfish(learned: Dict[str, float], 
                                   sf_elo: int = 1320,
                                   n_games: int = 12,
                                   depth: int = 2) -> Tuple[float, str]:
    """Play games to evaluate a set of learned weights."""
    import importlib
    from chess_engine import tuned_weights as tw
    
    # Save original weights
    original_weights = {}
    weight_names = ['BISHOP_PAIR_BONUS', 'ROOK_ON_OPEN_FILE_BONUS', 'MOBILITY_BONUS',
                    'KING_ATTACK_WEIGHT', 'THREAT_BONUS', 'CENTER_CONTROL_BONUS',
                    'KNIGHT_OUTPOST_BONUS', 'ROOK_ON_7TH_BONUS', 'HANGING_MINOR_PENALTY',
                    'TRAPPED_PIECE_PENALTY', 'KING_SHIELD_BONUS']
    
    for name in weight_names:
        if hasattr(tw, name):
            original_weights[name] = getattr(tw, name)
    
    # Apply learned weights
    apply_learned_weights(learned, blend_ratio=0.5)  # 50% blend
    
    # Reload evaluation modules
    modules_to_reload = [
        'chess_engine.evaluation.features_special',
        'chess_engine.evaluation.features_tactics',
        'chess_engine.evaluation.features_king_safety',
        'chess_engine.evaluation.features_space',
        'chess_engine.evaluation.features_control',
        'chess_engine.evaluation.features_knights',
        'chess_engine.evaluation.features_rooks',
        'chess_engine.eval_main',
        'chess_engine.engine',
    ]
    
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except:
                pass
    
    from chess_engine.engine import Engine
    
    # Play games
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    
    results = []
    for i in range(n_games):
        board = chess.Board()
        our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        moves = 0
        
        while not board.is_game_over() and moves < 40:
            if board.turn == our_color:
                move = Engine(board.copy(), depth=depth).get_best_move()
            else:
                move = sf.play(board, chess.engine.Limit(time=0.02)).move
            if move is None:
                break
            board.push(move)
            moves += 1
        
        if board.is_checkmate():
            result = 1.0 if (not board.turn) == our_color else 0.0
        elif board.is_game_over():
            result = 0.5
        else:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
            mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) 
                     for p, v in vals.items())
            if our_color == chess.BLACK:
                mat = -mat
            result = 1.0 if mat > 2 else (0.0 if mat < -2 else 0.5)
        
        results.append(result)
    
    sf.quit()
    
    # Restore original weights
    for name, val in original_weights.items():
        setattr(tw, name, val)
    
    # Reload again to restore
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except:
                pass
    
    score = sum(results) / len(results)
    wins = sum(1 for r in results if r == 1.0)
    draws = sum(1 for r in results if r == 0.5)
    losses = sum(1 for r in results if r == 0.0)
    result_str = f"{wins}W-{draws}D-{losses}L = {score*100:.0f}%"
    
    return score, result_str


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(n_positions: int = 5000,
                   n_games_per_config: int = 12,
                   configs: Optional[List[TuningConfig]] = None):
    """Run the full tuning experiment."""
    
    print("=" * 70)
    print("SUPERVISED WEIGHT TUNING EXPERIMENT")
    print("=" * 70)
    
    # Generate or load training data
    data_file = os.path.join(DATA_DIR, f'positions_{n_positions}.pkl')
    positions = generate_training_positions(n_positions, sf_depth=12, output_file=data_file)
    
    # Use default configs if not specified
    if configs is None:
        configs = get_ablation_configs()
    
    print(f"\nTesting {len(configs)} configurations...")
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*50}")
        print(f"Config: {config.name}")
        print(f"{'='*50}")
        
        # Train weights
        start = time.time()
        learned, rmse = train_weights(positions, config)
        train_time = time.time() - start
        
        print(f"  Training RMSE: {rmse:.1f} cp")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Learned weights: {learned}")
        
        # Save weights
        save_weights_to_file(learned, config.name)
        
        # Evaluate by playing games
        print(f"  Evaluating vs SF 1320 ({n_games_per_config} games)...")
        score, result_str = evaluate_weights_vs_stockfish(
            learned, sf_elo=1320, n_games=n_games_per_config, depth=2
        )
        
        print(f"  Result: {result_str}")
        
        results[config.name] = {
            'config': {
                'clip_min': config.clip_min,
                'clip_max': config.clip_max,
                'use_tanh': config.use_tanh,
                'phase_min': config.phase_min,
                'phase_max': config.phase_max,
                'learn_residual': config.learn_residual,
                'ridge_alpha': config.ridge_alpha,
            },
            'learned_weights': learned,
            'rmse': rmse,
            'game_score': score,
            'game_result': result_str,
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['game_score'], reverse=True)
    
    for name, data in sorted_results:
        print(f"{name:20s}: {data['game_result']:15s} (RMSE={data['rmse']:.1f})")
    
    best_name = sorted_results[0][0]
    best_data = sorted_results[0][1]
    
    print(f"\nBest config: {best_name}")
    print(f"Best weights: {best_data['learned_weights']}")
    
    # Save all results
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_file}")
    
    return best_name, best_data


def quick_test():
    """Quick test with fewer positions and games."""
    configs = [
        TuningConfig(name="clip_300", clip_min=-300, clip_max=300),
        TuningConfig(name="clip_500", clip_min=-500, clip_max=500),
        TuningConfig(name="residual_500", clip_min=-500, clip_max=500, learn_residual=True),
    ]
    
    return run_experiment(n_positions=2000, n_games_per_config=8, configs=configs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer configs')
    parser.add_argument('--positions', type=int, default=5000, help='Number of positions to generate')
    parser.add_argument('--games', type=int, default=12, help='Games per config for evaluation')
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        run_experiment(n_positions=args.positions, n_games_per_config=args.games)
