#!/usr/bin/env python3
"""
Multi-Strategy Weight Tuner

Tunes evaluation weights using Stockfish evaluations as ground truth.
Tests multiple tuning strategies:
1. Raw SF eval (no clipping)
2. Clipped SF eval (±300cp, ±500cp, ±800cp)
3. Soft clipping (tanh scaling)
4. Win-Draw-Loss classification targets
5. Material-only residual learning

Each strategy produces a set of weights, which are then tested
against SF 1320 to determine the best approach.
"""

import os
import sys
import json
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

# Weights to tune (subset of most impactful ones)
TUNABLE_WEIGHTS = [
    'BISHOP_PAIR_BONUS',
    'ROOK_ON_OPEN_FILE_BONUS', 
    'ROOK_ON_SEMI_OPEN_FILE_BONUS',
    'KING_SHIELD_BONUS',
    'KING_OPEN_FILE_PENALTY',
    'KING_ATTACK_WEIGHT',
    'MOBILITY_BONUS',
    'TEMPO_BONUS',
    'KNIGHT_OUTPOST_BONUS',
    'CENTER_CONTROL_BONUS',
    'THREAT_BONUS',
    'ROOK_ON_7TH_BONUS',
    'TRAPPED_PIECE_PENALTY',
    'HANGING_MINOR_PENALTY',
    'HANGING_ROOK_PENALTY',
    'HANGING_QUEEN_PENALTY',
    'PASSED_PAWN_BLOCKER_BONUS',
]

@dataclass
class TuningConfig:
    name: str
    clip_min: float = -10000
    clip_max: float = 10000
    use_tanh: bool = False
    tanh_scale: float = 500
    use_wdl: bool = False
    residual_only: bool = False  # Learn residual from material
    
def generate_positions(n_positions: int = 2000) -> List[Tuple[chess.Board, int]]:
    """Generate diverse positions with Stockfish evaluations."""
    print(f"Generating {n_positions} positions with SF evaluations...")
    
    positions = []
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    
    # Generate from random games
    games_needed = n_positions // 30 + 1
    
    for game_idx in range(games_needed):
        if len(positions) >= n_positions:
            break
            
        board = chess.Board()
        
        # Play 5-40 random moves to get diverse positions
        n_moves = random.randint(5, 40)
        for _ in range(n_moves):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            # Bias toward reasonable moves
            move = random.choice(moves)
            board.push(move)
        
        if board.is_game_over():
            continue
            
        # Evaluate with Stockfish (depth 12 for good accuracy)
        try:
            info = sf.analyse(board, chess.engine.Limit(depth=12))
            score = info.get('score')
            if score is None:
                continue
            score = score.white()
            
            if score.is_mate():
                # Convert mate to large score
                mate_in = score.mate()
                if mate_in is None:
                    continue
                cp = 10000 if mate_in > 0 else -10000
            else:
                cp = score.score()
                if cp is None:
                    continue
            
            positions.append((board.copy(), cp))
            
            if len(positions) % 200 == 0:
                print(f"  Generated {len(positions)} positions...")
                
        except Exception as e:
            continue
    
    sf.quit()
    print(f"Generated {len(positions)} positions")
    return positions[:n_positions]

def extract_features(board: chess.Board) -> Dict[str, float]:
    """Extract evaluation features from a position."""
    from chess_engine.evaluation.features_special import get_bishop_pair_score, get_rook_placement_score
    from chess_engine.evaluation.features_pawn import get_pawn_structure_score
    from chess_engine.evaluation.features_knights import get_knight_outpost_score
    from chess_engine.evaluation.features_king_safety import get_king_safety_score
    from chess_engine.evaluation.features_space import get_mobility_score
    from chess_engine.evaluation.features_control import get_center_control_score
    from chess_engine.evaluation.features_tactics import get_tactical_score
    from chess_engine.evaluation.features_rooks import get_rook_features_score
    from chess_engine.evaluation.features_material import get_material_score
    from chess_engine.evaluation.features_game_phase import calculate_game_phase
    
    phase = calculate_game_phase(board)
    
    features = {
        'material': get_material_score(board),
        'bishop_pair': get_bishop_pair_score(board),
        'rook_placement': get_rook_placement_score(board),
        'pawn_structure': get_pawn_structure_score(board),
        'knight_outpost': get_knight_outpost_score(board),
        'king_safety': get_king_safety_score(board),
        'mobility': get_mobility_score(board),
        'center_control': get_center_control_score(board),
        'tactical': get_tactical_score(board),
        'rook_features': get_rook_features_score(board),
        'phase': phase,
    }
    
    return features

def apply_config(sf_eval: int, config: TuningConfig) -> float:
    """Apply tuning configuration to transform SF evaluation."""
    if config.use_wdl:
        # Convert to WDL probability
        if sf_eval > 200:
            return 1.0  # Win
        elif sf_eval < -200:
            return 0.0  # Loss
        else:
            return 0.5  # Draw
    
    if config.use_tanh:
        # Soft clipping with tanh
        return np.tanh(sf_eval / config.tanh_scale) * config.tanh_scale
    
    # Hard clipping
    return max(config.clip_min, min(config.clip_max, sf_eval))

def tune_weights_linear(positions: List[Tuple[chess.Board, int]], 
                        config: TuningConfig) -> Dict[str, float]:
    """Tune weights using linear regression."""
    print(f"\nTuning with strategy: {config.name}")
    
    # Build feature matrix and target vector
    X = []
    y = []
    
    for board, sf_eval in positions:
        features = extract_features(board)
        target = apply_config(sf_eval, config)
        
        if config.residual_only:
            # Target is SF eval minus material
            target = target - features['material']
        
        # Feature vector (excluding material for residual learning)
        feature_vec = [
            features['bishop_pair'],
            features['rook_placement'],
            features['pawn_structure'],
            features['knight_outpost'],
            features['king_safety'],
            features['mobility'],
            features['center_control'],
            features['tactical'],
            features['rook_features'],
        ]
        
        X.append(feature_vec)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    # Add small regularization to avoid overfitting
    from numpy.linalg import lstsq
    
    # Ridge regression: (X'X + λI)^-1 X'y
    lambda_reg = 0.01
    XtX = X.T @ X + lambda_reg * np.eye(X.shape[1])
    Xty = X.T @ y
    
    try:
        weights = np.linalg.solve(XtX, Xty)
    except:
        weights = lstsq(X, y, rcond=None)[0]
    
    # Map back to weight names (approximate mapping)
    weight_names = [
        'BISHOP_PAIR_BONUS',
        'ROOK_ON_OPEN_FILE_BONUS',
        'PAWN_STRUCTURE_WEIGHT',  # Aggregate
        'KNIGHT_OUTPOST_BONUS',
        'KING_SAFETY_WEIGHT',  # Aggregate
        'MOBILITY_BONUS',
        'CENTER_CONTROL_BONUS',
        'TACTICAL_WEIGHT',  # Aggregate
        'ROOK_FEATURES_WEIGHT',  # Aggregate
    ]
    
    learned = {name: float(w) for name, w in zip(weight_names, weights)}
    
    # Compute training error
    predictions = X @ weights
    if config.residual_only:
        # Add back material for comparison
        predictions = predictions + np.array([extract_features(b)['material'] for b, _ in positions])
        y_compare = np.array([apply_config(sf, config) for _, sf in positions])
    else:
        y_compare = y
    
    mse = np.mean((predictions - y_compare) ** 2)
    rmse = np.sqrt(mse)
    print(f"  Training RMSE: {rmse:.1f} centipawns")
    
    return learned

def test_weights_vs_stockfish(weight_adjustments: Dict[str, float], 
                              n_games: int = 8) -> Tuple[float, str]:
    """Test weight adjustments against Stockfish 1320."""
    from chess_engine import tuned_weights as tw
    from chess_engine.engine import Engine
    import importlib
    
    # Apply weight adjustments
    original_values = {}
    for name, delta in weight_adjustments.items():
        if hasattr(tw, name):
            original_values[name] = getattr(tw, name)
            # Scale the learned weight appropriately
            current = getattr(tw, name)
            # Blend: 70% original, 30% learned
            new_val = int(current * 0.7 + delta * 0.3)
            setattr(tw, name, new_val)
    
    # Reload modules to pick up changes
    importlib.reload(sys.modules['chess_engine.eval_main'])
    importlib.reload(sys.modules['chess_engine.engine'])
    from chess_engine.engine import Engine
    
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': 1320})
    
    results = []
    for i in range(n_games):
        board = chess.Board()
        our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        moves = 0
        
        while not board.is_game_over() and moves < 60:
            if board.turn == our_color:
                move = Engine(board.copy(), depth=3).get_best_move()
            else:
                move = sf.play(board, chess.engine.Limit(time=0.02)).move
            if move is None:
                break
            board.push(move)
            moves += 1
        
        if board.is_checkmate():
            result = 'W' if (not board.turn) == our_color else 'L'
        elif board.is_game_over():
            result = 'D'
        else:
            vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
            mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
            if our_color == chess.BLACK:
                mat = -mat
            result = 'W' if mat > 2 else ('L' if mat < -2 else 'D')
        
        results.append(result)
    
    sf.quit()
    
    # Restore original values
    for name, val in original_values.items():
        setattr(tw, name, val)
    
    # Reload to restore
    importlib.reload(sys.modules['chess_engine.eval_main'])
    importlib.reload(sys.modules['chess_engine.engine'])
    
    w, d, l = results.count('W'), results.count('D'), results.count('L')
    score = (w + 0.5 * d) / n_games
    result_str = f"{w}W-{d}D-{l}L = {score*100:.0f}%"
    
    return score, result_str

def main():
    print("=" * 60)
    print("MULTI-STRATEGY WEIGHT TUNER")
    print("=" * 60)
    
    # Generate training positions
    positions = generate_positions(1500)
    
    # Define tuning strategies
    strategies = [
        TuningConfig(name="raw", clip_min=-10000, clip_max=10000),
        TuningConfig(name="clip_300", clip_min=-300, clip_max=300),
        TuningConfig(name="clip_500", clip_min=-500, clip_max=500),
        TuningConfig(name="clip_800", clip_min=-800, clip_max=800),
        TuningConfig(name="tanh_300", use_tanh=True, tanh_scale=300),
        TuningConfig(name="tanh_500", use_tanh=True, tanh_scale=500),
        TuningConfig(name="residual", residual_only=True, clip_min=-500, clip_max=500),
    ]
    
    results = {}
    
    for config in strategies:
        # Tune weights
        learned_weights = tune_weights_linear(positions, config)
        
        # Test against Stockfish
        print(f"  Testing {config.name} weights vs SF 1320...")
        score, result_str = test_weights_vs_stockfish(learned_weights, n_games=8)
        
        results[config.name] = {
            'weights': learned_weights,
            'score': score,
            'result': result_str,
        }
        
        print(f"  Result: {result_str}")
    
    # Find best strategy
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    best_name = None
    best_score = -1
    
    for name, data in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"{name:15s}: {data['result']}")
        if data['score'] > best_score:
            best_score = data['score']
            best_name = name
    
    print(f"\nBest strategy: {best_name} ({best_score*100:.0f}%)")
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'tuning_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python
        save_results = {}
        for name, data in results.items():
            save_results[name] = {
                'weights': {k: float(v) for k, v in data['weights'].items()},
                'score': float(data['score']),
                'result': data['result'],
            }
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Run more games with best strategy to verify
    print(f"\nVerification run with {best_name} (16 games)...")
    best_weights = results[best_name]['weights']
    score, result_str = test_weights_vs_stockfish(best_weights, n_games=16)
    print(f"Verification: {result_str}")
    
    return best_name, results[best_name]

if __name__ == '__main__':
    main()
