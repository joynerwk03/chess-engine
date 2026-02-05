#!/usr/bin/env python3
"""
Quick Weight Optimizer

Faster approach: directly optimize weights by playing games.
Uses coordinate descent on a small set of key weights.
"""

import os
import sys
import json
import time
import random
import copy

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

# Key weights to optimize (most impactful)
OPTIMIZE_WEIGHTS = {
    'KING_ATTACK_WEIGHT': {'min': 5, 'max': 25, 'step': 5},
    'MOBILITY_BONUS': {'min': 2, 'max': 10, 'step': 2},
    'THREAT_BONUS': {'min': 30, 'max': 80, 'step': 10},
    'HANGING_MINOR_PENALTY': {'min': -200, 'max': -80, 'step': 20},
    'TRAPPED_PIECE_PENALTY': {'min': -150, 'max': -50, 'step': 25},
    'ROOK_ON_7TH_BONUS': {'min': 50, 'max': 120, 'step': 15},
    'BISHOP_PAIR_BONUS': {'min': 40, 'max': 80, 'step': 10},
    'TEMPO_BONUS': {'min': 20, 'max': 50, 'step': 10},
}

def play_games(weights: dict, n_games: int = 6, depth: int = 3) -> float:
    """Play games against SF 1320 and return score."""
    from chess_engine import tuned_weights as tw
    import importlib
    
    # Apply weights
    for name, value in weights.items():
        if hasattr(tw, name):
            setattr(tw, name, int(value))
    
    # Force reload
    for mod_name in list(sys.modules.keys()):
        if 'chess_engine' in mod_name and mod_name != 'chess_engine.tuned_weights':
            try:
                importlib.reload(sys.modules[mod_name])
            except:
                pass
    
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
            mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
            if our_color == chess.BLACK:
                mat = -mat
            result = 1.0 if mat > 2 else (0.0 if mat < -2 else 0.5)
        
        results.append(result)
    
    sf.quit()
    return sum(results) / len(results)

def get_current_weights() -> dict:
    """Get current weight values."""
    from chess_engine import tuned_weights as tw
    return {name: getattr(tw, name) for name in OPTIMIZE_WEIGHTS.keys()}

def coordinate_descent(n_rounds: int = 3, games_per_test: int = 6) -> dict:
    """Optimize weights using coordinate descent."""
    print("=" * 60)
    print("COORDINATE DESCENT WEIGHT OPTIMIZER")
    print("=" * 60)
    
    current_weights = get_current_weights()
    print(f"\nStarting weights: {current_weights}")
    
    # Test baseline
    print("\nTesting baseline...")
    baseline_score = play_games(current_weights, n_games=games_per_test)
    print(f"Baseline score: {baseline_score*100:.0f}%")
    
    best_weights = current_weights.copy()
    best_score = baseline_score
    
    for round_num in range(n_rounds):
        print(f"\n--- Round {round_num + 1}/{n_rounds} ---")
        improved = False
        
        for weight_name, bounds in OPTIMIZE_WEIGHTS.items():
            current_val = best_weights[weight_name]
            
            # Try values in range
            test_values = []
            val = bounds['min']
            while val <= bounds['max']:
                if val != current_val:
                    test_values.append(val)
                val += bounds['step']
            
            if not test_values:
                continue
            
            # Random subset for speed
            if len(test_values) > 3:
                test_values = random.sample(test_values, 3)
            
            for test_val in test_values:
                test_weights = best_weights.copy()
                test_weights[weight_name] = test_val
                
                score = play_games(test_weights, n_games=games_per_test)
                
                symbol = '↑' if score > best_score else ('↓' if score < best_score else '=')
                print(f"  {weight_name}={test_val}: {score*100:.0f}% {symbol}")
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
                    improved = True
                    print(f"  *** New best! ***")
        
        if not improved:
            print("No improvement this round, stopping early.")
            break
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Best score: {best_score*100:.0f}%")
    print(f"Best weights: {best_weights}")
    print(f"Improvement: {(best_score - baseline_score)*100:+.0f}%")
    
    return best_weights

def grid_search_key_weights(games_per_test: int = 4) -> dict:
    """Grid search on the most important weights."""
    print("=" * 60)
    print("FOCUSED GRID SEARCH")
    print("=" * 60)
    
    # Focus on just 3 key weights
    key_weights = {
        'KING_ATTACK_WEIGHT': [10, 15, 20],
        'HANGING_MINOR_PENALTY': [-180, -150, -120],
        'MOBILITY_BONUS': [4, 6, 8],
    }
    
    current = get_current_weights()
    best_weights = current.copy()
    best_score = 0
    
    total_combos = 1
    for vals in key_weights.values():
        total_combos *= len(vals)
    
    print(f"Testing {total_combos} combinations...")
    
    combo_num = 0
    for ka in key_weights['KING_ATTACK_WEIGHT']:
        for hm in key_weights['HANGING_MINOR_PENALTY']:
            for mb in key_weights['MOBILITY_BONUS']:
                combo_num += 1
                
                test_weights = current.copy()
                test_weights['KING_ATTACK_WEIGHT'] = ka
                test_weights['HANGING_MINOR_PENALTY'] = hm
                test_weights['MOBILITY_BONUS'] = mb
                
                score = play_games(test_weights, n_games=games_per_test)
                
                print(f"[{combo_num}/{total_combos}] KA={ka} HM={hm} MB={mb}: {score*100:.0f}%")
                
                if score > best_score:
                    best_score = score
                    best_weights = test_weights.copy()
                    print(f"  *** New best! ***")
    
    print(f"\n{'='*60}")
    print(f"Best: KA={best_weights['KING_ATTACK_WEIGHT']} "
          f"HM={best_weights['HANGING_MINOR_PENALTY']} "
          f"MB={best_weights['MOBILITY_BONUS']} = {best_score*100:.0f}%")
    
    return best_weights

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['coordinate', 'grid'], default='grid')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--games', type=int, default=6)
    args = parser.parse_args()
    
    if args.method == 'coordinate':
        best = coordinate_descent(n_rounds=args.rounds, games_per_test=args.games)
    else:
        best = grid_search_key_weights(games_per_test=args.games)
    
    # Save results
    output_file = os.path.join(os.path.dirname(__file__), 'quick_tune_results.json')
    with open(output_file, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved to {output_file}")
    
    # Verification run
    print("\nVerification (12 games)...")
    final_score = play_games(best, n_games=12)
    print(f"Verification: {final_score*100:.0f}%")

if __name__ == '__main__':
    main()
