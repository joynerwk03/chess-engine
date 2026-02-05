#!/usr/bin/env python3
"""
Master improvement script - runs comprehensive optimization.
1. Saves current version as baseline
2. Runs evolutionary tuning on key weights
3. Tests improvements
4. Keeps the best configuration
"""
import os
import sys
import time
import subprocess
import json
import random
import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_fast_game(depth, sf_elo, our_color, max_moves=40):
    """Play one fast game"""
    from chess_engine.engine import Engine
    
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        if board.turn == our_color:
            eng = Engine(board.copy(), depth=depth)
            move = eng.get_best_move()
        else:
            result = sf.play(board, chess.engine.Limit(time=0.01))
            move = result.move
        if move is None:
            break
        board.push(move)
        moves += 1
    
    sf.quit()
    
    if board.is_checkmate():
        winner = not board.turn
        return 1.0 if winner == our_color else 0.0
    elif board.is_game_over():
        return 0.5
    else:
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
        if our_color == chess.BLACK:
            mat = -mat
        return max(0.0, min(1.0, 0.5 + mat / 12.0))

def benchmark(depth=3, num_games=8):
    """Run benchmark and return score"""
    total = 0.0
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        total += play_fast_game(depth, 1320, color, max_moves=50)
    return total / num_games

def get_weight(name):
    """Get current weight value"""
    from chess_engine import tuned_weights
    return getattr(tuned_weights, name, 0)

def set_weight(name, value):
    """Set weight value"""
    from chess_engine import tuned_weights
    setattr(tuned_weights, name, int(value))

def reload_engine():
    """Force reload of engine modules"""
    import importlib
    mods = [
        'chess_engine.tuned_weights',
        'chess_engine.evaluation.features_tactics',
        'chess_engine.evaluation.features_king_safety',
        'chess_engine.evaluation.features_pawn',
        'chess_engine.evaluation.features_piece_activity',
        'chess_engine.evaluation.features_material',
        'chess_engine.eval_main',
        'chess_engine.engine',
    ]
    for mod_name in mods:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])

# Key weights to tune with their reasonable ranges
WEIGHTS = {
    'THREAT_BONUS': (20, 60),
    'HANGING_MINOR_PENALTY': (-180, -60),
    'KING_SHIELD_BONUS': (5, 20),
    'KING_OPEN_FILE_PENALTY': (-100, -40),
    'MOBILITY_KNIGHT': (3, 10),
    'MOBILITY_BISHOP': (3, 12),
    'TRAPPED_PIECE_PENALTY': (-150, -50),
    'KNIGHT_OUTPOST_BONUS': (20, 60),
    'BISHOP_PAIR_BONUS': (30, 70),
    'ROOK_ON_OPEN_FILE_BONUS': (20, 50),
    'DOUBLED_PAWN_PENALTY': (-30, -10),
    'DEVELOPMENT_BONUS': (10, 30),
}

def main():
    print("=" * 60)
    print("MASTER ENGINE IMPROVEMENT")
    print("=" * 60)
    print()
    
    # 1. Baseline benchmark
    print("Phase 1: Baseline benchmark...")
    baseline_score = benchmark(depth=3, num_games=6)
    print(f"Baseline score: {baseline_score:.1%}")
    print()
    
    # Save baseline weights
    baseline_weights = {name: get_weight(name) for name in WEIGHTS}
    
    # 2. Quick evolutionary search
    print("Phase 2: Evolutionary weight tuning (5 generations)...")
    
    best_weights = baseline_weights.copy()
    best_score = baseline_score
    
    # Population of weight sets
    population = [baseline_weights.copy()]
    for _ in range(3):  # 4 total individuals
        mutant = baseline_weights.copy()
        for name in mutant:
            if random.random() < 0.4:
                lo, hi = WEIGHTS[name]
                mutant[name] = int(mutant[name] + random.gauss(0, (hi - lo) * 0.2))
                mutant[name] = max(lo, min(hi, mutant[name]))
        population.append(mutant)
    
    for gen in range(5):
        print(f"\n--- Generation {gen + 1}/5 ---")
        
        scores = []
        for i, weights in enumerate(population):
            # Apply weights
            for name, val in weights.items():
                set_weight(name, val)
            reload_engine()
            
            # Evaluate with 4 games
            score = benchmark(depth=2, num_games=4)  # Depth 2 for speed
            scores.append(score)
            print(f"  Individual {i + 1}: {score:.1%}")
            
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
                print(f"    ** NEW BEST **")
        
        # Selection: keep top 2
        paired = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        survivors = [p[1] for p in paired[:2]]
        
        # Create next generation
        population = survivors.copy()
        while len(population) < 4:
            parent = random.choice(survivors)
            child = parent.copy()
            for name in child:
                if random.random() < 0.3:
                    lo, hi = WEIGHTS[name]
                    child[name] = int(child[name] + random.gauss(0, (hi - lo) * 0.15))
                    child[name] = max(lo, min(hi, child[name]))
            population.append(child)
    
    # 3. Apply best weights
    print("\n" + "=" * 60)
    print("Phase 3: Applying best weights")
    for name, val in best_weights.items():
        set_weight(name, val)
    reload_engine()
    
    # 4. Final verification at depth 3
    print("\nPhase 4: Final verification at depth 3...")
    final_score = benchmark(depth=3, num_games=8)
    print(f"Final score: {final_score:.1%}")
    
    # 5. Report
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline: {baseline_score:.1%}")
    print(f"Final:    {final_score:.1%}")
    print(f"Change:   {(final_score - baseline_score):+.1%}")
    
    print("\nWeight changes:")
    for name in sorted(WEIGHTS.keys()):
        old = baseline_weights[name]
        new = best_weights[name]
        if old != new:
            print(f"  {name}: {old} -> {new}")
    
    # Save results
    results = {
        'baseline_score': baseline_score,
        'final_score': final_score,
        'baseline_weights': baseline_weights,
        'best_weights': best_weights,
    }
    
    results_dir = os.path.join(os.path.dirname(__file__), 'tuning_results')
    os.makedirs(results_dir, exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(results_dir, f'run_{timestamp}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to tuning_results/run_{timestamp}.json")
    
    return final_score

if __name__ == '__main__':
    main()
