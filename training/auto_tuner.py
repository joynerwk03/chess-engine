#!/usr/bin/env python3
"""
Auto-tuner: Uses Stockfish to tune evaluation weights via gradient-free optimization.
Runs faster games with fewer moves to iterate quickly.
"""
import chess
import chess.engine
import sys
import os
import json
import random
import copy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'tuning_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Weights to tune and their ranges
TUNABLE_WEIGHTS = {
    'HANGING_PAWN_PENALTY': (-80, -20),
    'HANGING_MINOR_PENALTY': (-200, -80),
    'HANGING_ROOK_PENALTY': (-300, -150),
    'HANGING_QUEEN_PENALTY': (-500, -250),
    'ATTACKED_BY_LESSER_PENALTY': (-100, -30),
    'THREAT_BONUS': (15, 60),
    'PIN_BONUS': (10, 50),
    'PINNED_PIECE_PENALTY': (-80, -20),
    'KING_SHIELD_BONUS': (3, 15),
    'KING_OPEN_FILE_PENALTY': (-100, -30),
    'KING_ATTACK_WEIGHT': (2, 10),
    'MOBILITY_KNIGHT': (2, 8),
    'MOBILITY_BISHOP': (2, 10),
    'MOBILITY_ROOK': (1, 6),
    'MOBILITY_QUEEN': (1, 4),
    'KNIGHT_OUTPOST_BONUS': (15, 50),
    'BISHOP_PAIR_BONUS': (20, 60),
    'ROOK_ON_OPEN_FILE_BONUS': (15, 45),
    'ROOK_ON_SEMI_OPEN_FILE_BONUS': (8, 25),
    'CONNECTED_ROOKS_BONUS': (8, 30),
    'DOUBLED_PAWN_PENALTY': (-30, -8),
    'ISOLATED_PAWN_PENALTY': (-25, -8),
    'BACKWARD_PAWN_PENALTY': (-20, -5),
    'PAWN_CHAIN_BONUS': (3, 15),
    'TRAPPED_PIECE_PENALTY': (-120, -40),
    'DEVELOPMENT_BONUS': (8, 25),
}

def load_weights():
    """Load current weights from tuned_weights.py"""
    from chess_engine import tuned_weights
    current = {}
    for name in TUNABLE_WEIGHTS:
        if hasattr(tuned_weights, name):
            current[name] = getattr(tuned_weights, name)
        else:
            # Use middle of range as default
            lo, hi = TUNABLE_WEIGHTS[name]
            current[name] = (lo + hi) // 2
    return current

def apply_weights(weights):
    """Dynamically apply weights to the tuned_weights module"""
    from chess_engine import tuned_weights
    for name, value in weights.items():
        if hasattr(tuned_weights, name):
            setattr(tuned_weights, name, value)

def play_fast_game(our_depth, sf_elo, our_color, max_moves=40):
    """Play a fast game with limited moves for quick evaluation"""
    from chess_engine.engine import Engine
    
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    stockfish.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        if board.turn == our_color:
            engine = Engine(board.copy(), depth=our_depth)
            move = engine.get_best_move()
        else:
            result = stockfish.play(board, chess.engine.Limit(time=0.01))
            move = result.move
        if move is None:
            break
        board.push(move)
        moves += 1
    
    stockfish.quit()
    
    # Score the game
    if board.is_checkmate():
        winner = not board.turn
        return 1.0 if winner == our_color else 0.0
    elif board.is_game_over():
        return 0.5
    else:
        # Evaluate material advantage
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        material = 0
        for pt, val in values.items():
            material += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        if our_color == chess.BLACK:
            material = -material
        # Convert to score: +3 material = 0.75, -3 = 0.25
        return max(0.0, min(1.0, 0.5 + material / 12.0))

def evaluate_weights(weights, num_games=6, depth=2, sf_elo=1320):
    """Evaluate a weight configuration by playing games"""
    apply_weights(weights)
    
    # Force reimport to pick up weight changes
    import importlib
    from chess_engine import eval_main
    from chess_engine.evaluation import features_tactics, features_king_safety
    from chess_engine.evaluation import features_pawn, features_piece_activity
    importlib.reload(features_tactics)
    importlib.reload(features_king_safety)
    importlib.reload(features_pawn)
    importlib.reload(features_piece_activity)
    importlib.reload(eval_main)
    
    total_score = 0.0
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        score = play_fast_game(depth, sf_elo, color)
        total_score += score
    
    return total_score / num_games

def mutate_weights(weights, mutation_rate=0.3, mutation_strength=0.2):
    """Create a mutated copy of weights"""
    new_weights = copy.deepcopy(weights)
    for name in new_weights:
        if random.random() < mutation_rate:
            lo, hi = TUNABLE_WEIGHTS[name]
            range_size = hi - lo
            delta = random.gauss(0, mutation_strength * range_size)
            new_weights[name] = int(max(lo, min(hi, new_weights[name] + delta)))
    return new_weights

def crossover_weights(w1, w2):
    """Create a child by crossing over two weight sets"""
    child = {}
    for name in w1:
        child[name] = w1[name] if random.random() < 0.5 else w2[name]
    return child

def evolutionary_tune(generations=10, population_size=6, games_per_eval=4):
    """Use evolutionary algorithm to tune weights"""
    print(f"Starting evolutionary tuning: {generations} generations, pop={population_size}")
    print(f"Games per evaluation: {games_per_eval}")
    print()
    
    # Initialize population with current weights + random variations
    base_weights = load_weights()
    population = [base_weights]
    for _ in range(population_size - 1):
        population.append(mutate_weights(base_weights, mutation_rate=0.5, mutation_strength=0.3))
    
    best_ever = base_weights
    best_score_ever = 0.0
    
    for gen in range(generations):
        print(f"=== Generation {gen + 1}/{generations} ===")
        
        # Evaluate all individuals
        scores = []
        for i, weights in enumerate(population):
            score = evaluate_weights(weights, num_games=games_per_eval, depth=2, sf_elo=1320)
            scores.append(score)
            print(f"  Individual {i + 1}: {score:.1%}")
        
        # Track best
        best_idx = scores.index(max(scores))
        if scores[best_idx] > best_score_ever:
            best_score_ever = scores[best_idx]
            best_ever = copy.deepcopy(population[best_idx])
            print(f"  ** New best: {best_score_ever:.1%} **")
        
        # Select top half
        paired = list(zip(scores, population))
        paired.sort(key=lambda x: x[0], reverse=True)
        survivors = [p[1] for p in paired[:population_size // 2]]
        
        # Create next generation
        new_population = survivors.copy()
        while len(new_population) < population_size:
            if random.random() < 0.5 and len(survivors) >= 2:
                # Crossover
                p1, p2 = random.sample(survivors, 2)
                child = crossover_weights(p1, p2)
                child = mutate_weights(child, mutation_rate=0.2, mutation_strength=0.15)
            else:
                # Mutation
                parent = random.choice(survivors)
                child = mutate_weights(parent, mutation_rate=0.4, mutation_strength=0.2)
            new_population.append(child)
        
        population = new_population
        print()
    
    return best_ever, best_score_ever

def save_weights(weights, score, filename=None):
    """Save tuned weights to a file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuned_{timestamp}_{score:.0%}.json"
    
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump({'weights': weights, 'score': score}, f, indent=2)
    print(f"Saved weights to {filepath}")
    return filepath

def main():
    print("=" * 60)
    print("Chess Engine Auto-Tuner")
    print("=" * 60)
    print()
    
    # Quick evolutionary tuning
    best_weights, best_score = evolutionary_tune(
        generations=5,
        population_size=4,
        games_per_eval=4
    )
    
    print()
    print("=" * 60)
    print(f"Best score achieved: {best_score:.1%}")
    print("=" * 60)
    
    # Save best weights
    save_weights(best_weights, best_score)
    
    # Print weight changes
    original = load_weights()
    print("\nWeight changes from baseline:")
    for name in sorted(best_weights.keys()):
        if name in original and best_weights[name] != original[name]:
            print(f"  {name}: {original[name]} -> {best_weights[name]}")
    
    # Apply best weights permanently
    print("\nApplying best weights...")
    apply_weights(best_weights)
    
    return best_weights, best_score

if __name__ == '__main__':
    main()
