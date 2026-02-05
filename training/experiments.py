#!/usr/bin/env python3
"""
Focused improvement experiments - test specific strategic changes.
Each experiment modifies one aspect and measures impact.
"""
import os
import sys
import chess
import chess.engine
import importlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_game(depth, sf_elo, our_color, max_moves=50):
    """Play one game"""
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
            result = sf.play(board, chess.engine.Limit(time=0.02))
            move = result.move
        if move is None:
            break
        board.push(move)
        moves += 1
    
    sf.quit()
    
    if board.is_checkmate():
        return 'W' if (not board.turn) == our_color else 'L', moves
    elif board.is_game_over():
        return 'D', moves
    else:
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
        if our_color == chess.BLACK:
            mat = -mat
        if mat > 2:
            return 'W', moves
        elif mat < -2:
            return 'L', moves
        return 'D', moves

def benchmark(depth=3, num_games=8, name=""):
    """Run benchmark"""
    results = []
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result, moves = play_game(depth, 1320, color, max_moves=60)
        results.append(result)
        c = 'W' if color == chess.WHITE else 'B'
        print(f"  {name} Game {i+1} ({c}): {result}")
    
    wins = results.count('W')
    draws = results.count('D')
    losses = results.count('L')
    score = (wins + 0.5 * draws) / num_games
    print(f"  {name} Result: {wins}W-{draws}D-{losses}L = {score:.1%}")
    return score

def reload_all():
    """Force reload all modules"""
    mods = [
        'chess_engine.tuned_weights',
        'chess_engine.evaluation.features_tactics',
        'chess_engine.evaluation.features_king_safety', 
        'chess_engine.evaluation.features_pawn',
        'chess_engine.evaluation.features_piece_activity',
        'chess_engine.evaluation.features_material',
        'chess_engine.evaluation.features_control',
        'chess_engine.evaluation.features_special',
        'chess_engine.eval_main',
        'chess_engine.search',
        'chess_engine.engine',
    ]
    for m in mods:
        if m in sys.modules:
            importlib.reload(sys.modules[m])

def run_experiments():
    """Run focused experiments"""
    print("=" * 60)
    print("FOCUSED IMPROVEMENT EXPERIMENTS")
    print("=" * 60)
    
    # Baseline
    print("\n[BASELINE]")
    reload_all()
    baseline = benchmark(depth=3, num_games=8, name="Base")
    
    results = {"baseline": baseline}
    
    # Experiment 1: Increase tactical weights
    print("\n[EXP 1: Higher tactical weights]")
    from chess_engine import tuned_weights
    original_threat = tuned_weights.THREAT_BONUS
    original_hanging = tuned_weights.HANGING_MINOR_PENALTY
    
    tuned_weights.THREAT_BONUS = 50
    tuned_weights.HANGING_MINOR_PENALTY = -150
    reload_all()
    
    exp1 = benchmark(depth=3, num_games=8, name="Exp1")
    results["exp1_tactics"] = exp1
    
    # Restore
    tuned_weights.THREAT_BONUS = original_threat
    tuned_weights.HANGING_MINOR_PENALTY = original_hanging
    
    # Experiment 2: Stronger king safety
    print("\n[EXP 2: Stronger king safety]")
    original_shield = tuned_weights.KING_SHIELD_BONUS
    original_open = tuned_weights.KING_OPEN_FILE_PENALTY
    
    tuned_weights.KING_SHIELD_BONUS = 15
    tuned_weights.KING_OPEN_FILE_PENALTY = -90
    reload_all()
    
    exp2 = benchmark(depth=3, num_games=8, name="Exp2")
    results["exp2_king_safety"] = exp2
    
    # Restore
    tuned_weights.KING_SHIELD_BONUS = original_shield
    tuned_weights.KING_OPEN_FILE_PENALTY = original_open
    
    # Experiment 3: Higher mobility weights
    print("\n[EXP 3: Higher mobility]")
    original_mobility = tuned_weights.MOBILITY_BONUS
    original_trapped = tuned_weights.TRAPPED_PIECE_PENALTY
    
    tuned_weights.MOBILITY_BONUS = 8
    tuned_weights.TRAPPED_PIECE_PENALTY = -80
    reload_all()
    
    exp3 = benchmark(depth=3, num_games=8, name="Exp3")
    results["exp3_mobility"] = exp3
    
    # Restore
    tuned_weights.MOBILITY_BONUS = original_mobility
    tuned_weights.TRAPPED_PIECE_PENALTY = original_trapped
    
    # Report
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, score in results.items():
        diff = score - baseline
        marker = "**BEST**" if score == max(results.values()) else ""
        print(f"{name:20s}: {score:.1%} ({diff:+.1%}) {marker}")
    
    # Find best and apply if better than baseline
    best_name = max(results.items(), key=lambda x: x[1])[0]
    best_score = results[best_name]
    
    if best_score > baseline and best_name != "baseline":
        print(f"\n>>> Best experiment: {best_name} ({best_score:.1%})")
        print(">>> Apply these changes permanently in tuned_weights.py")
    else:
        print("\n>>> Baseline is best, no changes recommended")
    
    return results

if __name__ == '__main__':
    run_experiments()
