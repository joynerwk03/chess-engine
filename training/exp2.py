#!/usr/bin/env python3
"""
More aggressive experiments - testing stronger weight combinations.
"""
import os
import sys
import chess
import chess.engine
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_game(depth, sf_elo, our_color, max_moves=50):
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
        return 'W' if (not board.turn) == our_color else 'L'
    elif board.is_game_over():
        return 'D'
    else:
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
        if our_color == chess.BLACK:
            mat = -mat
        return 'W' if mat > 2 else ('L' if mat < -2 else 'D')

def benchmark(depth=3, num_games=8, name=""):
    results = []
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result = play_game(depth, 1320, color, max_moves=55)
        results.append(result)
    
    wins = results.count('W')
    draws = results.count('D')
    losses = results.count('L')
    score = (wins + 0.5 * draws) / num_games
    print(f"{name:25s}: {wins}W-{draws}D-{losses}L = {score:.1%}")
    return score

def reload_all():
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
    for m in mods:
        if m in sys.modules:
            importlib.reload(sys.modules[m])

def set_weights(**kwargs):
    from chess_engine import tuned_weights
    for name, value in kwargs.items():
        setattr(tuned_weights, name, value)
    reload_all()

def main():
    print("=" * 60)
    print("AGGRESSIVE WEIGHT EXPERIMENTS")
    print("=" * 60)
    
    from chess_engine import tuned_weights
    
    # Save original values
    originals = {
        'HANGING_MINOR_PENALTY': tuned_weights.HANGING_MINOR_PENALTY,
        'HANGING_ROOK_PENALTY': tuned_weights.HANGING_ROOK_PENALTY,
        'HANGING_QUEEN_PENALTY': tuned_weights.HANGING_QUEEN_PENALTY,
        'ATTACKED_BY_LESSER_PENALTY': tuned_weights.ATTACKED_BY_LESSER_PENALTY,
        'THREAT_BONUS': tuned_weights.THREAT_BONUS,
        'BISHOP_PAIR_BONUS': tuned_weights.BISHOP_PAIR_BONUS,
        'TRAPPED_PIECE_PENALTY': tuned_weights.TRAPPED_PIECE_PENALTY,
    }
    
    results = {}
    
    # Baseline (current)
    print("\n[Testing current weights]")
    reload_all()
    results['current'] = benchmark(3, 8, "Current")
    
    # Even stronger hanging penalties
    print("\n[Exp: Even stronger hanging]")
    set_weights(
        HANGING_MINOR_PENALTY=-200,
        HANGING_ROOK_PENALTY=-180,
        HANGING_QUEEN_PENALTY=-300
    )
    results['stronger_hanging'] = benchmark(3, 8, "Stronger hanging")
    
    # Restore
    for k, v in originals.items():
        setattr(tuned_weights, k, v)
    
    # Stronger attacked_by_lesser
    print("\n[Exp: Stronger attacked by lesser]")
    set_weights(ATTACKED_BY_LESSER_PENALTY=-100)
    results['attacked_lesser'] = benchmark(3, 8, "Attacked by lesser")
    
    # Restore
    for k, v in originals.items():
        setattr(tuned_weights, k, v)
    
    # Stronger trapped
    print("\n[Exp: Stronger trapped penalty]")
    set_weights(TRAPPED_PIECE_PENALTY=-100)
    results['trapped'] = benchmark(3, 8, "Trapped penalty")
    
    # Restore
    for k, v in originals.items():
        setattr(tuned_weights, k, v)
    
    # Combo: Hanging + Attacked
    print("\n[Exp: Combined tactical]")
    set_weights(
        HANGING_MINOR_PENALTY=-180,
        ATTACKED_BY_LESSER_PENALTY=-80,
        TRAPPED_PIECE_PENALTY=-80
    )
    results['combo'] = benchmark(3, 8, "Combined tactical")
    
    # Restore
    for k, v in originals.items():
        setattr(tuned_weights, k, v)
    reload_all()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_name = max(results.keys(), key=lambda k: results[k])
    for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        marker = "** BEST **" if name == best_name else ""
        print(f"{name:25s}: {score:.1%} {marker}")

if __name__ == '__main__':
    main()
