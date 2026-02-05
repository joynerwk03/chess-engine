#!/usr/bin/env python3
"""Quick 6-game experiments"""
import os, sys, chess, chess.engine, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_game(depth, sf_elo, our_color):
    from chess_engine.engine import Engine
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    moves = 0
    while not board.is_game_over() and moves < 50:
        if board.turn == our_color:
            move = Engine(board.copy(), depth=depth).get_best_move()
        else:
            move = sf.play(board, chess.engine.Limit(time=0.02)).move
        if move is None: break
        board.push(move)
        moves += 1
    sf.quit()
    if board.is_checkmate():
        return 1.0 if (not board.turn) == our_color else 0.0
    elif board.is_game_over():
        return 0.5
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
    if our_color == chess.BLACK: mat = -mat
    return 1.0 if mat > 2 else (0.0 if mat < -2 else 0.5)

def test(n=6):
    return sum(play_game(3, 1320, chess.WHITE if i%2==0 else chess.BLACK) for i in range(n)) / n

def reload():
    for m in ['chess_engine.tuned_weights', 'chess_engine.eval_main', 'chess_engine.engine']:
        if m in sys.modules: importlib.reload(sys.modules[m])

from chess_engine import tuned_weights as tw

# Baseline
print("Baseline:", end=" ", flush=True)
reload()
base = test()
print(f"{base:.0%}")

# Exp 1: Higher rook bonus
print("Exp1 (Rook bonus +15):", end=" ", flush=True)
orig = tw.ROOK_ON_OPEN_FILE_BONUS
tw.ROOK_ON_OPEN_FILE_BONUS = 50
reload()
e1 = test()
print(f"{e1:.0%}")
tw.ROOK_ON_OPEN_FILE_BONUS = orig

# Exp 2: Higher bishop pair
print("Exp2 (Bishop pair +20):", end=" ", flush=True)
orig = tw.BISHOP_PAIR_BONUS
tw.BISHOP_PAIR_BONUS = 80
reload()
e2 = test()
print(f"{e2:.0%}")
tw.BISHOP_PAIR_BONUS = orig

# Exp 3: Bigger hanging queen
print("Exp3 (Hanging queen -300):", end=" ", flush=True)
orig = tw.HANGING_QUEEN_PENALTY
tw.HANGING_QUEEN_PENALTY = -300
reload()
e3 = test()
print(f"{e3:.0%}")
tw.HANGING_QUEEN_PENALTY = orig

reload()
print(f"\nBest: {max([('base',base),('rook',e1),('bishop',e2),('queen',e3)], key=lambda x:x[1])}")
