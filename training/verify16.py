#!/usr/bin/env python3
"""
Large verification: 16 games at depth 3.
"""
import os
import sys
import chess
import chess.engine
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_game(depth, sf_elo, our_color, max_moves=60):
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
        return ('W' if mat > 2 else ('L' if mat < -2 else 'D')), moves

def main():
    print("=" * 60)
    print("LARGE VERIFICATION: 16 games at depth 3 vs SF 1320")
    print("=" * 60)
    
    start = time.time()
    results = []
    
    for i in range(16):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        c = 'W' if color == chess.WHITE else 'B'
        result, moves = play_game(3, 1320, color, max_moves=60)
        results.append(result)
        print(f"Game {i+1:2d} ({c}): {result} in {moves} moves")
    
    elapsed = time.time() - start
    
    wins = results.count('W')
    draws = results.count('D')
    losses = results.count('L')
    score = (wins + 0.5 * draws) / 16 * 100
    
    print("=" * 60)
    print(f"Results: {wins}W-{draws}D-{losses}L = {score:.0f}%")
    print(f"Time: {elapsed:.0f}s ({elapsed/16:.1f}s per game)")
    print("=" * 60)
    
    if score >= 85:
        print("EXCELLENT! Engine is performing very well!")
    elif score >= 70:
        print("GOOD: Engine is strong")
    elif score >= 60:
        print("OK: Engine is competitive")
    else:
        print("NEEDS WORK: Performance below target")

if __name__ == '__main__':
    main()
