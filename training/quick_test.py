#!/usr/bin/env python3
"""
Quick engine testing - plays a few fast games against Stockfish.
Designed to complete in under 2 minutes.
"""
import chess
import chess.engine
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_engine.engine import Engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

def play_game(our_depth: int, sf_elo: int, our_color: bool, max_moves: int = 40) -> tuple:
    """
    Play a quick game. Returns (result, moves_played).
    result: 1 = we win, 0 = draw, -1 = we lose
    """
    board = chess.Board()
    
    try:
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return 0, 0
    
    moves = 0
    try:
        while not board.is_game_over() and moves < max_moves:
            if board.turn == our_color:
                # Our engine's turn
                engine = Engine(board.copy(), depth=our_depth)
                move = engine.get_best_move()
                if move is None:
                    break
            else:
                # Stockfish's turn (very fast)
                result = stockfish.play(board, chess.engine.Limit(time=0.01))
                move = result.move
            
            board.push(move)
            moves += 1
    finally:
        stockfish.quit()
    
    if board.is_checkmate():
        winner = not board.turn  # The side that delivered mate
        return (1 if winner == our_color else -1), moves
    elif board.is_game_over():
        return 0, moves  # Draw
    else:
        # Game truncated - evaluate material
        material = 0
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                  chess.ROOK: 5, chess.QUEEN: 9}
        for pt, val in values.items():
            material += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        
        if our_color == chess.WHITE:
            return (1 if material > 2 else (-1 if material < -2 else 0)), moves
        else:
            return (1 if material < -2 else (-1 if material > 2 else 0)), moves


def quick_benchmark(our_depth: int = 2, sf_elo: int = 1320, num_games: int = 4, max_moves: int = 50):
    """Run a quick benchmark - 4 games should take ~1-2 minutes."""
    print(f"Quick Benchmark: depth={our_depth} vs Stockfish ELO {sf_elo}")
    print(f"Playing {num_games} games (alternating colors), max {max_moves} moves...")
    print()
    
    wins, draws, losses = 0, 0, 0
    total_time = 0
    
    for i in range(num_games):
        our_color = (i % 2 == 0)  # Alternate white/black
        color_str = "White" if our_color else "Black"
        
        start = time.time()
        result, moves = play_game(our_depth, sf_elo, our_color, max_moves=max_moves)
        elapsed = time.time() - start
        total_time += elapsed
        
        result_str = {1: "WIN", 0: "DRAW", -1: "LOSS"}[result]
        print(f"  Game {i+1}: {color_str}, {result_str} in {moves} moves ({elapsed:.1f}s)")
        
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
    
    print()
    print(f"Results: {wins}W-{draws}D-{losses}L")
    print(f"Win rate: {100*wins/num_games:.0f}%")
    print(f"Score: {100*(wins + 0.5*draws)/num_games:.0f}%")
    print(f"Total time: {total_time:.1f}s")
    
    return wins, draws, losses


if __name__ == "__main__":
    quick_benchmark(our_depth=2, sf_elo=1320, num_games=4)
