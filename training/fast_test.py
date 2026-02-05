#!/usr/bin/env python3
"""
Fast engine testing - plays quick games at depth 2.
Designed to complete in under 2 minutes for 12 games.
"""
import chess
import chess.engine
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_engine.engine import Engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

def play_fast_game(our_depth: int = 2, sf_elo: int = 1320, 
                   our_color: bool = True, max_moves: int = 40) -> int:
    """Play a fast game. Returns 1=win, 0=draw, -1=loss."""
    board = chess.Board()
    
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    
    moves = 0
    try:
        while not board.is_game_over() and moves < max_moves:
            if board.turn == our_color:
                move = Engine(board.copy(), depth=our_depth).get_best_move()
            else:
                move = sf.play(board, chess.engine.Limit(time=0.01)).move
            if move is None:
                break
            board.push(move)
            moves += 1
    finally:
        sf.quit()
    
    if board.is_checkmate():
        return 1 if (not board.turn) == our_color else -1
    elif board.is_game_over():
        return 0
    else:
        # Material evaluation for truncated games
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) 
                 for p, v in vals.items())
        if our_color == chess.BLACK:
            mat = -mat
        return 1 if mat > 2 else (-1 if mat < -2 else 0)


def fast_benchmark(n_games: int = 12, sf_elo: int = 1320, depth: int = 2):
    """Run fast benchmark - 12 games in ~90 seconds."""
    print(f"Fast Test: depth={depth} vs SF ELO {sf_elo}, {n_games} games")
    
    wins, draws, losses = 0, 0, 0
    start = time.time()
    
    for i in range(n_games):
        our_color = i % 2 == 0
        result = play_fast_game(depth, sf_elo, our_color, max_moves=40)
        
        if result == 1:
            wins += 1
            status = "W"
        elif result == 0:
            draws += 1
            status = "D"
        else:
            losses += 1
            status = "L"
        
        print(f"  Game {i+1}: {status}", end=" ", flush=True)
    
    elapsed = time.time() - start
    total = wins + draws + losses
    score = (wins + 0.5 * draws) / total if total > 0 else 0
    
    print(f"\n\nResults: {wins}W-{draws}D-{losses}L = {score*100:.0f}% in {elapsed:.0f}s")
    return score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--elo", type=int, default=1320)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    
    fast_benchmark(args.games, args.elo, args.depth)
