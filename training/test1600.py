"""Test engine against Stockfish 1600 - 10 games, 1s per move."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.engine
from chess_engine.engine import Engine

NUM_GAMES = 10
TIME_LIMIT_MS = 1000
MAX_MOVES = 60  # Limit moves per game for speed
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'

def play_game(sf_elo, our_color, max_moves=60):
    board = chess.Board()
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    
    moves = 0
    while not board.is_game_over() and moves < max_moves:
        if board.turn == our_color:
            eng = Engine(board.copy(), time_limit_ms=TIME_LIMIT_MS)
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
    print("=" * 50)
    print("TEST: 10 games vs SF 1600 @ 1s/move")
    print("=" * 50)
    
    results = []
    for i in range(NUM_GAMES):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        c = 'W' if color == chess.WHITE else 'B'
        result, moves = play_game(1600, color, MAX_MOVES)
        results.append(result)
        print(f"Game {i+1:2d} ({c}): {result} in {moves} moves")
    
    wins = results.count('W')
    draws = results.count('D')
    losses = results.count('L')
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS vs SF 1600 ({TIME_LIMIT_MS}ms/move):")
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")
    print(f"Win rate: {wins}/{NUM_GAMES} = {100*wins/NUM_GAMES:.0f}%")
    print(f"Score: {wins + draws*0.5}/{NUM_GAMES} = {100*(wins + draws*0.5)/NUM_GAMES:.0f}%")

if __name__ == "__main__":
    main()
