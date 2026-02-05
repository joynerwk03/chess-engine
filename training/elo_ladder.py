#!/usr/bin/env python3
"""
ELO Ladder Test: Test against increasing Stockfish ELO levels.
This shows what ELO range our engine can consistently beat.
"""
import os
import sys
import chess
import chess.engine

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
        return 1.0 if (not board.turn) == our_color else 0.0
    elif board.is_game_over():
        return 0.5
    else:
        vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        mat = sum(v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) for p, v in vals.items())
        if our_color == chess.BLACK:
            mat = -mat
        return 1.0 if mat > 2 else (0.0 if mat < -2 else 0.5)

def test_elo(sf_elo, depth=3, num_games=6):
    """Test against a specific ELO"""
    total = 0.0
    for i in range(num_games):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        total += play_game(depth, sf_elo, color, max_moves=50)
    return total / num_games

def main():
    print("=" * 60)
    print("ELO LADDER TEST")
    print("Engine at depth 3 vs various Stockfish ELO levels")
    print("=" * 60)
    print()
    
    elo_levels = [1320, 1400, 1500, 1600, 1700, 1800]
    results = {}
    
    for elo in elo_levels:
        print(f"Testing vs ELO {elo}...", end=" ", flush=True)
        score = test_elo(elo, depth=3, num_games=6)
        results[elo] = score
        print(f"{score:.0%}")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for elo, score in results.items():
        bar = "█" * int(score * 20)
        status = "STRONG" if score >= 0.7 else ("OK" if score >= 0.5 else "WEAK")
        print(f"ELO {elo}: {bar:20s} {score:.0%} [{status}]")
    
    # Estimate our ELO
    # Assume 50% score = equal ELO
    estimated_elo = None
    for elo, score in sorted(results.items()):
        if score < 0.55:
            estimated_elo = elo
            break
    
    if estimated_elo:
        print(f"\nEstimated engine ELO: ~{estimated_elo}")
    else:
        print(f"\nEstimated engine ELO: >{max(elo_levels)}")

if __name__ == '__main__':
    main()
