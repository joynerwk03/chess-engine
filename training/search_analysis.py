#!/usr/bin/env python3
"""
Search speed analysis and optimization attempts.
"""
import os
import sys
import chess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_search_speed():
    """Measure search speed at different depths."""
    from chess_engine.engine import Engine
    from chess_engine.search import alpha_beta_search
    
    # Test positions of varying complexity
    positions = [
        ("Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
        ("Middle", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Complex", "r1bq1rk1/ppp2ppp/2n2n2/3p4/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 8"),
    ]
    
    print("=" * 70)
    print("SEARCH SPEED ANALYSIS")
    print("=" * 70)
    
    for name, fen in positions:
        print(f"\n{name} position:")
        board = chess.Board(fen)
        
        for depth in [2, 3, 4]:
            start = time.time()
            move, pv, score = alpha_beta_search(board.copy(), depth)
            elapsed = time.time() - start
            move_str = board.san(move) if move else "None"
            print(f"  Depth {depth}: {elapsed:.2f}s, move={move_str}, score={score}")

def profile_evaluation():
    """Profile which evaluation components are slowest."""
    from chess_engine.eval_main import evaluate_board_detailed
    
    board = chess.Board("r1bq1rk1/ppp2ppp/2n2n2/3p4/3PP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 0 8")
    
    print("\n" + "=" * 70)
    print("EVALUATION PROFILING")
    print("=" * 70)
    
    # Time full evaluation
    n = 500
    start = time.time()
    for _ in range(n):
        evaluate_board_detailed(board)
    total = time.time() - start
    print(f"Full evaluation: {n} evals in {total:.2f}s = {n/total:.0f} evals/sec")

def main():
    benchmark_search_speed()
    profile_evaluation()
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("1. If depth 4 is feasible (< 5s), use it for better play")
    print("2. If depth 3 is fast enough, focus on evaluation quality")
    print("3. Consider iterative deepening with time limits")

if __name__ == '__main__':
    main()
