#!/usr/bin/env python3
"""
Analyze games to find weaknesses in our engine.
Plays games against Stockfish, then analyzes with full-strength Stockfish
to identify where evaluation diverged and mistakes were made.
"""
import chess
import chess.engine
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_engine.engine import Engine
from chess_engine.eval_main import evaluate_board, evaluate_board_detailed

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def play_and_record_game(our_depth: int, sf_elo: int, our_color: bool, max_moves: int = 60):
    """
    Play a game and record all positions and moves.
    Returns (result, move_history, position_history)
    """
    board = chess.Board()
    move_history = []
    position_history = [board.fen()]
    our_evals = [evaluate_board(board)]
    
    try:
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return 0, [], [], []
    
    moves = 0
    try:
        while not board.is_game_over() and moves < max_moves:
            if board.turn == our_color:
                engine = Engine(board.copy(), depth=our_depth)
                move = engine.get_best_move()
                if move is None:
                    break
            else:
                result = stockfish.play(board, chess.engine.Limit(time=0.05))
                move = result.move
            
            board.push(move)
            move_history.append(move)
            position_history.append(board.fen())
            our_evals.append(evaluate_board(board))
            moves += 1
    finally:
        stockfish.quit()
    
    # Determine result
    if board.is_checkmate():
        winner = not board.turn
        result = 1 if winner == our_color else -1
    elif board.is_game_over():
        result = 0
    else:
        # Material count for truncated games
        material = 0
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                  chess.ROOK: 5, chess.QUEEN: 9}
        for pt, val in values.items():
            material += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        
        if our_color == chess.WHITE:
            result = 1 if material > 2 else (-1 if material < -2 else 0)
        else:
            result = 1 if material < -2 else (-1 if material > 2 else 0)
    
    return result, move_history, position_history, our_evals


def analyze_with_stockfish(position_history: list, analysis_depth: int = 20):
    """
    Analyze each position with full-strength Stockfish.
    Returns list of (eval_cp, best_move) for each position.
    """
    sf_analysis = []
    
    try:
        stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        # Full strength - no ELO limit
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return []
    
    try:
        for i, fen in enumerate(position_history):
            board = chess.Board(fen)
            info = stockfish.analyse(board, chess.engine.Limit(depth=analysis_depth))
            
            score = info["score"].white()
            if score.is_mate():
                mate_in = score.mate()
                eval_cp = 10000 if mate_in > 0 else -10000
            else:
                eval_cp = score.score()
            
            best_move = info.get("pv", [None])[0]
            sf_analysis.append((eval_cp, best_move))
            
            if (i + 1) % 10 == 0:
                print(f"  Analyzed {i + 1}/{len(position_history)} positions...")
    finally:
        stockfish.quit()
    
    return sf_analysis


def find_mistakes(move_history, position_history, our_evals, sf_analysis, our_color):
    """
    Find positions where our engine made significant mistakes.
    A mistake is where:
    1. We played a move that caused SF eval to drop significantly
    2. Our eval didn't see the problem (eval divergence)
    """
    mistakes = []
    
    for i in range(len(move_history)):
        # Check if it was our move
        board = chess.Board(position_history[i])
        if board.turn != our_color:
            continue
        
        # Get eval before and after our move
        sf_before = sf_analysis[i][0]
        sf_after = sf_analysis[i + 1][0]
        our_eval_before = our_evals[i]
        
        # Normalize to our perspective
        if our_color == chess.BLACK:
            sf_before = -sf_before
            sf_after = -sf_after
        
        # Calculate eval drop (negative = we made things worse)
        eval_drop = sf_after - sf_before
        
        # Significant mistake: eval dropped by more than 100cp
        if eval_drop < -100:
            our_move = move_history[i]
            sf_best = sf_analysis[i][1]
            
            mistakes.append({
                'move_num': i + 1,
                'fen': position_history[i],
                'our_move': our_move,
                'sf_best_move': sf_best,
                'eval_drop': eval_drop,
                'sf_eval_before': sf_before,
                'sf_eval_after': sf_after,
                'our_eval': our_eval_before,
            })
    
    return mistakes


def analyze_mistake_position(mistake: dict):
    """
    Deep analysis of a mistake position to understand why our engine erred.
    """
    board = chess.Board(mistake['fen'])
    
    print(f"\n{'='*60}")
    print(f"Move {mistake['move_num']}: We played {mistake['our_move']}, SF recommends {mistake['sf_best_move']}")
    print(f"Eval drop: {mistake['eval_drop']} cp")
    print(f"SF eval before: {mistake['sf_eval_before']} cp, after: {mistake['sf_eval_after']} cp")
    print(f"Our eval: {mistake['our_eval']} cp")
    print(f"\nPosition:")
    print(board)
    
    # Get detailed breakdown of our evaluation
    breakdown = evaluate_board_detailed(board)
    print(f"\nOur evaluation breakdown:")
    factors = [(k, v[1]) for k, v in breakdown.items() if not k.startswith('_') and abs(v[1]) > 10]
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    for k, v in factors[:8]:
        print(f"  {k}: {v}")
    
    print(f"  Phase: {breakdown.get('_phase', 'N/A')}")
    
    # Analyze what SF sees that we don't
    print(f"\nDiagnosis:")
    eval_gap = abs(mistake['our_eval'] - mistake['sf_eval_before'])
    if eval_gap > 150:
        print(f"  ⚠️  Large eval gap ({eval_gap} cp) - we're misevaluating this position")
    
    # Check if it's an endgame
    phase = breakdown.get('_phase', 1.0)
    if phase < 0.3:
        print(f"  📊 Endgame position (phase={phase:.2f})")
    
    # Check material
    material = breakdown.get('Material', (0, 0))[1]
    if abs(material) > 200:
        print(f"  ♟️  Material imbalance: {material} cp")


def play_until_loss(our_depth: int = 3, sf_elo: int = 1320, max_games: int = 20):
    """
    Play games until we lose, then analyze that game.
    """
    print(f"Playing games until loss (depth={our_depth} vs SF {sf_elo})...")
    
    for game_num in range(max_games):
        our_color = game_num % 2 == 0
        color_str = "White" if our_color else "Black"
        
        print(f"\nGame {game_num + 1}: Playing as {color_str}...")
        result, moves, positions, our_evals = play_and_record_game(
            our_depth, sf_elo, our_color, max_moves=60
        )
        
        result_str = {1: "WIN", 0: "DRAW", -1: "LOSS"}[result]
        print(f"  Result: {result_str} in {len(moves)} moves")
        
        if result <= 0:  # Loss or draw
            print(f"\n{'#'*60}")
            print(f"ANALYZING {result_str} (Game {game_num + 1})")
            print(f"{'#'*60}")
            
            print("\nAnalyzing with full-strength Stockfish (depth 20)...")
            sf_analysis = analyze_with_stockfish(positions, analysis_depth=20)
            
            mistakes = find_mistakes(moves, positions, our_evals, sf_analysis, our_color)
            
            if mistakes:
                print(f"\nFound {len(mistakes)} significant mistakes:")
                for m in mistakes[:5]:  # Show top 5 mistakes
                    analyze_mistake_position(m)
            else:
                print("\nNo obvious mistakes found - might be gradual positional decline")
                # Show positions where eval diverged most
                print("\nPositions with largest eval divergence:")
                divergences = []
                for i in range(len(positions)):
                    sf_eval = sf_analysis[i][0]
                    our_eval = our_evals[i]
                    if our_color == chess.BLACK:
                        sf_eval = -sf_eval
                    gap = abs(sf_eval - our_eval)
                    divergences.append((i, gap, sf_eval, our_eval))
                
                divergences.sort(key=lambda x: x[1], reverse=True)
                for i, gap, sf_e, our_e in divergences[:5]:
                    print(f"  Move {i}: SF={sf_e}, Ours={our_e}, Gap={gap}")
                    board = chess.Board(positions[i])
                    phase = evaluate_board_detailed(board).get('_phase', 1.0)
                    print(f"    Phase: {phase:.2f}")
            
            return result, mistakes
    
    print(f"\nPlayed {max_games} games without losing!")
    return 1, []


if __name__ == "__main__":
    play_until_loss(our_depth=3, sf_elo=1320, max_games=10)
