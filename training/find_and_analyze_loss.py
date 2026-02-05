#!/usr/bin/env python3
"""
Find a loss and deeply analyze it with Stockfish to identify weaknesses.
"""
import chess
import chess.engine
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_engine.engine import Engine
from chess_engine.eval_main import evaluate_board

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

def play_game(our_depth, sf_elo, our_color, max_moves=60):
    """Play a game and record all positions."""
    board = chess.Board()
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    stockfish.configure({'UCI_LimitStrength': True, 'UCI_Elo': sf_elo})
    
    game_record = []
    moves = 0
    
    while not board.is_game_over() and moves < max_moves:
        our_eval = evaluate_board(board)
        
        if board.turn == our_color:
            engine = Engine(board.copy(), depth=our_depth)
            move = engine.get_best_move()
            player = 'US'
        else:
            result = stockfish.play(board, chess.engine.Limit(time=0.02))
            move = result.move
            player = 'SF'
        
        game_record.append({
            'ply': moves,
            'fen': board.fen(),
            'move': move.uci(),
            'player': player,
            'our_eval': our_eval,
        })
        
        board.push(move)
        moves += 1
    
    # Determine result
    if board.is_checkmate():
        winner = not board.turn
        result = 1 if winner == our_color else -1
    elif board.is_game_over():
        result = 0
    else:
        material = 0
        values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        for pt, val in values.items():
            material += val * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        if our_color == chess.WHITE:
            result = 1 if material > 2 else (-1 if material < -2 else 0)
        else:
            result = 1 if material < -2 else (-1 if material > 2 else 0)
    
    stockfish.quit()
    return result, game_record, board


def analyze_game_with_stockfish(game_record, our_color):
    """
    Analyze each position with strong Stockfish to find mistakes.
    Returns list of (ply, fen, our_move, best_move, our_eval, sf_eval, eval_loss)
    """
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    stockfish.configure({'UCI_LimitStrength': False})  # Full strength
    
    mistakes = []
    
    for i, rec in enumerate(game_record):
        if rec['player'] != 'US':
            continue
            
        board = chess.Board(rec['fen'])
        
        # Get Stockfish's analysis
        info = stockfish.analyse(board, chess.engine.Limit(depth=18))
        sf_eval = info['score'].white().score(mate_score=10000)
        best_move = info.get('pv', [None])[0]
        
        if sf_eval is None:
            continue
            
        # Adjust eval for our perspective
        if our_color == chess.BLACK:
            sf_eval = -sf_eval
            
        our_eval = rec['our_eval']
        if our_color == chess.BLACK:
            our_eval = -our_eval
        
        # Check what happened after our move
        if i + 1 < len(game_record):
            next_board = chess.Board(game_record[i+1]['fen'])
            next_info = stockfish.analyse(next_board, chess.engine.Limit(depth=18))
            next_sf_eval = next_info['score'].white().score(mate_score=10000)
            if next_sf_eval is not None:
                if our_color == chess.BLACK:
                    next_sf_eval = -next_sf_eval
                
                # Eval loss = how much worse position got after our move
                # (from our perspective)
                eval_loss = sf_eval - next_sf_eval
                
                if eval_loss > 50:  # Lost more than 50 cp
                    mistakes.append({
                        'ply': rec['ply'],
                        'fen': rec['fen'],
                        'our_move': rec['move'],
                        'best_move': best_move.uci() if best_move else '?',
                        'our_eval': rec['our_eval'],
                        'sf_eval_before': sf_eval,
                        'sf_eval_after': next_sf_eval,
                        'eval_loss': eval_loss,
                    })
    
    stockfish.quit()
    return mistakes


def main():
    print("=" * 70)
    print("LOSS FINDER AND ANALYZER")
    print("=" * 70)
    print()
    print("Playing games at depth 2 to quickly find losses...")
    print()
    
    for game_num in range(1, 30):
        our_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
        color_str = 'White' if our_color else 'Black'
        
        result, record, final_board = play_game(2, 1320, our_color, 60)
        result_str = {1: 'WIN', 0: 'DRAW', -1: 'LOSS'}[result]
        
        print(f"Game {game_num} ({color_str}): {result_str} in {len(record)} moves")
        
        if result == -1:
            print()
            print("=" * 70)
            print("LOSS FOUND! Analyzing with Stockfish depth 18...")
            print("=" * 70)
            print()
            
            mistakes = analyze_game_with_stockfish(record, our_color)
            
            if mistakes:
                print(f"Found {len(mistakes)} significant mistakes (>50cp loss):")
                print()
                
                for m in sorted(mistakes, key=lambda x: x['eval_loss'], reverse=True)[:10]:
                    print(f"Ply {m['ply']}: Lost {m['eval_loss']:.0f} cp")
                    print(f"  Position: {m['fen']}")
                    print(f"  Our move: {m['our_move']}")
                    print(f"  Best move: {m['best_move']}")
                    print(f"  Our eval: {m['our_eval']}, SF eval: {m['sf_eval_before']:.0f} -> {m['sf_eval_after']:.0f}")
                    print()
                
                # Deep dive into worst mistake
                worst = max(mistakes, key=lambda x: x['eval_loss'])
                print("=" * 70)
                print("DEEP ANALYSIS OF WORST MISTAKE")
                print("=" * 70)
                print()
                
                board = chess.Board(worst['fen'])
                print(board)
                print()
                print(f"FEN: {worst['fen']}")
                print(f"We played: {worst['our_move']}")
                print(f"Should have played: {worst['best_move']}")
                print(f"Eval loss: {worst['eval_loss']:.0f} cp")
                print()
                
                # Analyze what our engine thought
                print("What our engine evaluated:")
                print(f"  Our eval of position: {worst['our_eval']} cp")
                
                # Check if we saw the best move
                our_move_obj = chess.Move.from_uci(worst['our_move'])
                best_move_obj = chess.Move.from_uci(worst['best_move'])
                
                board_copy = board.copy()
                board_copy.push(our_move_obj)
                our_move_result_eval = evaluate_board(board_copy)
                
                board_copy = board.copy()
                board_copy.push(best_move_obj)
                best_move_result_eval = evaluate_board(board_copy)
                
                print(f"  After our move ({worst['our_move']}): {our_move_result_eval} cp")
                print(f"  After best move ({worst['best_move']}): {best_move_result_eval} cp")
                print()
                
                # This tells us if our evaluation is the problem or our search
                diff = our_move_result_eval - best_move_result_eval
                if our_color == chess.BLACK:
                    diff = -diff
                
                if abs(diff) < 30:
                    print("DIAGNOSIS: Evaluation doesn't distinguish moves well")
                    print("  -> Need better positional/tactical evaluation")
                else:
                    print("DIAGNOSIS: Evaluation sees difference, search didn't find it")
                    print("  -> May need deeper search or better move ordering")
                
            else:
                print("No significant mistakes found (all moves within 50cp of best)")
            
            break
    else:
        print("No losses found in 30 games!")


if __name__ == "__main__":
    main()
