#!/usr/bin/env python3
"""
Engine testing and benchmarking framework.
Tests engine strength against Stockfish at various levels and against itself.
"""
import chess
import chess.engine
import time
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass
from chess_engine.engine import Engine

# Path to Stockfish - adjust if needed
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"  # macOS Homebrew
if not os.path.exists(STOCKFISH_PATH):
    STOCKFISH_PATH = "/usr/local/bin/stockfish"  # Alternative path
if not os.path.exists(STOCKFISH_PATH):
    STOCKFISH_PATH = "/usr/bin/stockfish"  # Linux default


@dataclass
class GameResult:
    winner: Optional[bool]  # True=White, False=Black, None=Draw
    moves: int
    reason: str
    white_time: float
    black_time: float


def play_game(
    white_engine,
    black_engine,
    max_moves: int = 100,
    white_is_stockfish: bool = False,
    black_is_stockfish: bool = False,
    stockfish_elo: int = 1000,
    stockfish_time: float = 0.01,
    our_depth: int = 3,
) -> GameResult:
    """Play a game between two engines."""
    board = chess.Board()
    white_time = 0.0
    black_time = 0.0
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        if board.turn == chess.WHITE:
            start = time.time()
            if white_is_stockfish:
                result = white_engine.play(board, chess.engine.Limit(time=stockfish_time))
                move = result.move
            else:
                white_engine.board = board.copy()
                white_engine.depth = our_depth
                move = white_engine.get_best_move()
            white_time += time.time() - start
        else:
            start = time.time()
            if black_is_stockfish:
                result = black_engine.play(board, chess.engine.Limit(time=stockfish_time))
                move = result.move
            else:
                black_engine.board = board.copy()
                black_engine.depth = our_depth
                move = black_engine.get_best_move()
            black_time += time.time() - start
        
        if move is None:
            break
        board.push(move)
    
    # Determine result
    if board.is_checkmate():
        winner = not board.turn  # The side that just moved won
        reason = "checkmate"
    elif board.is_stalemate():
        winner = None
        reason = "stalemate"
    elif board.is_insufficient_material():
        winner = None
        reason = "insufficient material"
    elif board.can_claim_threefold_repetition():
        winner = None
        reason = "threefold repetition"
    elif board.is_fifty_moves():
        winner = None
        reason = "fifty moves"
    elif move_num >= max_moves - 1:
        winner = None
        reason = "max moves reached"
    else:
        winner = None
        reason = "unknown"
    
    return GameResult(
        winner=winner,
        moves=len(board.move_stack),
        reason=reason,
        white_time=white_time,
        black_time=black_time,
    )


def test_vs_stockfish(
    num_games: int = 10,
    stockfish_elo: int = 1000,
    our_depth: int = 3,
    stockfish_time: float = 0.01,
) -> dict:
    """Test our engine against Stockfish at a given ELO."""
    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish not found at {STOCKFISH_PATH}")
        return {"error": "Stockfish not found"}
    
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    
    our_engine = Engine(chess.Board(), depth=our_depth)
    
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Testing vs Stockfish (ELO {stockfish_elo})...")
    
    for i in range(num_games):
        # Alternate colors
        we_are_white = (i % 2 == 0)
        
        if we_are_white:
            result = play_game(
                our_engine, stockfish,
                white_is_stockfish=False, black_is_stockfish=True,
                stockfish_elo=stockfish_elo, stockfish_time=stockfish_time,
                our_depth=our_depth
            )
            if result.winner is True:
                wins += 1
            elif result.winner is False:
                losses += 1
            else:
                draws += 1
        else:
            result = play_game(
                stockfish, our_engine,
                white_is_stockfish=True, black_is_stockfish=False,
                stockfish_elo=stockfish_elo, stockfish_time=stockfish_time,
                our_depth=our_depth
            )
            if result.winner is False:
                wins += 1
            elif result.winner is True:
                losses += 1
            else:
                draws += 1
        
        print(f"  Game {i+1}: {'Win' if (result.winner is True and we_are_white) or (result.winner is False and not we_are_white) else 'Loss' if (result.winner is False and we_are_white) or (result.winner is True and not we_are_white) else 'Draw'} ({result.reason}, {result.moves} moves)")
    
    stockfish.quit()
    
    score = wins + draws * 0.5
    total = num_games
    win_rate = score / total
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "total": total,
        "win_rate": win_rate,
        "stockfish_elo": stockfish_elo,
    }


def quick_strength_test(our_depth: int = 3, num_games: int = 6) -> dict:
    """Quick test against Stockfish at multiple ELO levels."""
    results = {}
    
    for elo in [800, 1000, 1200]:
        print(f"\n--- Testing vs Stockfish ELO {elo} ---")
        result = test_vs_stockfish(
            num_games=num_games,
            stockfish_elo=elo,
            our_depth=our_depth,
            stockfish_time=0.01,
        )
        if "error" not in result:
            results[elo] = result
            print(f"Result: {result['wins']}W-{result['losses']}L-{result['draws']}D ({result['win_rate']*100:.0f}%)")
    
    return results


def tactical_test() -> dict:
    """Test tactical puzzle solving."""
    puzzles = [
        # (FEN, best_move_uci, description)
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "h5f7", "Scholar's Mate"),
        ("r1b1k2r/ppppqppp/2n2n2/2b1p2Q/2B1P3/3P1N2/PPP2PPP/RNB1K2R w KQkq - 0 1", "h5f7", "Mate threat"),
        ("r2qk2r/ppp2ppp/2n1b3/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w KQkq - 0 1", "f3f7", "Fork"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1", "g8f6", "Development"),
        ("8/8/8/8/4k3/8/4P3/4K3 w - - 0 1", "e2e4", "Pawn push endgame"),
    ]
    
    correct = 0
    total = len(puzzles)
    
    print("\n--- Tactical Puzzle Test ---")
    for fen, expected, desc in puzzles:
        board = chess.Board(fen)
        engine = Engine(board, depth=4)
        move = engine.get_best_move()
        
        is_correct = move and move.uci() == expected
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"  {status} {desc}: {move.uci() if move else 'None'} (expected {expected})")
    
    return {"correct": correct, "total": total, "accuracy": correct / total}


if __name__ == "__main__":
    print("=" * 60)
    print("Chess Engine Benchmark")
    print("=" * 60)
    
    # Tactical test
    tactical = tactical_test()
    print(f"\nTactical: {tactical['correct']}/{tactical['total']} ({tactical['accuracy']*100:.0f}%)")
    
    # Strength test
    print("\n" + "=" * 60)
    strength = quick_strength_test(our_depth=3, num_games=4)
    
    print("\n" + "=" * 60)
    print("Summary:")
    for elo, result in strength.items():
        print(f"  vs ELO {elo}: {result['win_rate']*100:.0f}% score")
