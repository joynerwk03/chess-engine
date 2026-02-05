#!/usr/bin/env python3
"""
Generate endgame training positions using Stockfish evaluations.
Creates balanced endgame positions for training.
"""
import chess
import chess.engine
import random
import numpy as np
import os

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "endgame_data.npz")

# Endgame templates (piece configurations)
ENDGAME_TEMPLATES = [
    # King + Pawn endgames
    "8/8/8/8/4P3/4K3/8/4k3",  # K+P vs K
    "8/4p3/8/8/4P3/4K3/8/4k3",  # K+P vs K+P
    "8/3pp3/8/8/3PP3/4K3/8/4k3",  # K+2P vs K+2P
    
    # Rook endgames
    "8/8/8/8/4P3/4K3/8/r3k3",  # K+P vs K+R
    "4r3/8/8/8/4P3/4K3/8/4k3",  # K+P vs K+R (different)
    "R7/8/8/4k3/4p3/4K3/8/8",  # K+R vs K+P
    "8/8/8/r3k3/4p3/4K3/4R3/8",  # K+R+P vs K+R+P
    
    # Minor piece endgames
    "8/8/8/4k3/4p3/4K3/4B3/8",  # K+B vs K+P
    "8/8/4n3/4k3/8/4K3/4B3/8",  # K+B vs K+N
    "8/8/8/4k3/8/4K3/3NN3/8",  # K+2N vs K
    
    # Queen endgames
    "8/8/8/4k3/4p3/4K3/4Q3/8",  # K+Q vs K+P
    "q7/8/8/4k3/8/4K3/4Q3/8",  # K+Q vs K+Q
]

def generate_random_endgame():
    """Generate a random legal endgame position."""
    board = chess.Board(None)  # Empty board
    
    # Place kings (not adjacent)
    while True:
        wk = random.randint(0, 63)
        bk = random.randint(0, 63)
        wk_f, wk_r = chess.square_file(wk), chess.square_rank(wk)
        bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
        if abs(wk_f - bk_f) > 1 or abs(wk_r - bk_r) > 1:
            break
    
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    
    # Add 1-4 pawns for each side
    occupied = {wk, bk}
    for color in [chess.WHITE, chess.BLACK]:
        num_pawns = random.randint(1, 3)
        for _ in range(num_pawns):
            for attempt in range(20):
                # Pawns can't be on rank 1 or 8
                rank = random.randint(1, 6)
                file = random.randint(0, 7)
                sq = chess.square(file, rank)
                if sq not in occupied:
                    board.set_piece_at(sq, chess.Piece(chess.PAWN, color))
                    occupied.add(sq)
                    break
    
    # Maybe add a minor/major piece
    if random.random() < 0.5:
        piece_type = random.choice([chess.KNIGHT, chess.BISHOP, chess.ROOK])
        color = random.choice([chess.WHITE, chess.BLACK])
        for attempt in range(20):
            sq = random.randint(0, 63)
            if sq not in occupied:
                board.set_piece_at(sq, chess.Piece(piece_type, color))
                occupied.add(sq)
                break
    
    board.turn = random.choice([chess.WHITE, chess.BLACK])
    
    # Verify position is legal
    if board.is_valid() and not board.is_game_over():
        return board
    return None


def generate_endgame_data(num_positions: int = 5000):
    """Generate endgame positions with Stockfish evaluations."""
    print(f"Generating {num_positions} endgame positions...")
    
    try:
        sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Error starting Stockfish: {e}")
        return
    
    fens = []
    scores = []
    
    # From templates
    for template in ENDGAME_TEMPLATES:
        try:
            board = chess.Board(template)
            if board.is_valid() and not board.is_game_over():
                info = sf.analyse(board, chess.engine.Limit(depth=15))
                score = info["score"].white().score(mate_score=10000)
                if score is not None and abs(score) < 3000:
                    fens.append(template)
                    scores.append(score)
        except:
            pass
    
    print(f"  From templates: {len(fens)}")
    
    # Random endgames
    attempts = 0
    while len(fens) < num_positions and attempts < num_positions * 3:
        attempts += 1
        board = generate_random_endgame()
        if board is None:
            continue
        
        try:
            info = sf.analyse(board, chess.engine.Limit(depth=12))
            score = info["score"].white().score(mate_score=10000)
            if score is not None and abs(score) < 3000:
                fens.append(board.fen())
                scores.append(score)
                
                if len(fens) % 500 == 0:
                    print(f"  Generated: {len(fens)}")
        except:
            pass
    
    sf.quit()
    
    # Save
    np.savez(OUTPUT_PATH, 
             fens=np.array(fens, dtype=object),
             scores=np.array(scores, dtype=np.int16))
    
    print(f"Saved {len(fens)} endgame positions to {OUTPUT_PATH}")
    return fens, scores


if __name__ == "__main__":
    generate_endgame_data(2000)  # Generate 2000 positions (should be fast)
