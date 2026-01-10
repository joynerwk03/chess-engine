#!/usr/bin/env python3
"""
Data generation for chess engine tuning.
Reads a large PGN, filters for high-quality checkmate games, splits by game (train/val/test),
extracts quiet middlegame positions, evaluates with Stockfish, and saves (FEN, score) pairs.
"""
from __future__ import annotations

import os
import sys
import platform
import random
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    import chess
    import chess.pgn
    import chess.engine
except Exception as e:
    print("Missing dependency. Please install python-chess, numpy, and tqdm: pip install python-chess numpy tqdm", file=sys.stderr)
    raise

# =============================
# Configuration (edit as needed)
# =============================
# Path relative to project root (run from project root directory)
PGN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lichess_games.pgn")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")  # directory to write train_data.npz, etc.
RANDOM_SEED = 42

# Filtering criteria
MIN_ELO = 2200
REQUIRE_TERMINATION_NORMAL = True
REQUIRE_CHECKMATE_END = True

# Position extraction
SKIP_FIRST_FULL_MOVES = 5          # skip first N full moves (i.e., 2*N plies)
MIN_PLY_SKIP = SKIP_FIRST_FULL_MOVES * 2
MIDDLEGAME_MIN_NON_PAWN_PIECES = 6  # count of non-pawn, non-king pieces total (both sides)

# Stockfish analysis
EVAL_TIME_PER_POS = 0.1  # seconds per position
MAX_ABS_CP = 2000        # clamp evaluations to this range (centipawns)
STOCKFISH_PATH_OVERRIDE = None  # set to a path string to skip auto-detection
STOCKFISH_THREADS = max(1, os.cpu_count() or 1)
STOCKFISH_HASH_MB = 128

# Optional caps to avoid huge output/compute (set to None for unlimited)
MAX_GAMES_PER_SET: Optional[int] = None    # max games to process per split
MAX_POSITIONS_PER_GAME: Optional[int] = None  # max quiet positions taken from each game

# Train/Val/Test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


@dataclass
class FilteredGame:
    moves: List[str]
    starting_fen: Optional[str] = None


# =============================
# Helpers
# =============================

def find_stockfish() -> str:
    """Locate Stockfish engine binary.

    Checks:
    - STOCKFISH_PATH_OVERRIDE
    - $STOCKFISH_PATH env
    - PATH via shutil.which('stockfish')
    - Common macOS and Linux locations
    - Common Windows locations
    """
    candidates: List[str] = []

    if STOCKFISH_PATH_OVERRIDE:
        candidates.append(STOCKFISH_PATH_OVERRIDE)

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        candidates.append(env_path)

    which_path = shutil.which("stockfish")
    if which_path:
        candidates.append(which_path)

    system = platform.system().lower()
    if system == "darwin" or system == "linux":
        candidates.extend([
            "/opt/homebrew/bin/stockfish",  # Apple Silicon Homebrew
            "/usr/local/bin/stockfish",     # Intel macOS / some Linux
            "/usr/bin/stockfish",
        ])
    if system == "windows":
        program_files = os.environ.get("ProgramFiles", r"C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)")
        candidates.extend([
            os.path.join(program_files, "Stockfish", "stockfish.exe"),
            os.path.join(program_files, "Stockfish", "bin", "stockfish.exe"),
            os.path.join(program_files_x86, "Stockfish", "stockfish.exe"),
        ])

    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand

    msg = (
        "Could not find Stockfish executable. Set STOCKFISH_PATH_OVERRIDE in this script, "
        "or set the STOCKFISH_PATH environment variable, or install Stockfish and ensure it is in PATH.\n"
        "Example macOS path: /opt/homebrew/bin/stockfish\n"
        "Example Windows path: C\\\Program Files\\Stockfish\\bin\\stockfish.exe"
    )
    print(msg, file=sys.stderr)
    sys.exit(1)


def is_middlegame(board: chess.Board) -> bool:
    # Count non-pawn, non-king pieces
    count = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        count += len(board.pieces(piece_type, chess.WHITE))
        count += len(board.pieces(piece_type, chess.BLACK))
    return count >= MIDDLEGAME_MIN_NON_PAWN_PIECES


def has_promotion_move(board: chess.Board) -> bool:
    # Fast check: promotions can only occur when a pawn reaches last rank
    # We still need to scan legal moves to see if any is a promotion
    for move in board.legal_moves:
        if move.promotion is not None:
            return True
    return False


def has_checking_move(board: chess.Board) -> bool:
    # Prefer dedicated generator if available
    gen_checks = getattr(board, "generate_legal_checks", None)
    if callable(gen_checks):
        for _ in gen_checks():
            return True
        return False
    # Fallback: test each legal move
    for move in board.legal_moves:
        if board.gives_check(move):
            return True
    return False


def is_quiet_position(board: chess.Board) -> bool:
    if board.is_check():
        return False
    # Any captures available?
    for _ in board.generate_legal_captures():
        return False
    # Any checking moves available?
    if has_checking_move(board):
        return False
    # Any promotions available?
    if has_promotion_move(board):
        return False
    return True


def read_and_filter_games(pgn_path: str) -> Tuple[List[FilteredGame], int]:
    filtered: List[FilteredGame] = []
    total = 0

    if not os.path.isfile(pgn_path):
        print(f"PGN not found: {pgn_path}", file=sys.stderr)
        sys.exit(1)

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(desc="Scanning games", unit="game")
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total += 1

            headers = game.headers
            try:
                white_elo = int(headers.get("WhiteElo", "0") or 0)
                black_elo = int(headers.get("BlackElo", "0") or 0)
            except ValueError:
                white_elo = 0
                black_elo = 0

            if white_elo < MIN_ELO or black_elo < MIN_ELO:
                pbar.update(1)
                continue

            result = headers.get("Result", "*")
            if result not in {"1-0", "0-1"}:
                pbar.update(1)
                continue

            term = (headers.get("Termination", "") or "").strip().lower()
            if REQUIRE_TERMINATION_NORMAL and term != "normal":
                pbar.update(1)
                continue

            # Build board (support non-standard starting positions)
            starting_fen = headers.get("FEN")
            board = chess.Board(fen=starting_fen) if starting_fen else chess.Board()

            moves: List[str] = []
            for mv in game.mainline_moves():
                moves.append(mv.uci())
                board.push(mv)

            if REQUIRE_CHECKMATE_END and not board.is_checkmate():
                pbar.update(1)
                continue

            filtered.append(FilteredGame(moves=moves, starting_fen=starting_fen))
            pbar.update(1)
        pbar.close()

    return filtered, total


def split_games(games: List[FilteredGame]) -> Tuple[List[FilteredGame], List[FilteredGame], List[FilteredGame]]:
    n = len(games)
    if n == 0:
        return [], [], []
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)
    train = games[:train_end]
    val = games[train_end:val_end]
    test = games[val_end:]
    return train, val, test


def evaluate_position(engine: chess.engine.SimpleEngine, board: chess.Board) -> Optional[int]:
    try:
        info = engine.analyse(board, chess.engine.Limit(time=EVAL_TIME_PER_POS))
        score = info.get("score")
        if score is None:
            return None
        s_white = score.white()
        if s_white.is_mate():
            return None
        cp = s_white.score()
        if cp is None:
            return None
        # Clamp
        if cp > MAX_ABS_CP:
            cp = MAX_ABS_CP
        elif cp < -MAX_ABS_CP:
            cp = -MAX_ABS_CP
        return int(cp)
    except chess.engine.EngineTerminatedError:
        return None
    except Exception:
        return None


def extract_positions(
    engine: chess.engine.SimpleEngine,
    games: List[FilteredGame],
    desc: str,
) -> Tuple[List[str], List[int]]:
    fens: List[str] = []
    scores: List[int] = []

    games_iter = games
    if MAX_GAMES_PER_SET is not None:
        games_iter = games[:MAX_GAMES_PER_SET]

    for g in tqdm(games_iter, desc=desc, unit="game"):
        board = chess.Board(fen=g.starting_fen) if g.starting_fen else chess.Board()
        positions_taken = 0
        for ply_idx, uci in enumerate(g.moves):
            # Play the move
            move = chess.Move.from_uci(uci)
            board.push(move)

            if ply_idx + 1 < MIN_PLY_SKIP:
                continue
            if not is_middlegame(board):
                continue
            if not is_quiet_position(board):
                continue

            cp = evaluate_position(engine, board)
            if cp is None:
                continue

            fens.append(board.fen())
            scores.append(cp)
            positions_taken += 1

            if MAX_POSITIONS_PER_GAME is not None and positions_taken >= MAX_POSITIONS_PER_GAME:
                break

    return fens, scores


def generate_labeled_data() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    stockfish_path = find_stockfish()

    print("Reading and filtering games...")
    filtered_games, total_scanned = read_and_filter_games(PGN_PATH)
    print(f"Scanned {total_scanned} games. Filtered: {len(filtered_games)}")

    # Shuffle and split (game-level)
    random.shuffle(filtered_games)
    train_games, val_games, test_games = split_games(filtered_games)
    print(
        f"Split -> train: {len(train_games)}, val: {len(val_games)}, test: {len(test_games)}"
    )

    # Engine options
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        try:
            engine.configure({"Threads": STOCKFISH_THREADS, "Hash": STOCKFISH_HASH_MB})
        except Exception:
            pass

        # Extract positions per split
        train_fens, train_scores = extract_positions(engine, train_games, desc="Extracting train")
        val_fens, val_scores = extract_positions(engine, val_games, desc="Extracting val")
        test_fens, test_scores = extract_positions(engine, test_games, desc="Extracting test")

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save compressed NPZ files
    def save_npz(name: str, fens: List[str], scores: List[int]) -> None:
        path = os.path.join(OUTPUT_DIR, name)
        # Cast to compact dtypes
        fens_arr = np.array(fens, dtype=object)
        scores_arr = np.array(scores, dtype=np.int16)
        np.savez_compressed(path, fens=fens_arr, scores=scores_arr)
        print(f"Saved {len(fens)} samples -> {path}")

    save_npz("train_data.npz", train_fens, train_scores)
    save_npz("val_data.npz", val_fens, val_scores)
    save_npz("test_data.npz", test_fens, test_scores)


if __name__ == "__main__":
    generate_labeled_data()
