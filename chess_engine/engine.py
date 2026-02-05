import chess
from chess_engine.search import alpha_beta_search, iterative_deepening_search
from chess_engine.eval_main import evaluate_board_detailed, get_top_factors
import chess.polyglot
import random
import os


BOOK_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "books", "gm2001.bin")


def probe_opening_book(board: chess.Board) -> chess.Move | None:
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entries = list(reader.find_all(board))
            if not entries:
                return None
            moves = [e.move for e in entries]
            weights = [e.weight for e in entries]
            return random.choices(moves, weights=weights, k=1)[0]
    except Exception:
        return None


class Engine:
    def __init__(self, board: chess.Board, depth: int = 4, time_limit_ms: int = 0):
        """
        Initialize the chess engine.
        
        Args:
            board: The chess board
            depth: Fixed search depth (used if time_limit_ms is 0)
            time_limit_ms: Time limit in milliseconds (if > 0, uses iterative deepening)
        """
        self.board = board
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.last_pv = []  # Principal variation from last search
        self.last_score = 0  # Score from last search
        self.last_depth = 0  # Depth reached in last search
        self.last_eval_breakdown = {}  # Detailed evaluation of expected position
        self.used_book = False  # Whether last move came from opening book

    def get_best_move(self) -> chess.Move | None:
        # Try opening book first for variety and speed
        book_move = probe_opening_book(self.board)
        if book_move is not None and book_move in self.board.legal_moves:
            self.used_book = True
            self.last_pv = [book_move]
            self.last_score = 0
            self.last_depth = 0
            self.last_eval_breakdown = {}
            return book_move
        
        self.used_book = False
        # Fallback to search on a copy to avoid mutating caller state
        board_copy = self.board.copy(stack=False)
        
        if self.time_limit_ms > 0:
            # Use iterative deepening with time limit (convert ms to seconds)
            score, best_move, pv, depth_reached = iterative_deepening_search(board_copy, self.time_limit_ms / 1000.0)
            self.last_depth = depth_reached
            self.last_pv = pv if pv else ([best_move] if best_move else [])
        else:
            # Use fixed depth search
            best_move, pv, score = alpha_beta_search(board_copy, self.depth)
            self.last_pv = pv
            self.last_depth = self.depth
        
        # Convert score to always be from White's perspective
        # Search returns score from side-to-move's perspective
        self.last_score = score if self.board.turn == chess.WHITE else -score
        
        # Get detailed evaluation of the expected final position
        if self.last_pv:
            temp_board = self.board.copy()
            for move in self.last_pv:
                if move in temp_board.legal_moves:
                    temp_board.push(move)
            self.last_eval_breakdown = evaluate_board_detailed(temp_board)
        else:
            self.last_eval_breakdown = evaluate_board_detailed(self.board)
        
        return best_move
    
    def get_evaluation_summary(self, n_factors: int = 5) -> list:
        """
        Returns the top N most impactful evaluation factors from the last search.
        Each item is (factor_name, score) where positive favors White.
        """
        if not self.last_eval_breakdown:
            return []
        return get_top_factors(self.last_eval_breakdown, n_factors)
    
    def get_pv_string(self) -> str:
        """Returns the principal variation as a string of moves."""
        if not self.last_pv:
            return ""
        return " → ".join(move.uci() for move in self.last_pv)
    
    def set_depth(self, depth: int):
        """Set the search depth (for fixed depth mode)."""
        self.depth = max(1, min(depth, 8))  # Clamp between 1 and 8
        
    def set_time_limit(self, time_limit_ms: int):
        """Set the time limit in milliseconds (0 to use fixed depth instead)."""
        self.time_limit_ms = max(0, time_limit_ms)
