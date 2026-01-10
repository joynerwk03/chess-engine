import chess
from chess_engine.search import search as alpha_beta_search
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
    def __init__(self, board: chess.Board, depth: int = 3):
        self.board = board
        self.depth = depth

    def get_best_move(self) -> chess.Move | None:
        # Try opening book first for variety and speed
        book_move = probe_opening_book(self.board)
        if book_move is not None and book_move in self.board.legal_moves:
            return book_move
        # Fallback to search on a copy to avoid mutating caller state
        board_copy = self.board.copy(stack=False)
        return alpha_beta_search(board_copy, self.depth)
