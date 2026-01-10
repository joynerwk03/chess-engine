import chess
import random

random.seed(13)

# Zobrist hashing setup for future scalability
PIECE_KEYS = {
    (piece, color, sq): random.getrandbits(64)
    for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    for color in [chess.WHITE, chess.BLACK]
    for sq in chess.SQUARES
}
SIDE_TO_MOVE_KEY = random.getrandbits(64)

# Optional extra domains for completeness (not required by prompt)
CASTLING_KEYS = {
    chess.BB_A1: random.getrandbits(64),
    chess.BB_H1: random.getrandbits(64),
    chess.BB_A8: random.getrandbits(64),
    chess.BB_H8: random.getrandbits(64),
}
EN_PASSANT_KEYS = {file_idx: random.getrandbits(64) for file_idx in range(8)}


def compute_hash(board: chess.Board) -> int:
    h = 0
    for sq, piece in board.piece_map().items():
        h ^= PIECE_KEYS[(piece.piece_type, piece.color, sq)]
    if board.turn == chess.BLACK:
        h ^= SIDE_TO_MOVE_KEY

    # Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= CASTLING_KEYS[chess.BB_H1]
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= CASTLING_KEYS[chess.BB_A1]
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= CASTLING_KEYS[chess.BB_H8]
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= CASTLING_KEYS[chess.BB_A8]

    ep = board.ep_square
    if ep is not None:
        h ^= EN_PASSANT_KEYS[chess.square_file(ep)]
    return h


def compute_zobrist_hash(board: chess.Board) -> int:
    # Alias for clarity with prompt naming
    return compute_hash(board)
