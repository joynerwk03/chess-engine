import chess
from chess_engine.tuned_weights import (
    PST_PAWN_MG,
    PST_KNIGHT_MG,
    PST_BISHOP_MG,
    PST_ROOK_MG,
    PST_QUEEN_MG,
    PST_KING_MG,
    PST_PAWN_EG,
    PST_KNIGHT_EG,
    PST_BISHOP_EG,
    PST_ROOK_EG,
    PST_QUEEN_EG,
    PST_KING_EG,
)

PST_MAP_MG = {
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK: PST_ROOK_MG,
    chess.QUEEN: PST_QUEEN_MG,
    chess.KING: PST_KING_MG,
}
PST_MAP_EG = {
    chess.PAWN: PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT_EG,
    chess.BISHOP: PST_BISHOP_EG,
    chess.ROOK: PST_ROOK_EG,
    chess.QUEEN: PST_QUEEN_EG,
    chess.KING: PST_KING_EG,
}


def get_pst_score(board: chess.Board) -> tuple[int, int]:
    mg = 0
    eg = 0
    for sq, piece in board.piece_map().items():
        mg_pst = PST_MAP_MG[piece.piece_type]
        eg_pst = PST_MAP_EG[piece.piece_type]
        if piece.color == chess.WHITE:
            mg += mg_pst[sq]
            eg += eg_pst[sq]
        else:
            mg -= mg_pst[chess.square_mirror(sq)]
            eg -= eg_pst[chess.square_mirror(sq)]
    return mg, eg
