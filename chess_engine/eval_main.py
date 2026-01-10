import chess
from chess_engine.evaluation.features_material import get_material_score
from chess_engine.evaluation.features_pst import get_pst_score
from chess_engine.evaluation.features_special import (
    get_bishop_pair_score,
    get_rook_placement_score,
)
from chess_engine.evaluation.features_game_phase import calculate_game_phase
from chess_engine.evaluation.features_pawn import get_pawn_structure_score
from chess_engine.evaluation.features_knights import get_knight_outpost_score
from chess_engine.evaluation.features_king_safety import get_king_safety_score
from chess_engine.evaluation.features_space import get_mobility_score, get_space_score
from chess_engine.evaluation.features_control import get_uncastled_king_penalty, get_center_control_score
from chess_engine.tuned_weights import (
    BISHOP_PAIR_BONUS,
    ROOK_ON_OPEN_FILE_BONUS,
    ROOK_ON_SEMI_OPEN_FILE_BONUS,
    TEMPO_BONUS,
)

MATE_SCORE = 1_000_000


def evaluate_board(board: chess.Board) -> int:
    # Terminal states
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0

    # Base material
    material = get_material_score(board)

    # PSTs for MG/EG
    mg_pst, eg_pst = get_pst_score(board)

    # Phase for MG-only feature gating
    phase = calculate_game_phase(board)

    # Phase-independent
    bishop_pair = get_bishop_pair_score(board)
    rook_files = get_rook_placement_score(board)
    pawn_struct = get_pawn_structure_score(board)

    # Middlegame-only strategic heuristics
    mg_knight_outposts = get_knight_outpost_score(board)
    mg_king_safety = get_king_safety_score(board)
    mg_mobility = get_mobility_score(board)
    mg_space = get_space_score(board)
    mg_uncastled = get_uncastled_king_penalty(board, phase)
    mg_center = get_center_control_score(board)

    mg_total = (
        material + mg_pst + bishop_pair + rook_files + pawn_struct
        + mg_knight_outposts + mg_king_safety + mg_mobility + mg_space
        + mg_uncastled + mg_center
    )
    eg_total = material + eg_pst + bishop_pair + rook_files + pawn_struct

    blended = phase * mg_total + (1.0 - phase) * eg_total

    # Tempo bonus for side to move
    blended += TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS

    return int(round(blended))
