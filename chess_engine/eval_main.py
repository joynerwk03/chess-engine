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
# New feature imports
from chess_engine.evaluation.features_tactics import get_tactical_score
from chess_engine.evaluation.features_rooks import get_rook_features_score
from chess_engine.evaluation.features_bishops import get_bishop_features_score
from chess_engine.evaluation.features_piece_activity import get_piece_activity_score
from chess_engine.evaluation.features_development import get_development_score
from chess_engine.evaluation.features_endgame import get_endgame_score
from chess_engine.tuned_weights import (
    BISHOP_PAIR_BONUS,
    ROOK_ON_OPEN_FILE_BONUS,
    ROOK_ON_SEMI_OPEN_FILE_BONUS,
    TEMPO_BONUS,
)

MATE_SCORE = 1_000_000


def evaluate_board_detailed(board: chess.Board) -> dict:
    """
    Returns a detailed breakdown of all evaluation components.
    Keys are human-readable feature names, values are (raw_score, blended_score) tuples.
    The blended_score accounts for game phase weighting.
    """
    breakdown = {}
    
    # Terminal states
    if board.is_checkmate():
        score = -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
        return {"Checkmate": (score, score), "_total": score}
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return {"Draw": (0, 0), "_total": 0}

    # Phase for blending
    phase = calculate_game_phase(board)
    
    # Base material (phase-independent)
    material = get_material_score(board)
    breakdown["Material"] = (material, material)

    # PSTs for MG/EG
    mg_pst, eg_pst = get_pst_score(board)
    pst_blended = int(round(phase * mg_pst + (1.0 - phase) * eg_pst))
    breakdown["Piece Placement"] = (mg_pst, pst_blended)

    # Phase-independent features
    bishop_pair = get_bishop_pair_score(board)
    breakdown["Bishop Pair"] = (bishop_pair, bishop_pair)
    
    rook_files = get_rook_placement_score(board)
    breakdown["Rook on Open Files"] = (rook_files, rook_files)
    
    pawn_struct = get_pawn_structure_score(board)
    breakdown["Pawn Structure"] = (pawn_struct, pawn_struct)

    # Middlegame-only strategic heuristics (scaled by phase)
    mg_knight_outposts = get_knight_outpost_score(board)
    breakdown["Knight Outposts"] = (mg_knight_outposts, int(round(phase * mg_knight_outposts)))
    
    mg_king_safety = get_king_safety_score(board)
    breakdown["King Safety"] = (mg_king_safety, int(round(phase * mg_king_safety)))
    
    mg_mobility = get_mobility_score(board)
    breakdown["Mobility"] = (mg_mobility, int(round(phase * mg_mobility)))
    
    mg_space = get_space_score(board)
    breakdown["Space Control"] = (mg_space, int(round(phase * mg_space)))
    
    mg_uncastled = get_uncastled_king_penalty(board, phase)
    breakdown["King Castling"] = (mg_uncastled, int(round(phase * mg_uncastled)))
    
    mg_center = get_center_control_score(board)
    breakdown["Center Control"] = (mg_center, int(round(phase * mg_center)))

    # NEW FEATURES
    # Tactical features (phase-independent - always important)
    tactical = get_tactical_score(board)
    breakdown["Tactics"] = (tactical, tactical)
    
    # Rook features (7th rank, connected)
    rook_features = get_rook_features_score(board)
    breakdown["Rook Activity"] = (rook_features, rook_features)
    
    # Bishop features (bad bishop, fianchetto)
    bishop_features = get_bishop_features_score(board)
    breakdown["Bishop Quality"] = (bishop_features, bishop_features)
    
    # Piece activity (trapped pieces, coordination)
    piece_activity = get_piece_activity_score(board)
    breakdown["Piece Activity"] = (piece_activity, piece_activity)
    
    # Development (opening only, scaled by phase)
    development = get_development_score(board)
    breakdown["Development"] = (development, int(round(phase * development)))
    
    # Endgame features (scaled inversely by phase)
    endgame = get_endgame_score(board, phase)
    breakdown["Endgame"] = (endgame, endgame)

    # Tempo bonus
    tempo = TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS
    breakdown["Tempo"] = (tempo, tempo)

    # Calculate total
    total = sum(v[1] for k, v in breakdown.items() if not k.startswith("_"))
    breakdown["_total"] = total
    breakdown["_phase"] = phase
    
    return breakdown


def get_top_factors(breakdown: dict, n: int = 5) -> list:
    """
    Returns the top N most impactful factors (by absolute blended value).
    Returns list of (name, blended_score) tuples, sorted by |score| descending.
    """
    factors = [(k, v[1]) for k, v in breakdown.items() 
               if not k.startswith("_") and v[1] != 0]
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    return factors[:n]


def evaluate_board(board: chess.Board) -> int:
    """
    Fast evaluation for search - prioritizes speed over accuracy.
    Returns score from White's perspective.
    """
    # Terminal states
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # Core features only
    material = get_material_score(board)
    mg_pst, eg_pst = get_pst_score(board)
    bishop_pair = get_bishop_pair_score(board)
    rook_files = get_rook_placement_score(board)
    
    # Phase for blending
    phase = calculate_game_phase(board)
    
    # Blend PST scores
    pst_blended = phase * mg_pst + (1.0 - phase) * eg_pst

    # Total
    total = material + pst_blended + bishop_pair + rook_files

    # Tempo
    total += TEMPO_BONUS if board.turn == chess.WHITE else -TEMPO_BONUS

    return int(round(total))
