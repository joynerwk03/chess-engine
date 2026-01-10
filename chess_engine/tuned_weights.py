# Initial, reasonable evaluation weights in centipawns
# Piece values
PIECE_VALUE_PAWN = 100
PIECE_VALUE_KNIGHT = 320
PIECE_VALUE_BISHOP = 330
PIECE_VALUE_ROOK = 500
PIECE_VALUE_QUEEN = 900
PIECE_VALUE_KING = 0

# Strategic bonuses
BISHOP_PAIR_BONUS = 30
ROOK_ON_OPEN_FILE_BONUS = 25
ROOK_ON_SEMI_OPEN_FILE_BONUS = 15

# Heuristic Weights
KING_SHIELD_BONUS = 5
KING_OPEN_FILE_PENALTY = -10
KING_ATTACK_WEIGHT = 2  # Multiplier for threat score near the king
MOBILITY_BONUS = 1
ROOK_CONNECTIVITY_BONUS = 15
SPACE_BONUS = 2
TRAPPED_BISHOP_PENALTY = -50
TEMPO_BONUS = 10
KNIGHT_OUTPOST_BONUS = 30
UNCASTLED_KING_PENALTY = -25
CENTER_CONTROL_BONUS = 10
OUTSIDE_PASSED_PAWN_BONUS = 15

# Pawn structure parameters
DOUBLED_PAWN_PENALTY = -10
ISOLATED_PAWN_PENALTY = -10
# Bonus increases as pawn advances (index by rank from White's perspective)
PASSED_PAWN_BONUS = [0, 5, 10, 20, 35, 60, 100, 200]

# Phase-specific Piece-Square Tables (single source of truth)
# Middlegame PSTs (MG)
PST_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     50,  50,  50,  50,  50,  50,  50,  50,
     10,  10,  20,  30,  30,  20,  10,  10,
      5,  10,  25,  50,  50,  25,  10,   5,
      5,   5,  10,  25,  25,  10,   5,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_KNIGHT_MG = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
PST_BISHOP_MG = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
PST_ROOK_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10,  10,  10,  10,  10,   5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      0,   0,   0,   5,   5,   0,   0,   0,
]
PST_QUEEN_MG = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -10,   0,   5,   5,   5,   5,   0, -10,
     -5,   0,   5,   5,   5,   5,   0,  -5,
      0,   0,   5,   5,   5,   5,   0,  -5,
    -10,   5,   5,   5,   5,   5,   0, -10,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
PST_KING_MG = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
     20,  20,   0,   0,   0,   0,  20,  20,
     20,  30,  10,   0,   0,  10,  30,  20,
]

# Endgame PSTs (EG)
PST_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     10,  10,  10,  20,  20,  10,  10,  10,
     10,  15,  20,  30,  30,  20,  15,  10,
     15,  20,  30,  40,  40,  30,  20,  15,
     20,  30,  40,  60,  60,  40,  30,  20,
     25,  35,  50,  70,  70,  50,  35,  25,
     30,  50,  70,  90,  90,  70,  50,  30,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_KNIGHT_EG = [
    -40, -30, -20, -20, -20, -20, -30, -40,
    -30, -10,   0,   0,   0,   0, -10, -30,
    -20,   0,  10,  15,  15,  10,   0, -20,
    -20,   0,  15,  20,  20,  15,   0, -20,
    -20,   0,  15,  20,  20,  15,   0, -20,
    -20,   0,  10,  15,  15,  10,   0, -20,
    -30, -10,   0,   5,   5,   0, -10, -30,
    -40, -30, -20, -20, -20, -20, -30, -40,
]
PST_BISHOP_EG = [
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
     -5,   0,   0,   5,   5,   0,   0,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   5,  10,  15,  15,  10,   5,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
    -10,  -5,   0,   0,   0,   0,  -5, -10,
    -10, -10, -10, -10, -10, -10, -10, -10,
]
PST_ROOK_EG = [
      0,   0,   5,  10,  10,   5,   0,   0,
      0,   0,   5,  10,  10,   5,   0,   0,
      0,   0,  10,  15,  15,  10,   0,   0,
      0,   5,  10,  15,  15,  10,   5,   0,
      0,   5,  10,  15,  15,  10,   5,   0,
      0,   0,  10,  15,  15,  10,   0,   0,
      0,   0,   5,  10,  10,   5,   0,   0,
      0,   0,   5,  10,  10,   5,   0,   0,
]
PST_QUEEN_EG = [
    -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   0,  10,  15,  15,  10,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
    -10,  -5,   0,   0,   0,   0,  -5, -10,
    -10, -10, -10, -10, -10, -10, -10, -10,
]
PST_KING_EG = [
    -50, -30, -10,   0,   0, -10, -30, -50,
    -30, -10,  10,  20,  20,  10, -10, -30,
    -10,  10,  20,  30,  30,  20,  10, -10,
      0,  20,  30,  40,  40,  30,  20,   0,
      0,  20,  30,  40,  40,  30,  20,   0,
    -10,  10,  20,  30,  30,  20,  10, -10,
    -30, -10,  10,  20,  20,  10, -10, -30,
    -50, -30, -10,   0,   0, -10, -30, -50,
]

# Optional override from learned linear model (if present)
try:
    from learned_weights import FEATURE_WEIGHTS  # type: ignore
except Exception:
    FEATURE_WEIGHTS = None  # type: ignore

if FEATURE_WEIGHTS:
    import chess  # used to parse square names

    def _apply_weight(name: str, current: int) -> int:
        w = FEATURE_WEIGHTS.get(name)
        if w is None:
            return current
        try:
            return int(round(float(w)))
        except Exception:
            return current

    # Map material weights
    PIECE_VALUE_PAWN = _apply_weight('material_PAWN', PIECE_VALUE_PAWN)
    PIECE_VALUE_KNIGHT = _apply_weight('material_KNIGHT', PIECE_VALUE_KNIGHT)
    PIECE_VALUE_BISHOP = _apply_weight('material_BISHOP', PIECE_VALUE_BISHOP)
    PIECE_VALUE_ROOK = _apply_weight('material_ROOK', PIECE_VALUE_ROOK)
    PIECE_VALUE_QUEEN = _apply_weight('material_QUEEN', PIECE_VALUE_QUEEN)
    # KING remains 0

    # Map heuristic weights (only those used by evaluation)
    BISHOP_PAIR_BONUS = _apply_weight('bishop_pair', BISHOP_PAIR_BONUS)
    ROOK_ON_OPEN_FILE_BONUS = _apply_weight('rooks_open_file', ROOK_ON_OPEN_FILE_BONUS)
    ROOK_ON_SEMI_OPEN_FILE_BONUS = _apply_weight('rooks_semi_open_file', ROOK_ON_SEMI_OPEN_FILE_BONUS)
    DOUBLED_PAWN_PENALTY = _apply_weight('doubled_pawns', DOUBLED_PAWN_PENALTY)
    ISOLATED_PAWN_PENALTY = _apply_weight('isolated_pawns', ISOLATED_PAWN_PENALTY)
    OUTSIDE_PASSED_PAWN_BONUS = _apply_weight('outside_passed_pawns', OUTSIDE_PASSED_PAWN_BONUS)
    KNIGHT_OUTPOST_BONUS = _apply_weight('knight_outposts', KNIGHT_OUTPOST_BONUS)
    KING_SHIELD_BONUS = _apply_weight('king_pawn_shield', KING_SHIELD_BONUS)
    KING_OPEN_FILE_PENALTY = _apply_weight('king_open_file', KING_OPEN_FILE_PENALTY)
    MOBILITY_BONUS = _apply_weight('mobility', MOBILITY_BONUS)
    SPACE_BONUS = _apply_weight('space', SPACE_BONUS)
    CENTER_CONTROL_BONUS = _apply_weight('center_control', CENTER_CONTROL_BONUS)
    UNCASTLED_KING_PENALTY = _apply_weight('uncastled', UNCASTLED_KING_PENALTY)
    TEMPO_BONUS = _apply_weight('tempo', TEMPO_BONUS)

    # Update PST tables from learned weights
    def _set_pst_from_weights(prefix: str, table_map: dict[str, list[int]]) -> None:
        for key, w in FEATURE_WEIGHTS.items():
            if not key.startswith(prefix):
                continue
            parts = key.split('_')
            if len(parts) != 4:
                continue
            _, phase, piece_letter, square_name = parts
            tbl = table_map.get(piece_letter)
            if tbl is None:
                continue
            try:
                idx = chess.parse_square(square_name.lower())
            except Exception:
                continue
            try:
                tbl[idx] = int(round(float(w)))
            except Exception:
                pass

    mg_map = {'P': PST_PAWN_MG, 'N': PST_KNIGHT_MG, 'B': PST_BISHOP_MG, 'R': PST_ROOK_MG, 'Q': PST_QUEEN_MG, 'K': PST_KING_MG}
    eg_map = {'P': PST_PAWN_EG, 'N': PST_KNIGHT_EG, 'B': PST_BISHOP_EG, 'R': PST_ROOK_EG, 'Q': PST_QUEEN_EG, 'K': PST_KING_EG}
    _set_pst_from_weights('PST_MG_', mg_map)
    _set_pst_from_weights('PST_EG_', eg_map)
