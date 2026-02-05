# Evaluation weights in centipawns
# Updated with v2 tuner: fixed material + learned positional weights

# Piece values (FIXED - not learned, these are well-established)
PIECE_VALUE_PAWN = 100
PIECE_VALUE_KNIGHT = 320
PIECE_VALUE_BISHOP = 330
PIECE_VALUE_ROOK = 500
PIECE_VALUE_QUEEN = 900
PIECE_VALUE_KING = 0

# Strategic bonuses (learned from v2 tuner - positional residuals)
BISHOP_PAIR_BONUS = 59      # Learned: 58.8 (classic ~50cp bonus confirmed)
ROOK_ON_OPEN_FILE_BONUS = 35  # Learned: 35.3
ROOK_ON_SEMI_OPEN_FILE_BONUS = 19  # Learned: 18.6

# Heuristic Weights (learned from v2 tuner)
KING_SHIELD_BONUS = 7       # Learned: 6.5
KING_OPEN_FILE_PENALTY = -65  # Learned: -65.4 (big penalty!)
KING_ATTACK_WEIGHT = 15     # King ring attackers weight (boosted from 10)
MOBILITY_BONUS = 5          # Learned: 5.0
ROOK_CONNECTIVITY_BONUS = 15
SPACE_BONUS = 2
TRAPPED_BISHOP_PENALTY = -50
TEMPO_BONUS = 38            # Learned: 37.5
KNIGHT_OUTPOST_BONUS = 15   # Keep reasonable default (not in v2 features)
UNCASTLED_KING_PENALTY = -25
CENTER_CONTROL_BONUS = 3    # Learned: 3.3
OUTSIDE_PASSED_PAWN_BONUS = 15

# ============================================
# NEW EVALUATION WEIGHTS (Learned from v2 tuner)
# ============================================

# --- Tactical: Hanging Pieces & Threats ---
HANGING_PIECE_PENALTY = -30  # Keep reasonable default
THREAT_BONUS = 50           # Keep reasonable default

# --- Rook Features ---
ROOK_ON_7TH_BONUS = 81      # Learned: 81.2 (very strong!)
CONNECTED_ROOKS_BONUS = 2   # Learned: 2.3

# --- Bad Bishop ---
BAD_BISHOP_SEVERE_PENALTY = -15  # Keep reasonable default
BAD_BISHOP_MODERATE_PENALTY = -8  # Keep reasonable default

# --- Fianchetto ---
FIANCHETTO_BONUS = 10       # Keep reasonable default

# --- Trapped Pieces ---
TRAPPED_PIECE_PENALTY = -100  # Increased from -50 (exp2 81.2%)
KNIGHT_ON_RIM_PENALTY = -15  # Keep reasonable default

# --- Pawn Structure ---
BACKWARD_PAWN_PENALTY = -10  # Keep reasonable default
PAWN_CHAIN_BONUS = 5        # Keep reasonable default

# --- Development ---
UNDEVELOPED_MINOR_PENALTY = -15  # Per undeveloped piece
EARLY_QUEEN_PENALTY = -20   # Queen out before minors

# --- King Safety Extensions ---
PAWN_STORM_BONUS = 5        # Keep reasonable default
KING_TROPISM_BONUS = 6      # Learned: 5.5

# --- Endgame Specific ---
KING_CENTRALIZATION_BONUS = 10  # Keep conservative (v2 learned negative?!)

# ============================================
# LEGACY WEIGHTS (kept for compatibility)
# ============================================
HANGING_PAWN_PENALTY = -30
HANGING_MINOR_PENALTY = -150  # Increased from -80 (exp1 93.8%)
HANGING_ROOK_PENALTY = -120
HANGING_QUEEN_PENALTY = -200
ATTACKED_BY_LESSER_PENALTY = -50
DISCOVERED_ATTACK_BONUS = 15
PIN_BONUS = 25
PINNED_PIECE_PENALTY = -20
DOUBLED_ROOKS_ON_7TH_BONUS = 80
BAD_BISHOP_PENALTY = -15
VERY_BAD_BISHOP_PENALTY = -30
KNIGHT_PAWN_BONUS = 3
BISHOP_OPEN_BONUS = 5
PAWN_CHAIN_BASE_ATTACK_BONUS = 10
PAWN_MAJORITY_BONUS = 15
PASSED_PAWN_BLOCKER_BONUS = 20
UNBLOCKED_PASSED_PAWN_PENALTY = -25
QUEEN_TROPISM_BONUS = 3
ROOK_TROPISM_BONUS = 2
MINOR_TROPISM_BONUS = 1
KING_ESCAPE_SQUARES_BONUS = 5
KING_NO_ESCAPE_PENALTY = -40
KING_OPPOSITION_BONUS = 20
RULE_OF_SQUARE_BONUS = 30

# ============================================
# END WEIGHTS
# ============================================

# Pawn structure parameters (learned from v2 tuner)
DOUBLED_PAWN_PENALTY = -20   # Learned: -19.6
ISOLATED_PAWN_PENALTY = -14  # Learned: -13.7
# Passed pawn bonus - exponentially increasing as pawn advances
# VERY aggressive for endgame conversion
PASSED_PAWN_BONUS = [0, 20, 40, 80, 140, 220, 350, 500]

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

# NOTE: Weights above are curated from v2 tuner results + chess knowledge.
# The old automatic override from learned_weights.py has been disabled
# because v1 tuner produced incorrect material values (pawn=157).
# To re-enable automatic weight loading, uncomment the code below.

# --- DISABLED: Old automatic weight loading ---
# try:
#     from training.learned_weights import FEATURE_WEIGHTS  # type: ignore
# except Exception:
#     FEATURE_WEIGHTS = None  # type: ignore
FEATURE_WEIGHTS = None  # Disabled - using curated weights above

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
    
    # Map new feature weights
    HANGING_PIECE_PENALTY = _apply_weight('hanging_pieces', HANGING_PIECE_PENALTY)
    THREAT_BONUS = _apply_weight('threats', THREAT_BONUS)
    ROOK_ON_7TH_BONUS = _apply_weight('rooks_on_7th', ROOK_ON_7TH_BONUS)
    CONNECTED_ROOKS_BONUS = _apply_weight('connected_rooks', CONNECTED_ROOKS_BONUS)
    BAD_BISHOP_SEVERE_PENALTY = _apply_weight('bad_bishop_severe', BAD_BISHOP_SEVERE_PENALTY)
    BAD_BISHOP_MODERATE_PENALTY = _apply_weight('bad_bishop_moderate', BAD_BISHOP_MODERATE_PENALTY)
    FIANCHETTO_BONUS = _apply_weight('fianchetto', FIANCHETTO_BONUS)
    TRAPPED_PIECE_PENALTY = _apply_weight('trapped_pieces', TRAPPED_PIECE_PENALTY)
    KNIGHT_ON_RIM_PENALTY = _apply_weight('knights_on_rim', KNIGHT_ON_RIM_PENALTY)
    BACKWARD_PAWN_PENALTY = _apply_weight('backward_pawns', BACKWARD_PAWN_PENALTY)
    PAWN_CHAIN_BONUS = _apply_weight('pawn_chains', PAWN_CHAIN_BONUS)
    UNDEVELOPED_MINOR_PENALTY = _apply_weight('undeveloped_pieces', UNDEVELOPED_MINOR_PENALTY)
    EARLY_QUEEN_PENALTY = _apply_weight('early_queen_out', EARLY_QUEEN_PENALTY)
    PAWN_STORM_BONUS = _apply_weight('pawn_storm', PAWN_STORM_BONUS)
    KING_TROPISM_BONUS = _apply_weight('king_tropism', KING_TROPISM_BONUS)
    KING_CENTRALIZATION_BONUS = _apply_weight('king_centralization', KING_CENTRALIZATION_BONUS)

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
