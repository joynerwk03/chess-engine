import chess
from chess_engine.eval_main import evaluate_board
from chess_engine.zobrist import compute_zobrist_hash

NEG_INF = -10**9
POS_INF = 10**9

# Transposition table entry: { 'score': int, 'depth': int, 'flag': 'EXACT'|'LOWERBOUND'|'UPPERBOUND' }
transposition_table: dict[int, dict] = {}

# MVV-LVA piece base values (centipawns)
_PV = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# Precompute MVV-LVA scores: victim_value * 10 - attacker_value
mvv_lva_scores = {
    (att, vic): (_PV[vic] * 10 - _PV[att])
    for att in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
    for vic in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
}


def evaluate_for_side_to_move(board: chess.Board) -> int:
    score = evaluate_board(board)
    return score if board.turn == chess.WHITE else -score


def _capture_score(board: chess.Board, move: chess.Move) -> int:
    # En passant victim is always a pawn and sits behind the to_square
    attacker = board.piece_type_at(move.from_square)
    if attacker is None:
        return 0
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        victim = board.piece_type_at(move.to_square)
        if victim is None:
            return 0
    return mvv_lva_scores.get((attacker, victim), 0)


def score_move(board: chess.Board, move: chess.Move, killer_moves, ply: int) -> int:
    # Captures score by MVV-LVA
    if board.is_capture(move):
        return 10_000 + _capture_score(board, move)  # ensure captures outrank any quiets
    # Quiet killers
    killers = None
    if killer_moves is not None and ply < len(killer_moves):
        killers = killer_moves[ply]
    if killers and (move == killers[0] or move == killers[1]):
        return 99
    return 0


def order_moves(board: chess.Board, moves, killer_moves, ply: int):
    return sorted(moves, key=lambda m: score_move(board, m, killer_moves, ply), reverse=True)


def quiescence_search(board: chess.Board, alpha: int, beta: int, ply: int = 0) -> int:
    # Stand pat
    stand_pat = evaluate_for_side_to_move(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Captures only, ordered by MVV-LVA (killer table not used in qsearch)
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    ordered = order_moves(board, captures, None, ply)
    for move in ordered:
        try:
            board.push(move)
            score = -quiescence_search(board, -beta, -alpha, ply + 1)
        finally:
            board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int, killer_moves) -> int:
    # Handle terminal positions immediately to avoid iterating an empty move list
    if board.is_game_over():
        return evaluate_for_side_to_move(board)

    board_hash = compute_zobrist_hash(board)
    tt_entry = transposition_table.get(board_hash)
    if tt_entry and tt_entry['depth'] >= depth:
        if tt_entry['flag'] == 'EXACT':
            return tt_entry['score']
        elif tt_entry['flag'] == 'LOWERBOUND':
            alpha = max(alpha, tt_entry['score'])
        elif tt_entry['flag'] == 'UPPERBOUND':
            beta = min(beta, tt_entry['score'])
        if alpha >= beta:
            return tt_entry['score']

    if depth == 0:
        return quiescence_search(board, alpha, beta, ply)

    best = NEG_INF
    best_move = None
    moves = list(board.legal_moves)
    moves = order_moves(board, moves, killer_moves, ply)
    
    in_check = board.is_check()
    move_count = 0

    for move in moves:
        move_count += 1
        is_capture = board.is_capture(move)
        
        # Late Move Reductions (LMR):
        # Reduce depth for late quiet moves that are unlikely to be best
        reduction = 0
        if (move_count > 3 and 
            depth >= 3 and 
            not is_capture and 
            not in_check and
            not board.gives_check(move)):
            # Reduce by 1 ply for late quiet moves
            reduction = 1
            if move_count > 6:
                reduction = 2  # Even more reduction for very late moves
        
        try:
            board.push(move)
            # Search with reduced depth first
            if reduction > 0:
                score = -negamax(board, depth - 1 - reduction, -beta, -alpha, ply + 1, killer_moves)
                # Re-search at full depth if it looks promising
                if score > alpha:
                    score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, killer_moves)
            else:
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, killer_moves)
        finally:
            board.pop()
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            # Killer heuristic: store quiet beta-cutoff moves
            if not board.is_capture(move) and killer_moves is not None and ply < len(killer_moves):
                k0, k1 = killer_moves[ply]
                if move != k0:
                    killer_moves[ply][1] = k0
                    killer_moves[ply][0] = move
            # Store LOWERBOUND in TT
            transposition_table[board_hash] = {
                'score': best,
                'depth': depth,
                'flag': 'LOWERBOUND',
            }
            return best

    # Store in TT after full search
    flag = 'EXACT'
    if best <= alpha:
        flag = 'UPPERBOUND'
    transposition_table[board_hash] = {
        'score': best,
        'depth': depth,
        'flag': flag,
        'best_move': best_move,
    }
    return best


def search(board: chess.Board, depth: int) -> chess.Move | None:
    best_move, _, _ = alpha_beta_search(board, depth)
    return best_move


def alpha_beta_search(board: chess.Board, depth: int) -> tuple[chess.Move | None, list[chess.Move], int]:
    """
    Main search entry point. Returns (best_move, principal_variation, score).
    Score is from the perspective of the side to move.
    """
    # Clear TT at the start of each new top-level search
    transposition_table.clear()

    best_move = None
    best_score = NEG_INF
    alpha, beta = NEG_INF, POS_INF

    moves = list(board.legal_moves)
    if not moves:
        return None, [], 0

    # Initialize killer moves (two killers per ply)
    MAX_PLY = 128
    killer_moves = [[None, None] for _ in range(MAX_PLY)]

    moves = order_moves(board, moves, killer_moves, 0)

    for move in moves:
        try:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha, 1, killer_moves)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score
    
    # Build PV from transposition table
    pv = []
    if best_move:
        pv = [best_move]
        temp_board = board.copy()
        temp_board.push(best_move)
        for _ in range(depth - 1):
            tt_hash = compute_zobrist_hash(temp_board)
            tt_entry = transposition_table.get(tt_hash)
            if tt_entry and tt_entry.get('best_move'):
                pv.append(tt_entry['best_move'])
                temp_board.push(tt_entry['best_move'])
            else:
                break
    
    return best_move, pv, best_score
