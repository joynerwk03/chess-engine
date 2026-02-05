import pygame
import chess
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_engine import Engine

# Simple Pygame chess GUI
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 620
BOARD_SIZE = 512
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = SCREEN_HEIGHT - BOARD_SIZE
EVAL_PANEL_WIDTH = SCREEN_WIDTH - BOARD_SIZE
WHITE_SQUARE = (238, 238, 210)
GREEN_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 51, 100)
PANEL_BG = (40, 40, 40)
PANEL_HEADER_BG = (60, 60, 60)
POSITIVE_COLOR = (100, 200, 100)  # Green for White advantage
NEGATIVE_COLOR = (200, 100, 100)  # Red for Black advantage
NEUTRAL_COLOR = (200, 200, 200)   # Gray for neutral


def load_piece_images():
    pieces = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    images = {}
    images_dir = os.path.join(os.path.dirname(__file__), 'assets', 'pieces')
    for piece in pieces:
        path = f"{images_dir}/{piece}.png"
        image = pygame.image.load(path)
        images[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
    return images


def draw_board(screen):
    for row in range(8):
        for col in range(8):
            color = WHITE_SQUARE if (row + col) % 2 == 0 else GREEN_SQUARE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_pieces(screen, board, images):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_str = f"{'b' if piece.color == chess.BLACK else 'w'}{piece.symbol().lower()}"
            row = 7 - chess.square_rank(square)
            col = chess.square_file(square)
            screen.blit(images[piece_str], (col * SQUARE_SIZE, row * SQUARE_SIZE))


def draw_highlight(screen, square):
    if square is not None:
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(HIGHLIGHT_COLOR)
        screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))


def draw_info_panel(screen, font, message, time_limit_ms):
    pygame.draw.rect(screen, PANEL_BG, (0, BOARD_SIZE, BOARD_SIZE, INFO_PANEL_HEIGHT))
    text_surface = font.render(message, True, (255, 255, 255))
    screen.blit(text_surface, (10, BOARD_SIZE + 10))
    
    # Draw time controls
    time_sec = time_limit_ms / 1000.0
    time_text = font.render(f"Time: {time_sec:.1f}s", True, (255, 255, 255))
    screen.blit(time_text, (10, BOARD_SIZE + 50))
    
    # Draw +/- buttons
    button_font = pygame.font.SysFont("Arial", 20, bold=True)
    
    # Minus button
    minus_rect = pygame.Rect(120, BOARD_SIZE + 45, 30, 30)
    pygame.draw.rect(screen, (80, 80, 80), minus_rect)
    pygame.draw.rect(screen, (150, 150, 150), minus_rect, 2)
    minus_text = button_font.render("-", True, (255, 255, 255))
    screen.blit(minus_text, (minus_rect.centerx - minus_text.get_width()//2, 
                              minus_rect.centery - minus_text.get_height()//2))
    
    # Plus button
    plus_rect = pygame.Rect(160, BOARD_SIZE + 45, 30, 30)
    pygame.draw.rect(screen, (80, 80, 80), plus_rect)
    pygame.draw.rect(screen, (150, 150, 150), plus_rect, 2)
    plus_text = button_font.render("+", True, (255, 255, 255))
    screen.blit(plus_text, (plus_rect.centerx - plus_text.get_width()//2, 
                             plus_rect.centery - plus_text.get_height()//2))
    
    # Hint text
    hint_font = pygame.font.SysFont("Arial", 14)
    hint_text = hint_font.render("(0.5s-5s per move)", True, (150, 150, 150))
    screen.blit(hint_text, (200, BOARD_SIZE + 55))
    
    return minus_rect, plus_rect


def draw_eval_panel(screen, font, engine, thinking=False):
    """Draw the evaluation breakdown panel on the right side."""
    panel_x = BOARD_SIZE
    panel_y = 0
    panel_width = EVAL_PANEL_WIDTH
    panel_height = SCREEN_HEIGHT
    
    # Background
    pygame.draw.rect(screen, PANEL_BG, (panel_x, panel_y, panel_width, panel_height))
    
    # Header
    header_font = pygame.font.SysFont("Arial", 18, bold=True)
    pygame.draw.rect(screen, PANEL_HEADER_BG, (panel_x, panel_y, panel_width, 40))
    header_text = header_font.render("Engine Analysis", True, (255, 255, 255))
    screen.blit(header_text, (panel_x + 10, 10))
    
    small_font = pygame.font.SysFont("Arial", 14)
    y_offset = 50
    
    if thinking:
        think_text = font.render("Thinking...", True, (255, 200, 100))
        screen.blit(think_text, (panel_x + 10, y_offset))
        return
    
    if engine.used_book:
        book_text = font.render("Opening Book", True, (100, 200, 255))
        screen.blit(book_text, (panel_x + 10, y_offset))
        y_offset += 30
        book_hint = small_font.render("(Using GM games)", True, (150, 150, 150))
        screen.blit(book_hint, (panel_x + 10, y_offset))
        return
    
    # Score display
    score = engine.last_score
    if abs(score) >= 100000:  # Mate score
        score_str = "Mate" if score > 0 else "-Mate"
        score_color = POSITIVE_COLOR if score > 0 else NEGATIVE_COLOR
    else:
        # Convert centipawns to pawns for display
        score_pawns = score / 100.0
        score_str = f"{score_pawns:+.2f}"
        if score > 50:
            score_color = POSITIVE_COLOR
        elif score < -50:
            score_color = NEGATIVE_COLOR
        else:
            score_color = NEUTRAL_COLOR
    
    score_label = small_font.render("Evaluation:", True, (200, 200, 200))
    screen.blit(score_label, (panel_x + 10, y_offset))
    score_text = font.render(score_str, True, score_color)
    screen.blit(score_text, (panel_x + 100, y_offset - 3))
    y_offset += 30
    
    # Depth display
    depth_label = small_font.render(f"Depth: {engine.last_depth}", True, (200, 200, 200))
    screen.blit(depth_label, (panel_x + 10, y_offset))
    y_offset += 25
    
    # Principal variation
    if engine.last_pv:
        pv_label = small_font.render("Expected line:", True, (200, 200, 200))
        screen.blit(pv_label, (panel_x + 10, y_offset))
        y_offset += 20
        
        # Show PV moves (wrapped if needed)
        pv_str = engine.get_pv_string()
        pv_font = pygame.font.SysFont("Courier", 12)
        
        # Split into chunks that fit
        words = pv_str.split(" → ")
        line = ""
        for i, word in enumerate(words[:6]):  # Limit to 6 moves shown
            test_line = line + ("→ " if line else "") + word
            if pv_font.size(test_line)[0] > panel_width - 20:
                if line:
                    pv_text = pv_font.render(line, True, (180, 180, 180))
                    screen.blit(pv_text, (panel_x + 10, y_offset))
                    y_offset += 16
                line = word
            else:
                line = test_line
        if line:
            pv_text = pv_font.render(line, True, (180, 180, 180))
            screen.blit(pv_text, (panel_x + 10, y_offset))
            y_offset += 16
        
        y_offset += 15
    
    # Divider
    pygame.draw.line(screen, (80, 80, 80), (panel_x + 10, y_offset), 
                     (panel_x + panel_width - 10, y_offset))
    y_offset += 15
    
    # Key factors header
    factors_header = small_font.render("Key Evaluation Factors:", True, (200, 200, 200))
    screen.blit(factors_header, (panel_x + 10, y_offset))
    y_offset += 25
    
    # Get top factors
    factors = engine.get_evaluation_summary(6)
    
    if not factors:
        no_data = small_font.render("No analysis yet", True, (150, 150, 150))
        screen.blit(no_data, (panel_x + 10, y_offset))
        return
    
    factor_font = pygame.font.SysFont("Arial", 13)
    bar_width = 80
    bar_height = 12
    
    for name, value in factors:
        # Factor name
        name_text = factor_font.render(name, True, (220, 220, 220))
        screen.blit(name_text, (panel_x + 10, y_offset))
        
        # Value and bar
        bar_x = panel_x + 130
        bar_y = y_offset + 2
        
        # Draw bar background
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
        
        # Draw value bar (centered, extends left for negative, right for positive)
        if value != 0:
            max_val = 500  # Scale factor for visualization
            bar_fill = min(abs(value) / max_val, 1.0) * (bar_width // 2)
            
            if value > 0:
                fill_color = POSITIVE_COLOR
                fill_rect = (bar_x + bar_width // 2, bar_y, int(bar_fill), bar_height)
            else:
                fill_color = NEGATIVE_COLOR
                fill_rect = (bar_x + bar_width // 2 - int(bar_fill), bar_y, int(bar_fill), bar_height)
            
            pygame.draw.rect(screen, fill_color, fill_rect)
        
        # Center line
        pygame.draw.line(screen, (150, 150, 150), 
                        (bar_x + bar_width // 2, bar_y), 
                        (bar_x + bar_width // 2, bar_y + bar_height))
        
        # Value text
        value_str = f"{value:+d}" if abs(value) < 10000 else ("+" if value > 0 else "-") + "∞"
        value_text = factor_font.render(value_str, True, 
                                        POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR if value < 0 else NEUTRAL_COLOR)
        screen.blit(value_text, (bar_x + bar_width + 10, y_offset))
        
        y_offset += 28
    
    # Legend at bottom
    y_offset = SCREEN_HEIGHT - 60
    pygame.draw.line(screen, (80, 80, 80), (panel_x + 10, y_offset - 10), 
                     (panel_x + panel_width - 10, y_offset - 10))
    
    legend_font = pygame.font.SysFont("Arial", 11)
    legend1 = legend_font.render("+ = White advantage", True, POSITIVE_COLOR)
    legend2 = legend_font.render("- = Black advantage", True, NEGATIVE_COLOR)
    legend3 = legend_font.render("Values in centipawns", True, (150, 150, 150))
    
    screen.blit(legend1, (panel_x + 10, y_offset))
    screen.blit(legend2, (panel_x + 10, y_offset + 15))
    screen.blit(legend3, (panel_x + 10, y_offset + 30))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Chess Bot")
    font = pygame.font.SysFont("Arial", 24)

    images = load_piece_images()
    board = chess.Board()

    player_color = None
    while player_color is None:
        choice = input("Choose your color (w/b): ").lower()
        if choice == 'w':
            player_color = chess.WHITE
        elif choice == 'b':
            player_color = chess.BLACK
        else:
            print("Invalid choice. Please enter 'w' for White or 'b' for Black.")

    selected_square = None
    time_limit_ms = 1000  # Default 1 second per move
    engine = Engine(board, time_limit_ms=time_limit_ms)
    message = "White's Turn"
    thinking = False
    minus_rect = None
    plus_rect = None

    # Time limit options in milliseconds
    TIME_OPTIONS = [500, 1000, 2000, 3000, 5000]  # 0.5s, 1s, 2s, 3s, 5s
    time_index = 1  # Start at 1 second

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                
                # Check time buttons
                if minus_rect and minus_rect.collidepoint(x, y):
                    if time_index > 0:
                        time_index -= 1
                        time_limit_ms = TIME_OPTIONS[time_index]
                        engine.set_time_limit(time_limit_ms)
                elif plus_rect and plus_rect.collidepoint(x, y):
                    if time_index < len(TIME_OPTIONS) - 1:
                        time_index += 1
                        time_limit_ms = TIME_OPTIONS[time_index]
                        engine.set_time_limit(time_limit_ms)
                
                # Handle board clicks for player moves
                elif board.turn == player_color and not board.is_game_over():
                    if y < BOARD_SIZE and x < BOARD_SIZE:
                        col = x // SQUARE_SIZE
                        row = y // SQUARE_SIZE
                        square = chess.square(col, 7 - row)

                        if selected_square is None:
                            piece = board.piece_at(square)
                            if piece and piece.color == player_color:
                                selected_square = square
                        else:
                            move = chess.Move(selected_square, square)
                            promo_move = chess.Move(selected_square, square, chess.QUEEN)

                            if move in board.legal_moves:
                                board.push(move)
                                selected_square = None
                            elif promo_move in board.legal_moves:
                                board.push(promo_move)
                                selected_square = None
                            else:
                                piece = board.piece_at(square)
                                if piece and piece.color == player_color:
                                    selected_square = square
                                else:
                                    selected_square = None

        # Draw current state before engine thinks (so "Thinking..." shows)
        draw_board(screen)
        draw_highlight(screen, selected_square)
        draw_pieces(screen, board, images)
        
        if board.turn != player_color and not board.is_game_over():
            thinking = True
            minus_rect, plus_rect = draw_info_panel(screen, font, message, time_limit_ms)
            draw_eval_panel(screen, font, engine, thinking=True)
            pygame.display.flip()
            
            best_move = engine.get_best_move()
            if best_move and best_move in board.legal_moves:
                board.push(best_move)
            thinking = False

        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            message = f"Checkmate! {winner} wins."
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves():
            message = "Draw."
        elif board.turn == chess.WHITE:
            message = "White's Turn"
        else:
            message = "Black's Turn"

        draw_board(screen)
        draw_highlight(screen, selected_square)
        draw_pieces(screen, board, images)
        minus_rect, plus_rect = draw_info_panel(screen, font, message, time_limit_ms)
        draw_eval_panel(screen, font, engine, thinking=False)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
