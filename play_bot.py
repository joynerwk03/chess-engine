import pygame
import chess
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chess_engine import Engine

# Simple Pygame chess GUI
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 650
BOARD_SIZE = 512
SQUARE_SIZE = BOARD_SIZE // 8
INFO_PANEL_HEIGHT = SCREEN_HEIGHT - BOARD_SIZE
WHITE_SQUARE = (238, 238, 210)
GREEN_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 51, 100)


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


def draw_info_panel(screen, font, message):
    pygame.draw.rect(screen, (40, 40, 40), (0, BOARD_SIZE, SCREEN_WIDTH, INFO_PANEL_HEIGHT))
    text_surface = font.render(message, True, (255, 255, 255))
    screen.blit(text_surface, (10, BOARD_SIZE + 10))


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
    engine = Engine(board, depth=3)
    message = "White's Turn"

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if board.turn == player_color and not board.is_game_over():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    if y < BOARD_SIZE:
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

        if board.turn != player_color and not board.is_game_over():
            best_move = engine.get_best_move()
            if best_move and best_move in board.legal_moves:
                board.push(best_move)

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
        draw_info_panel(screen, font, message)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
