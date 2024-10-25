import pygame
import sys
import random
import torch
from dqn_agent import DQNAgent

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // 3
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = SQUARE_SIZE // 4

# Colors
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

# Delay duration in milliseconds
AGENT_MOVE_DELAY = 500
COMPUTER_MOVE_DELAY = 500

# Timing variables
waiting_for_agent_move = False
agent_move_time = 0
waiting_for_computer_move = False
computer_move_time = 0

# Player types
PLAYER_HUMAN = 'human'
PLAYER_COMPUTER = 'computer'
PLAYER_AI = 'ai'

PLAYER_TYPES = {
    '1': PLAYER_HUMAN,
    '2': PLAYER_COMPUTER,
    '3': PLAYER_AI
}

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Tic Tac Toe')
screen.fill(BG_COLOR)

# Board
board = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
game_over = False
font = pygame.font.SysFont(None, 40)

def draw_lines():
    # Horizontal lines
    pygame.draw.line(screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH)
    # Vertical lines
    pygame.draw.line(screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH)
    pygame.draw.line(screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH)

def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 'O':
                pygame.draw.circle(screen, CIRCLE_COLOR, 
                                   (int(col * SQUARE_SIZE + SQUARE_SIZE//2), 
                                    int(row * SQUARE_SIZE + SQUARE_SIZE//2)), 
                                   CIRCLE_RADIUS, CIRCLE_WIDTH)
            elif board[row][col] == 'X':
                # Draw two lines for X
                start_desc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE)
                end_desc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE)
                pygame.draw.line(screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH)
                
                start_asc = (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE)
                end_asc = (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE)
                pygame.draw.line(screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH)

def get_pos(mouse_pos):
    x, y = mouse_pos
    row = y // SQUARE_SIZE
    col = x // SQUARE_SIZE
    return row, col

def check_win(player):
    # Check rows
    for row in board:
        if all([cell == player for cell in row]):
            return True
    # Check columns
    for col in range(BOARD_COLS):
        if all([board[row][col] == player for row in range(BOARD_ROWS)]):
            return True
    # Check diagonals
    if all([board[i][i] == player for i in range(BOARD_ROWS)]):
        return True
    if all([board[i][BOARD_COLS - i - 1] == player for i in range(BOARD_ROWS)]):
        return True
    return False

def check_draw():
    for row in board:
        if None in row:
            return False
    return True

def restart():
    screen.fill(BG_COLOR)
    draw_lines()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = None
    global game_over, waiting_for_agent_move, agent_move_time, waiting_for_computer_move, computer_move_time
    game_over = False
    waiting_for_agent_move = False
    agent_move_time = 0
    waiting_for_computer_move = False
    computer_move_time = 0

def display_message(message):
    text = font.render(message, True, (255, 255, 255))
    rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
    screen.blit(text, rect)

def encode_board(board):
    """
    Encode the board state into a flattened numerical vector.
    0: Empty, 1: 'X', 2: 'O'
    """
    return [1 if cell == 'X' else 2 if cell == 'O' else 0 for row in board for cell in row]

def get_available_actions(board):
    """
    Returns a list of available action indices (0-8) based on the current board.
    """
    available = []
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] is None:
                action = row * BOARD_COLS + col
                available.append(action)
    return available

def get_current_player_num():
    """
    Determines the current player number based on the board state.
    
    Returns:
        int: 1 for Player 1's turn, 2 for Player 2's turn.
    """
    flat_board = [cell for row in board for cell in row]
    x_count = flat_board.count('X')
    o_count = flat_board.count('O')
    return 1 if x_count <= o_count else 2

def is_player_turn(player_num):
    """
    Checks if it's the specified player's turn.
    
    Args:
        player_num (int): The player number (1 or 2).
    
    Returns:
        bool: True if it's the player's turn, False otherwise.
    """
    current_player = get_current_player_num()
    return current_player == player_num

def agent_move(agent, player_num):
    """
    Makes a move for the AI agent ('X' or 'O') using the trained model.
    
    Args:
        agent (DQNAgent): The trained AI agent.
        player_num (int): The player number (1 or 2).
    """
    state = encode_board(board)
    available_actions = get_available_actions(board)
    if not available_actions:
        return
    action = agent.select_action(state, available_actions)
    row, col = divmod(action, BOARD_COLS)
    symbol = 'X' if player_num == 1 else 'O'
    board[row][col] = symbol
    if check_win(symbol):
        global game_over
        game_over = True
        display_message(f"Player {player_num} ({symbol}) wins! Click to restart.")
    elif check_draw():
        game_over = True
        display_message('Draw! Click to restart.')

def computer_move(player_num):
    """
    Makes a random move for the computer ('X' or 'O').
    
    Args:
        player_num (int): The player number (1 or 2).
    """
    available_actions = get_available_actions(board)
    if not available_actions:
        return
    action = random.choice(available_actions)
    row, col = divmod(action, BOARD_COLS)
    symbol = 'X' if player_num == 1 else 'O'
    board[row][col] = symbol
    if check_win(symbol):
        global game_over
        game_over = True
        display_message(f"Player {player_num} ({symbol}) wins! Click to restart.")
    elif check_draw():
        game_over = True
        display_message('Draw! Click to restart.')

def load_trained_model(model_path="dqn_tictactoe.pth"):
    agent = DQNAgent()
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.policy_net.eval()
    return agent

def select_player(player_num):
    """
    Prompts the user to select the type of the player.
    
    Args:
        player_num (int): The player number (1 or 2).
    
    Returns:
        str: The selected player type.
    """
    while True:
        print(f"\nSelect Player {player_num}:")
        print("1. Human")
        print("2. Computer (Random)")
        print("3. AI")
        choice = input("Enter choice (1-3): ").strip()
        if choice in PLAYER_TYPES:
            return PLAYER_TYPES[choice]
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    # Select player types
    print("Welcome to Tic Tac Toe!")
    player1_type = select_player(1)
    player2_type = select_player(2)
    
    # Load AI agents if needed
    agent1 = None
    agent2 = None
    if player1_type == PLAYER_AI:
        try:
            agent1 = load_trained_model()
            print("AI agent for Player 1 loaded successfully.")
        except FileNotFoundError:
            print("Trained AI model not found for Player 1. Please train the model first.")
            sys.exit()
    if player2_type == PLAYER_AI:
        try:
            agent2 = load_trained_model()
            print("AI agent for Player 2 loaded successfully.")
        except FileNotFoundError:
            print("Trained AI model not found for Player 2. Please train the model first.")
            sys.exit()
    
    draw_lines()
    global game_over, waiting_for_agent_move, agent_move_time, waiting_for_computer_move, computer_move_time
    game_over = False
    waiting_for_agent_move = False
    agent_move_time = 0
    waiting_for_computer_move = False
    computer_move_time = 0
    
    while True:
        current_time = pygame.time.get_ticks()  # Current time in milliseconds
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
            # Handle game restart
            if event.type == pygame.MOUSEBUTTONDOWN and game_over:
                restart()
                # Reload AI agents if needed
                if player1_type == PLAYER_AI:
                    agent1 = load_trained_model()
                if player2_type == PLAYER_AI:
                    agent2 = load_trained_model()
                continue
    
            # Handle human moves
            if event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                # Determine which player is making the move
                player_num = get_current_player_num()
                player_type = player1_type if player_num == 1 else player2_type
                if player_type == PLAYER_HUMAN:
                    mouse_pos = pygame.mouse.get_pos()
                    row, col = get_pos(mouse_pos)
                    if board[row][col] is None:
                        symbol = 'X' if player_num == 1 else 'O'
                        board[row][col] = symbol
                        if check_win(symbol):
                            game_over = True
                            display_message(f"Player {player_num} ({symbol}) wins! Click to restart.")
                        elif check_draw():
                            game_over = True
                            display_message('Draw! Click to restart.')
    
        # Determine current player
        current_player = get_current_player_num()
        current_player_type = player1_type if current_player == 1 else player2_type
        current_agent = agent1 if current_player == 1 else agent2
    
        # Handle AI or Computer moves
        if not game_over:
            if current_player_type == PLAYER_AI:
                if not waiting_for_agent_move:
                    waiting_for_agent_move = True
                    agent_move_time = current_time + AGENT_MOVE_DELAY
            elif current_player_type == PLAYER_COMPUTER:
                if not waiting_for_computer_move:
                    waiting_for_computer_move = True
                    computer_move_time = current_time + COMPUTER_MOVE_DELAY
    
        # Execute scheduled AI move
        if waiting_for_agent_move and current_time >= agent_move_time and not game_over:
            if current_player_type == PLAYER_AI and current_agent:
                agent_move(current_agent, player_num=current_player)
            waiting_for_agent_move = False
    
        # Execute scheduled Computer move
        if waiting_for_computer_move and current_time >= computer_move_time and not game_over:
            if current_player_type == PLAYER_COMPUTER:
                computer_move(player_num=current_player)
            waiting_for_computer_move = False
    
        draw_figures()
        pygame.display.update()

if __name__ == "__main__":
    main()