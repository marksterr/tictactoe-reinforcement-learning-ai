import random
import torch
import numpy as np
from dqn_agent import DQNAgent
import logging
from datetime import datetime

# Constants
BOARD_SIZE = 9
BOARD_ROWS = 3
BOARD_COLS = 3

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-5
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
TARGET_UPDATE = 10000
EPISODES = 1000000
LOSS_LOG_INTERVAL = 1000
CHECKPOINT_INTERVAL = 10000

def encode_board(board):
    """
    Encode the board state into a flattened numerical vector.
    0: Empty, 1: 'X', 2: 'O'
    """
    return [1 if cell == 'X' else 2 if cell == 'O' else 0 for cell in board]

def get_random_move(board):
    """
    Returns a random available move on the board.
    """
    available_moves = [i for i in range(BOARD_SIZE) if board[i] is None]
    return random.choice(available_moves) if available_moves else None

def check_win(board, player):
    # Define winning combinations
    win_combinations = [
        [0,1,2], [3,4,5], [6,7,8],  # Rows
        [0,3,6], [1,4,7], [2,5,8],  # Columns
        [0,4,8], [2,4,6]             # Diagonals
    ]
    for combo in win_combinations:
        if all(board[pos] == player for pos in combo):
            return True
    return False

def check_draw(board):
    return all(cell is not None for cell in board)

def setup_logging():
    """
    Configures logging to output to both console and a condensed log file.
    """
    log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create a custom logger
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create a concise formatter
    formatter = logging.Formatter('%(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def train_agent():
    logger = setup_logging()
    logger.info("Starting training session.")
    
    agent = DQNAgent(lr=LEARNING_RATE, gamma=GAMMA, epsilon_start=EPSILON_START, 
                    epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY)
    
    win_count = 0
    loss_count = 0
    draw_count = 0
    loss_total = 0
    
    for episode in range(1, EPISODES + 1):
        board = [None for _ in range(BOARD_SIZE)]
        state = encode_board(board)
        done = False

        while not done:
            # Agent's turn ('X')
            available_actions = [i for i in range(BOARD_SIZE) if board[i] is None]
            if not available_actions:
                break  # Draw

            action = agent.select_action(state, available_actions)
            board[action] = 'X'
            next_state = encode_board(board)

            if check_win(board, 'X'):
                reward = 1
                agent.remember(state, action, reward, next_state, True)
                done = True
                win_count += 1
            elif check_draw(board):
                reward = 0
                agent.remember(state, action, reward, next_state, True)
                done = True
                draw_count += 1
            else:
                # Opponent's turn ('O')
                opponent_action = get_random_move(board)
                if opponent_action is not None:
                    board[opponent_action] = 'O'
                    next_state = encode_board(board)
                    if check_win(board, 'O'):
                        reward = -1
                        agent.remember(state, action, reward, next_state, True)
                        done = True
                        loss_count += 1
                    elif check_draw(board):
                        reward = 0
                        agent.remember(state, action, reward, next_state, True)
                        done = True
                        draw_count += 1
                    else:
                        reward = 0
                        agent.remember(state, action, reward, next_state, False)
                        state = next_state
                else:
                    # No moves left
                    reward = 0
                    agent.remember(state, action, reward, next_state, True)
                    done = True
                    draw_count += 1

            # Optimize the model and accumulate loss
            loss = agent.optimize_model(BATCH_SIZE)
            if loss is not None:
                loss_total += loss.item()

        # Update the target network periodically
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()
            logger.info(f"Ep:{episode} - Target network updated.")

        # Logging every LOSS_LOG_INTERVAL episodes
        if episode % LOSS_LOG_INTERVAL == 0:
            avg_loss = loss_total / LOSS_LOG_INTERVAL if loss_total > 0 else 0
            log_message = f"Ep:{episode} W:{win_count} L:{loss_count} D:{draw_count} AvgLoss:{avg_loss:.4f}"
            logger.info(log_message)
            win_count = 0
            loss_count = 0
            draw_count = 0
            loss_total = 0

        # Save model checkpoints periodically
        if episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = f"dqn_tictactoe.pth"
            torch.save(agent.policy_net.state_dict(), checkpoint_path)
            logger.info(f"Ep:{episode} - Checkpoint saved at {checkpoint_path}")

    # Save the final trained model
    torch.save(agent.policy_net.state_dict(), "dqn_tictactoe_final.pth")
    logger.info("Training completed - Final model saved as 'dqn_tictactoe_final.pth'.")

if __name__ == "__main__":
    train_agent()