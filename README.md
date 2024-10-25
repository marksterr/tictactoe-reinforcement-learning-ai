Tic Tac Toe AI with Deep Q-Network (DQN)

Overview

This repository implements a Tic Tac Toe game with an AI opponent trained using a Deep Q-Network (DQN). The project includes scripts for training the AI agent and playing the game against the AI, a human, or a random move generator.

Features

Interactive Game Interface: Play Tic Tac Toe using a graphical interface built with Pygame.
AI Opponent: The AI is trained using DQN to play optimally.
Training Script: Train the AI agent to improve its gameplay over time.
Model Checkpointing: Save and load trained models for continued training or gameplay.
Getting Started

Prerequisites

Ensure you have the following installed:

Python 3.7 or higher: Download Python
pip: Python package installer (usually comes with Python)
Git: For cloning the repository (optional)
Installation

Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/tic-tac-toe-dqn.git
cd tic-tac-toe-dqn
Create a Virtual Environment (Optional but Recommended)

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

Copy code
pip install -r requirements.txt
If requirements.txt is not provided, install the necessary packages manually:

Copy code
pip install pygame torch numpy
Usage

Training the AI Agent

To train the DQN-based AI agent:

Run the Training Script

Copy code
python train.py
Training Logs and Checkpoints

Logs: Training progress, including wins, losses, draws, and average loss, will be logged to the console and a timestamped log file (e.g., training_20231024_123456.txt).
Checkpoints: The trained model's weights will be periodically saved as dqn_tictactoe.pth. The final trained model will be saved as dqn_tictactoe_final.pth upon completion.
Playing the Game

After training, you can play Tic Tac Toe against the AI agent or other opponents.

Ensure a Trained Model Exists

If you've trained the model, the checkpoint dqn_tictactoe.pth should be present in the repository.
Alternatively, download a pre-trained model if available.
Run the Game Script

Copy code
python game.py
Select Player Types

Upon running, you'll be prompted to select the type for each player:

markdown
Copy code
Select Player 1:
1. Human
2. Computer (Random)
3. AI
Enter choice (1-3):

Select Player 2:
1. Human
2. Computer (Random)
3. AI
Enter choice (1-3):
Human: Play manually by clicking on the desired grid cell.
Computer (Random): The opponent makes random valid moves.
AI: The opponent uses the trained DQN agent.
Gameplay

Making Moves: If playing as a human, click on the grid to place your mark (X or O).
AI/Computer Moves: The AI or random opponent will make moves automatically.
Game Over: Upon a win, loss, or draw, a message will be displayed. Click to restart the game.
Project Structure

bash
Copy code
tic-tac-toe-dqn/
├── game.py           # Pygame-based Tic Tac Toe game with AI integration
├── train.py          # Script to train the DQN agent
├── dqn_agent.py      # DQN agent implementation
├── dqn_tictactoe.pth # Trained model checkpoint (created after training)
├── requirements.txt  # Python dependencies
├── README.txt        # This readme file
└── training_logs/    # Directory containing training log files
Configuration

Hyperparameters

You can adjust various hyperparameters to experiment with the training process. These are primarily set in train.py and dqn_agent.py.

In train.py:

Batch Size

makefile
Copy code
BATCH_SIZE = 128  # Increased from 64
Learning Rate

makefile
Copy code
LEARNING_RATE = 1e-3  # Increased from 1e-5
Epsilon Decay

makefile
Copy code
EPSILON_DECAY = 100000  # Increased from 10,000
Target Network Update Frequency

makefile
Copy code
TARGET_UPDATE = 1000  # Changed from 10,000
Replay Memory Capacity

makefile
Copy code
REPLAY_MEMORY_CAPACITY = 200000  # Reduced from 500,000
Number of Episodes

makefile
Copy code
EPISODES = 500000  # Adjust as needed
In dqn_agent.py:

Neural Network Architecture

Modify the hidden layers and neurons as desired to experiment with different network complexities.

Normalization of Input States

The board states are normalized to [0, 1] to stabilize training.

Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

PyTorch for the deep learning framework.
Pygame for the game development library.
Inspirations from various deep reinforcement learning tutorials and resources.
