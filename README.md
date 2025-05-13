# Cosmic Conquest

![cosmic_conquest_image](https://github.com/user-attachments/assets/fa1f4834-0ebf-4ccb-99bd-48535d512571)


## An Interstellar Strategy Board Game

**Cosmic Conquest** is a turn-based strategy game set in a dynamic space environment where players compete to gain control of star systems across the galaxy. Navigate through an ever-changing cosmic landscape, manage resources, and outmaneuver your AI opponent to claim victory among the stars!

## 🌌 Features

- **Dynamic Star System Network**: Control nodes in a procedurally generated galaxy network
- **Resource Management**: Collect and spend resources to expand your interstellar empire
- **Cosmic Events**: Unpredictable space phenomena that alter the game board
- **Strategic AI Opponent**: Challenging AI using Minimax algorithm with Alpha-Beta pruning
- **Multiple Game Modes**:
  - **Conquest Mode**: Win by controlling a target number of star systems
  - **Resource Mode**: Win by accumulating target resources
  - **Elimination Mode**: Win by eliminating your opponent's ability to move
- **Varied Star System Types**:
  - Normal systems (standard nodes)
  - Resource-rich systems (generate extra resources)
  - Strategic systems (provide movement advantages)
  - Wormhole systems (enable long-distance travel)

## 📋 Requirements

- Python 3.6+
- Pygame library

## 🚀 Installation

1. Ensure you have Python installed on your system
2. Install the required Pygame library:
   ```
   pip install pygame
   ```
3. Clone or download this repository
4. Run the game:
   ```
   python cosmic_conquest.py
   ```

## 🎮 How to Play

1. **Main Menu**:
   - Select a game mode (Conquest, Resource, or Elimination)
   - Choose board size (Small, Medium, or Large)
   - Click "START GAME" to begin

2. **Gameplay**:
   - Players take turns moving between star systems
   - On your turn, click on a highlighted star system to move to it
   - Movement costs resources based on distance and system types
   - Collect resources from controlled systems each turn
   - Watch for cosmic events that may alter the game board
   - Strategic planning is key to victory!

3. **Controls**:
   - Mouse: Select menu options and make moves
   - Hover over star systems to view detailed information

## 🏆 Win Conditions

- **Conquest Mode**: Be the first to control 5 star systems
- **Resource Mode**: Be the first to accumulate 50 resource units
- **Elimination Mode**: Eliminate your opponent's ability to make legal moves

## 🤖 AI Strategy

The AI opponent uses a Minimax algorithm with Alpha-Beta pruning to make strategic decisions. It evaluates game states based on:
- Resource accumulation
- Number of controlled systems
- Strategic positioning

The AI occasionally makes random moves (20% chance) to add unpredictability to its strategy.

## 🌟 Game Elements

### Star System Types

- **Normal Systems**: Standard nodes with basic resource generation
- **Resource Systems**: Generate extra resources each turn when controlled
- **Strategic Systems**: Provide movement advantages (30% cheaper movement)
- **Wormhole Systems**: Enable long-distance travel to other wormholes

### Cosmic Events

Random events that occur throughout gameplay:
- **Toggle Edge**: Connections between systems may open or close
- **Bonus Resources**: Random resource bonuses awarded
- **Warp Storm**: Temporarily blocks connections to specific systems
- **Node Upgrade**: Upgrades normal systems to more valuable types

## 🛠️ Project Structure

- **Main Game Loop**: Handles game states and user input
- **Game State**: Manages the current state of the game board, player positions, and resources
- **AI Implementation**: Implements the Minimax algorithm with Alpha-Beta pruning
- **UI Functions**: Renders the game board, nodes, and interface elements

## 📜 License

This project is released under the MIT License.

## 👨‍💻 Developer Notes

Cosmic Conquest demonstrates advanced game development concepts including:
- Graph-based game board representation
- Dynamic element generation
- Resource management systems
- AI implementation with pruning optimization
- Event-driven gameplay mechanics

Enjoy your conquest of the cosmos!
