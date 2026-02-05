# Chess Engine with Machine Learning-Tuned Evaluation

A modular chess engine built from scratch in Python, featuring alpha-beta search with modern enhancements and a **supervised learning pipeline** for optimizing evaluation weights using Stockfish-labeled positions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
  - [Search Algorithm](#search-algorithm)
  - [Evaluation Function](#evaluation-function)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
- [Performance](#performance)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a complete chess engine with:

1. **Search**: Negamax with alpha-beta pruning, quiescence search, and transposition tables
2. **Evaluation**: Tapered evaluation with 15+ strategic heuristics and phase-specific piece-square tables
3. **ML Tuning**: Supervised learning pipeline using Ridge regression on 200k+ Stockfish-labeled positions
4. **GUI**: Interactive Pygame interface to play against the engine

The primary goal was to explore the intersection of classical game tree search and machine learning for parameter optimization.

---

## Features

### Search Enhancements
- **Alpha-Beta Pruning**: Fail-soft implementation with principal variation tracking
- **Iterative Deepening**: Time-controlled search with configurable time limits (0.5s - 5s)
- **Late Move Reductions (LMR)**: Reduces search depth for likely bad moves
- **Quiescence Search**: Resolves tactical sequences to avoid horizon effect
- **Transposition Table**: Zobrist hashing with depth-preferred replacement
- **Move Ordering**: MVV-LVA for captures, killer move heuristic for quiet moves
- **Opening Book**: Polyglot format support with weighted random selection

### Evaluation Components
- **Tapered Evaluation**: Smooth interpolation between middlegame and endgame scores
- **Material**: Standard piece values (P=100, N=320, B=330, R=500, Q=900)
- **Piece-Square Tables**: 12 tables (6 pieces × 2 phases) encouraging positional play
- **Pawn Structure**: Doubled, isolated, passed, and outside passed pawn detection
- **King Safety**: Pawn shield, open file penalties, attacker proximity scoring
- **Tactical Evaluation**: Hanging pieces, forks, pins, and skewers detection
- **Rook Features**: Open/semi-open files, 7th rank bonuses, rook connectivity
- **Bishop Features**: Fianchetto, long diagonal control, bad bishop detection
- **Piece Activity**: Mobility scoring for all pieces
- **Development**: Early game piece development and castling bonuses
- **Endgame**: King centralization, passed pawn advancement, mating patterns
- **Positional**: Bishop pair, knight outposts, space control

### Machine Learning
- **Data Generation**: Filters high-Elo (2200+) games ending in checkmate
- **Position Selection**: Extracts quiet middlegame positions only
- **Feature Engineering**: 741+ features including PST indicators and heuristic counts
- **Model**: Ridge regression with hyperparameter tuning via validation MSE
- **Data Integrity**: Game-level train/val/test splits to prevent data leakage

---

## Project Structure

```
chess-engine/
├── chess_engine/                 # Core engine package
│   ├── __init__.py
│   ├── engine.py                 # Engine class, opening book integration
│   ├── search.py                 # Negamax, alpha-beta, quiescence, TT
│   ├── evaluation.py             # Main evaluation function
│   ├── zobrist.py                # Zobrist hashing implementation
│   ├── tuned_weights.py          # All evaluation parameters & PSTs
│   └── evaluation/               # Modular evaluation features
│       ├── __init__.py
│       ├── features_material.py  # Material counting
│       ├── features_pst.py       # Piece-square table scoring
│       ├── features_pawn.py      # Pawn structure analysis
│       ├── features_knights.py   # Knight outpost detection
│       ├── features_king_safety.py
│       ├── features_space.py     # Mobility and space control
│       ├── features_control.py   # Center control, castling
│       ├── features_special.py   # Bishop pair, rook files
│       ├── features_game_phase.py
│       ├── features_tactics.py   # Hanging pieces, forks, pins
│       ├── features_rooks.py     # Rook placement and connectivity
│       ├── features_bishops.py   # Fianchetto, diagonals
│       ├── features_piece_activity.py  # Piece mobility
│       ├── features_development.py     # Opening development
│       └── features_endgame.py   # Endgame-specific scoring
│
├── training/                     # ML tuning pipeline
│   ├── __init__.py
│   ├── generate_data.py          # PGN → Stockfish-labeled positions
│   ├── tuner.py                  # Feature extraction & Ridge regression
│   └── learned_weights.py        # Auto-generated learned coefficients
│
├── assets/                       # Static resources
│   ├── pieces/                   # Chess piece images (PNG)
│   └── books/                    # Opening book (Polyglot .bin)
│
├── data/                         # Generated datasets (not in repo)
│   ├── train_data.npz
│   ├── val_data.npz
│   └── test_data.npz
│
├── play_bot.py                   # Pygame GUI to play against engine
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- [Stockfish](https://stockfishchess.org/download/) (for data generation only)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chess-engine.git
cd chess-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Play Against the Engine

```bash
python play_bot.py
```

Choose your color (w/b) and play using mouse clicks. The GUI features:
- **Adjustable time controls**: Use +/- buttons to set engine thinking time (0.5s - 5s)
- **Real-time evaluation panel**: Shows current score, search depth, and expected line
- **Evaluation breakdown**: Visual bar graphs showing key factors (material, king safety, etc.)
- **Opening book indicator**: Shows when the engine uses GM games for moves

### Train Custom Weights

1. **Download PGN data** from [Lichess Database](https://database.lichess.org/)

2. **Generate labeled positions**:
   ```bash
   python -m training.generate_data
   ```
   This filters games, extracts quiet positions, and labels them with Stockfish evaluations.

3. **Train weights**:
   ```bash
   python -m training.tuner
   ```
   Outputs `learned_weights.py` with optimized coefficients.

---

## Technical Details

### Search Algorithm

The engine uses **negamax with alpha-beta pruning** in a fail-soft framework:

```python
def negamax(board, depth, alpha, beta, ply):
    # Transposition table lookup
    tt_entry = transposition_table.get(zobrist_hash(board))
    if tt_entry and tt_entry['depth'] >= depth:
        # Use cached score based on bound type
        ...
    
    if depth == 0:
        return quiescence_search(board, alpha, beta)
    
    for move in order_moves(board, ply):
        board.push(move)
        score = -negamax(board, depth-1, -beta, -alpha, ply+1)
        board.pop()
        
        if score >= beta:
            update_killer_moves(move, ply)  # Killer heuristic
            return beta  # Beta cutoff
        alpha = max(alpha, score)
    
    return alpha
```

**Key optimizations:**
- **Zobrist Hashing**: O(1) incremental hash updates for TT lookup
- **MVV-LVA Ordering**: `score = victim_value * 10 - attacker_value`
- **Killer Moves**: Two slots per ply for quiet moves causing cutoffs
- **Quiescence Search**: Searches all captures until position is quiet

### Evaluation Function

The evaluation uses **tapered scoring** to smoothly transition between game phases:

```python
def evaluate(board):
    phase = calculate_phase(board)  # 1.0 = middlegame, 0.0 = endgame
    
    mg_score = material + pst_mg + bishop_pair + rook_files + ...
    eg_score = material + pst_eg + passed_pawns + king_activity + ...
    
    return int(phase * mg_score + (1 - phase) * eg_score)
```

**Phase calculation**: Based on remaining non-pawn material normalized to [0, 1]

### Machine Learning Pipeline

#### Data Generation

```
PGN File → Filter (Elo ≥ 2200, checkmate games) → Game-level split
    → Extract quiet positions → Stockfish evaluation → NPZ output
```

- **Quiet positions**: No checks, captures, or promotions pending
- **Game-level splitting**: Prevents data leakage between train/val/test

#### Feature Engineering

| Feature Type | Count | Description |
|--------------|-------|-------------|
| Material | 6 | Piece count differentials |
| PST (MG) | 384 | Phase-scaled piece-square indicators |
| PST (EG) | 384 | Phase-scaled piece-square indicators |
| Heuristics | ~20 | Pawn structure, king safety, mobility, etc. |

**Total**: ~794 features per position

#### Model Training

```python
# Ridge regression with hyperparameter search
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
best_model = None
best_mse = float('inf')

for alpha in alphas:
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    mse = mean_squared_error(y_val, model.predict(X_val))
    if mse < best_mse:
        best_model, best_mse = model, mse
```

---

## Performance

| Metric | Value |
|--------|-------|
| Search Depth | 4-6 ply + quiescence |
| Positions/second | ~10,000 (evaluation only) |
| Training Data | 200k+ labeled positions |
| Feature Dimensions | 741+ |
| Validation MSE | ~15,000 cp² (varies with data) |

---

## Future Improvements

- [ ] **Aspiration Windows** for faster search
- [ ] **Null Move Pruning**
- [ ] **Neural Network Evaluation** (NNUE-style)
- [ ] **UCI Protocol** for GUI compatibility
- [ ] **Endgame Tablebases** (Syzygy)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [python-chess](https://python-chess.readthedocs.io/) - Chess library
- [Stockfish](https://stockfishchess.org/) - Reference engine for data labeling
- [Lichess](https://lichess.org/) - Open game database
- Chess programming resources: [Chess Programming Wiki](https://www.chessprogramming.org/)
