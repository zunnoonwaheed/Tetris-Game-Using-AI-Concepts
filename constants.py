"""
Constants and configuration for the Tetris AI game.
"""
import pygame
import os

# Initialize pygame mixer for sound
pygame.mixer.init()

# Game dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SIDEBAR_WIDTH = 300
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
DARK_GRAY = (20, 20, 20)
LIGHT_GRAY = (100, 100, 100)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
CYAN = (0, 240, 240)
MAGENTA = (240, 0, 240)
YELLOW = (240, 240, 0)
ORANGE = (240, 160, 0)
PURPLE = (160, 0, 240)

# UI Colors
BG_COLOR = (15, 15, 30)
PANEL_COLOR = (30, 30, 45)
HIGHLIGHT_COLOR = (60, 60, 90)
ACCENT_COLOR = (100, 100, 255)
TEXT_COLOR = (220, 220, 220)

# Tetromino shapes and colors
SHAPES = [
    [[1, 1, 1, 1]],                  # I
    [[1, 1], [1, 1]],                # O
    [[1, 1, 1], [0, 1, 0]],          # T
    [[1, 1, 1], [1, 0, 0]],          # L
    [[1, 1, 1], [0, 0, 1]],          # J
    [[1, 1, 0], [0, 1, 1]],          # Z
    [[0, 1, 1], [1, 1, 0]]           # S
]

COLORS = [CYAN, YELLOW, PURPLE, ORANGE, BLUE, RED, GREEN]
SHAPE_NAMES = ["I", "O", "T", "L", "J", "Z", "S"]

# Game modes
GAME_MODES = {
    "CLASSIC": "Classic Tetris",
    "AI_PLAY": "AI Player",
    "AI_BATTLE": "Battle AI",
    "AI_TRAIN": "Train AI",
    "CONSTRAINT": "Constraint Mode",
    "MULTIPLAYER": "Multiplayer"
}

# Difficulty levels
DIFFICULTY_LEVELS = {
    "EASY": {"fall_speed": 500, "ai_delay": 300},
    "MEDIUM": {"fall_speed": 300, "ai_delay": 200},
    "HARD": {"fall_speed": 150, "ai_delay": 100},
    "EXPERT": {"fall_speed": 100, "ai_delay": 50}
}

# Sound effects
SOUND_EFFECTS = {
    "move": "sounds/move.wav",
    "rotate": "sounds/rotate.wav",
    "drop": "sounds/drop.wav",
    "clear": "sounds/clear.wav",
    "tetris": "sounds/tetris.wav",
    "level_up": "sounds/level_up.wav",
    "game_over": "sounds/game_over.wav",
    "menu_select": "sounds/menu_select.wav",
    "menu_navigate": "sounds/menu_navigate.wav",
    "ai_move": "sounds/ai_move.wav",
    "hold": "sounds/hold.wav"
}

# Background music
BACKGROUND_MUSIC = "sounds/background_music.mp3"

# AI parameters
DEFAULT_AI_WEIGHTS = {
    'holes': -4.0,
    'bumpiness': -1.0,
    'height': -1.5,
    'lines_cleared': 3.0,
    'wells': -1.0,
    'row_transitions': -1.0,
    'col_transitions': -0.5
}

# Neural network parameters
NN_INPUT_SIZE = 8
NN_HIDDEN_SIZE = 64
NN_OUTPUT_SIZE = 1
NN_LEARNING_RATE = 0.001
NN_MODEL_PATH = "tetris_nn_model.pth"

# Genetic algorithm parameters
GA_POPULATION_SIZE = 30
GA_GENERATIONS = 20
GA_MUTATION_RATE = 0.1
GA_TOURNAMENT_SIZE = 5
GA_ELITISM = True

# Constraint mode parameters
CONSTRAINT_DURATION = 30000  # 30 seconds
CONSTRAINT_MODES = ["GRAVITY", "NARROW", "BLOCKS", "INVISIBLE", "MIRROR"]

# Multiplayer parameters
MULTIPLAYER_MODES = ["VS_AI", "VS_PLAYER", "COOPERATIVE"]
