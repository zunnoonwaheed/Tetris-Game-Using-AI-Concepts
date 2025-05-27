"""
Advanced Tetris AI Game
-----------------------
A comprehensive Tetris implementation with multiple AI techniques:
- Search algorithms for optimal piece placement
- Genetic algorithms for parameter optimization
- Neural network for board evaluation
- Adversarial search for competitive play
- Constraint satisfaction for special game modes

Authors: [Your Team Names]
Date: May 2025
"""

import pygame
import random
import numpy as np
import time
import os
import sys
import math
import pickle
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Try to import torch for neural network functionality
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural network features will be disabled.")

# Initialize Pygame
pygame.init()
pygame.font.init()

# Game Constants
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

# Game modes
GAME_MODES = {
    "CLASSIC": "Classic Tetris",
    "AI_PLAY": "AI Player",
    "AI_BATTLE": "Battle AI",
    "AI_TRAIN": "Train AI",
    "CONSTRAINT": "Constraint Mode"
}

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Advanced Tetris AI")
clock = pygame.time.Clock()

# Load fonts
try:
    title_font = pygame.font.Font(None, 64)
    large_font = pygame.font.Font(None, 48)
    medium_font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    tiny_font = pygame.font.Font(None, 18)
except:
    # Fallback to system font if custom font fails
    title_font = pygame.font.SysFont('Arial', 64)
    large_font = pygame.font.SysFont('Arial', 48)
    medium_font = pygame.font.SysFont('Arial', 36)
    small_font = pygame.font.SysFont('Arial', 24)
    tiny_font = pygame.font.SysFont('Arial', 18)

# Helper Functions
def draw_text(surface, text, font, color, x, y, align="left"):
    """Draw text with alignment options"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    
    if align == "center":
        text_rect.center = (x, y)
    elif align == "right":
        text_rect.right = x
        text_rect.centery = y
    else:  # left alignment
        text_rect.left = x
        text_rect.centery = y
        
    surface.blit(text_surface, text_rect)
    return text_rect

def draw_button(surface, text, font, rect, color, hover_color, text_color, action=None, mouse_pos=None, mouse_clicked=False):
    """Draw an interactive button"""
    is_hover = rect.collidepoint(mouse_pos) if mouse_pos else False
    button_color = hover_color if is_hover else color
    
    # Draw button
    pygame.draw.rect(surface, button_color, rect, border_radius=5)
    pygame.draw.rect(surface, LIGHT_GRAY, rect, 2, border_radius=5)
    
    # Draw text
    text_surf = font.render(text, True, text_color)
    text_rect = text_surf.get_rect(center=rect.center)
    surface.blit(text_surf, text_rect)
    
    # Handle click
    if is_hover and mouse_clicked and action:
        action()
        
    return is_hover

def draw_panel(surface, rect, color=PANEL_COLOR, border=True, alpha=255):
    """Draw a panel with optional border and transparency"""
    if alpha < 255:
        panel_surf = pygame.Surface((rect.width, rect.height))
        panel_surf.fill(color)
        panel_surf.set_alpha(alpha)
        surface.blit(panel_surf, rect)
    else:
        pygame.draw.rect(surface, color, rect)
    
    if border:
        pygame.draw.rect(surface, LIGHT_GRAY, rect, 1)

def draw_progress_bar(surface, rect, progress, color, bg_color=DARK_GRAY):
    """Draw a progress bar"""
    # Background
    pygame.draw.rect(surface, bg_color, rect, border_radius=3)
    
    # Progress
    progress_rect = pygame.Rect(rect.left, rect.top, rect.width * progress, rect.height)
    if progress_rect.width > 0:
        pygame.draw.rect(surface, color, progress_rect, border_radius=3)
    
    # Border
    pygame.draw.rect(surface, LIGHT_GRAY, rect, 1, border_radius=3)

def plot_to_surface(figure, width, height):
    """Convert a matplotlib figure to a pygame surface"""
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    
    surf = pygame.image.fromstring(raw_data, canvas.get_width_height(), "RGB")
    surf = pygame.transform.smoothscale(surf, (width, height))
    return surf

# Neural Network Model (if PyTorch is available)
if TORCH_AVAILABLE:
    class TetrisNN(nn.Module):
        def __init__(self, input_size=8, hidden_size=64, output_size=1):
            super(TetrisNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
            
        def forward(self, x):
            return self.network(x)

# Game Classes
class Tetris:
    """Main Tetris game class"""
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.current_piece = self.new_piece()
        self.next_pieces = [self.new_piece() for _ in range(3)]  # Show 3 next pieces
        self.held_piece = None
        self.can_hold = True
        self.game_over = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.combo = 0
        self.fall_speed = 500  # ms
        self.last_fall_time = 0
        self.lock_delay = 500  # ms
        self.lock_timer = 0
        self.move_reset_limit = 15
        self.move_resets = 0
        self.ghost_piece_enabled = True
        self.game_mode = "CLASSIC"
        self.ai_active = False
        self.ai_delay = 100  # ms
        self.last_ai_move_time = 0
        self.ai_move_history = []
        self.ai_evaluation_history = []
        self.constraint_mode = None
        self.constraint_timer = 0
        self.constraint_duration = 30000  # 30 seconds
        self.stats = {
            "pieces_placed": 0,
            "tetris_clears": 0,
            "max_score": 0,
            "max_level": 1,
            "total_games": 0,
            "ai_wins": 0,
            "player_wins": 0
        }
        
        # Initialize AI components
        self.ai_weights = {
            'holes': -4.0,
            'bumpiness': -1.0,
            'height': -1.5,
            'lines_cleared': 3.0,
            'wells': -1.0,
            'row_transitions': -1.0,
            'col_transitions': -0.5
        }
        
        # Neural network model (if available)
        self.nn_model = None
        if TORCH_AVAILABLE:
            self.nn_model = TetrisNN()
            # Try to load pre-trained model if exists
            try:
                self.nn_model.load_state_dict(torch.load('tetris_nn_model.pth'))
                self.nn_model.eval()
                print("Neural network model loaded successfully")
            except:
                print("No pre-trained neural network model found")
    
    def new_piece(self):
        """Generate a new random piece"""
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': [row[:] for row in SHAPES[shape_idx]],  # Deep copy
            'color': COLORS[shape_idx],
            'x': self.width // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0,
            'rotation': 0,
            'type': shape_idx
        }
    
    def reset(self):
        """Reset the game state"""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.current_piece = self.new_piece()
        self.next_pieces = [self.new_piece() for _ in range(3)]
        self.held_piece = None
        self.can_hold = True
        self.game_over = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.combo = 0
        self.fall_speed = 500
        self.last_fall_time = pygame.time.get_ticks()
        self.lock_timer = 0
        self.move_resets = 0
        self.ai_move_history = []
        self.ai_evaluation_history = []
        self.constraint_mode = None
        self.constraint_timer = 0
        self.stats["total_games"] += 1
    
    def valid_position(self, piece=None, x=None, y=None):
        """Check if the piece is in a valid position"""
        if piece is None:
            piece = self.current_piece
        if x is None:
            x = piece['x']
        if y is None:
            y = piece['y']
        
        shape = piece['shape']
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    # Check if the piece is within bounds
                    if (y + row >= self.height or 
                        x + col < 0 or 
                        x + col >= self.width or 
                        y + row < 0):
                        return False
                    # Check if the piece overlaps with existing blocks
                    if y + row >= 0 and self.grid[y + row][x + col] != 0:
                        return False
        return True
    
    def rotate_piece(self, clockwise=True):
        """Rotate the current piece"""
        if self.game_over or self.paused:
            return False
            
        # Create a copy of the current piece
        new_piece = self.current_piece.copy()
        new_shape = [row[:] for row in new_piece['shape']]
        
        # Transpose the shape matrix
        new_shape = list(zip(*new_shape))
        
        # Reverse each row for clockwise rotation or each column for counter-clockwise
        if clockwise:
            new_shape = [list(row[::-1]) for row in new_shape]
        else:
            new_shape = list(reversed([list(row) for row in new_shape]))
        
        new_piece['shape'] = new_shape
        new_piece['rotation'] = (new_piece['rotation'] + (1 if clockwise else 3)) % 4
        
        # Try to place the rotated piece
        if self.valid_position(new_piece):
            self.current_piece = new_piece
            self._reset_lock_timer()
            return True
        
        # Wall kick - try to move the piece left or right if rotation is blocked
        for offset in [-1, 1, -2, 2]:
            new_piece_offset = new_piece.copy()
            new_piece_offset['x'] += offset
            if self.valid_position(new_piece_offset):
                self.current_piece = new_piece_offset
                self._reset_lock_timer()
                return True
        
        return False
    
    def move_piece(self, dx=0, dy=0):
        """Move the current piece"""
        if self.game_over or self.paused:
            return False
            
        if self.valid_position(x=self.current_piece['x'] + dx, y=self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            
            # Reset lock timer if moved horizontally or upward
            if dx != 0 or dy < 0:
                self._reset_lock_timer()
                
            return True
        return False
    
    def _reset_lock_timer(self):
        """Reset the lock timer if move resets are available"""
        if self.move_resets < self.move_reset_limit:
            self.lock_timer = 0
            self.move_resets += 1
    
    def drop_piece(self):
        """Drop the piece as far as it can go"""
        if self.game_over or self.paused:
            return False
            
        drop_distance = 0
        while self.valid_position(y=self.current_piece['y'] + drop_distance + 1):
            drop_distance += 1
        
        if drop_distance > 0:
            self.current_piece['y'] += drop_distance
            self.score += drop_distance * (self.level // 2 + 1)  # Bonus points for hard drop
            self.lock_piece()
            return True
        
        return False
    
    def hold_piece(self):
        """Hold the current piece and swap with held piece if exists"""
        if self.game_over or self.paused or not self.can_hold:
            return False
            
        if self.held_piece is None:
            self.held_piece = {
                'shape': [row[:] for row in SHAPES[self.current_piece['type']]],
                'color': COLORS[self.current_piece['type']],
                'x': self.width // 2 - len(SHAPES[self.current_piece['type']][0]) // 2,
                'y': 0,
                'rotation': 0,
                'type': self.current_piece['type']
            }
            self.current_piece = self.next_pieces.pop(0)
            self.next_pieces.append(self.new_piece())
        else:
            temp = self.held_piece
            self.held_piece = {
                'shape': [row[:] for row in SHAPES[self.current_piece['type']]],
                'color': COLORS[self.current_piece['type']],
                'x': self.width // 2 - len(SHAPES[self.current_piece['type']][0]) // 2,
                'y': 0,
                'rotation': 0,
                'type': self.current_piece['type']
            }
            self.current_piece = temp
        
        self.can_hold = False
        return True
    
    def lock_piece(self):
        """Lock the current piece in place and generate a new one"""
        if self.game_over or self.paused:
            return False
            
        shape = self.current_piece['shape']
        color_idx = COLORS.index(self.current_piece['color']) + 1  # +1 because 0 is empty
        
        # Add the piece to the grid
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    grid_y = self.current_piece['y'] + row
                    grid_x = self.current_piece['x'] + col
                    
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        self.grid[grid_y][grid_x] = color_idx
                    else:
                        # If any part of the piece is above the grid, game over
                        self.game_over = True
                        return False
        
        # Update statistics
        self.stats["pieces_placed"] += 1
        
        # Check for completed lines
        self.clear_lines()
        
        # Get the next piece
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self.new_piece())
        
        # Reset lock timer and move resets
        self.lock_timer = 0
        self.move_resets = 0
        
        # Allow holding again
        self.can_hold = True
        
        # Check if the new piece can be placed
        if not self.valid_position():
            self.game_over = True
            
            # Update max stats
            self.stats["max_score"] = max(self.stats["max_score"], self.score)
            self.stats["max_level"] = max(self.stats["max_level"], self.level)
            
            return False
            
        return True
    
    def clear_lines(self):
        """Clear completed lines and update score"""
        lines_to_clear = []
        for row in range(self.height):
            if all(self.grid[row]):
                lines_to_clear.append(row)
        
        if not lines_to_clear:
            self.combo = 0
            return 0
        
        # Remove the completed lines
        for row in sorted(lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, row, axis=0)
            # Add a new empty row at the top
            self.grid = np.vstack([np.zeros((1, self.width), dtype=int), self.grid])
        
        # Update score and level
        lines_count = len(lines_to_clear)
        self.lines_cleared += lines_count
        
        # Update combo
        self.combo += 1
        
        # Classic Tetris scoring with combo bonus
        line_scores = [100, 300, 500, 800]  # 1, 2, 3, 4 lines
        score_gain = line_scores[min(lines_count, 4) - 1] * self.level
        combo_bonus = (self.combo - 1) * 50 * self.level if self.combo > 1 else 0
        self.score += score_gain + combo_bonus
        
        # Update level
        self.level = max(1, self.lines_cleared // 10 + 1)
        
        # Adjust fall speed based on level
        self.fall_speed = max(50, 500 - (self.level - 1) * 20)
        
        # Update tetris clears stat
        if lines_count == 4:
            self.stats["tetris_clears"] += 1
        
        return lines_count
    
    def get_ghost_piece(self):
        """Get the position where the current piece would land"""
        if not self.ghost_piece_enabled:
            return None
            
        ghost = self.current_piece.copy()
        drop_distance = 0
        
        while self.valid_position(ghost, ghost['x'], ghost['y'] + drop_distance + 1):
            drop_distance += 1
        
        ghost['y'] += drop_distance
        return ghost
    
    def update(self, current_time):
        """Update game state"""
        if self.game_over or self.paused:
            return
        
        # Handle automatic falling
        if current_time - self.last_fall_time > self.fall_speed:
            self.last_fall_time = current_time
            
            # Try to move down
            if not self.move_piece(dy=1):
                # Start lock timer if piece can't move down
                if self.lock_timer == 0:
                    self.lock_timer = current_time
            
        # Check if lock delay has passed
        if self.lock_timer > 0 and current_time - self.lock_timer > self.lock_delay:
            self.lock_piece()
            self.lock_timer = 0
        
        # Update constraint mode if active
        if self.constraint_mode:
            if current_time - self.constraint_timer > self.constraint_duration:
                self.constraint_mode = None
            self.apply_constraint()
    
    def apply_constraint(self):
        """Apply the current constraint mode effect"""
        if self.constraint_mode == "GRAVITY":
            # Increased gravity - pieces fall faster
            self.fall_speed = max(50, self.fall_speed * 0.8)
        elif self.constraint_mode == "NARROW":
            # Narrow playfield - block off sides
            for row in range(self.height):
                if self.grid[row][0] == 0:
                    self.grid[row][0] = -1  # Special value for constraint blocks
                if self.grid[row][self.width-1] == 0:
                    self.grid[row][self.width-1] = -1
        elif self.constraint_mode == "BLOCKS":
            # Random blocks appear on the field
            if random.random() < 0.01:  # 1% chance per update
                x = random.randint(0, self.width-1)
                y = random.randint(self.height//2, self.height-1)
                if self.grid[y][x] == 0:
                    self.grid[y][x] = -1
    
    def activate_constraint_mode(self):
        """Activate a random constraint mode"""
        modes = ["GRAVITY", "NARROW", "BLOCKS"]
        self.constraint_mode = random.choice(modes)
        self.constraint_timer = pygame.time.get_ticks()
    
    def draw(self, surface, offset_x=0, offset_y=0, small=False):
        """Draw the game board"""
        cell_size = GRID_SIZE // 2 if small else GRID_SIZE
        
        # Draw the grid background
        grid_rect = pygame.Rect(
            offset_x, 
            offset_y, 
            self.width * cell_size, 
            self.height * cell_size
        )
        pygame.draw.rect(surface, DARK_GRAY, grid_rect)
        pygame.draw.rect(surface, LIGHT_GRAY, grid_rect, 1)
        
        # Draw grid lines
        for x in range(self.width + 1):
            pygame.draw.line(
                surface, 
                GRAY, 
                (offset_x + x * cell_size, offset_y),
                (offset_x + x * cell_size, offset_y + self.height * cell_size),
                1
            )
        
        for y in range(self.height + 1):
            pygame.draw.line(
                surface, 
                GRAY, 
                (offset_x, offset_y + y * cell_size),
                (offset_x + self.width * cell_size, offset_y + y * cell_size),
                1
            )
        
        # Draw the ghost piece
        if not small and not self.game_over and not self.paused:
            ghost_piece = self.get_ghost_piece()
            if ghost_piece:
                shape = ghost_piece['shape']
                for row in range(len(shape)):
                    for col in range(len(shape[row])):
                        if shape[row][col]:
                            pygame.draw.rect(
                                surface,
                                (*ghost_piece['color'][:3], 100),  # Semi-transparent
                                (
                                    offset_x + (ghost_piece['x'] + col) * cell_size,
                                    offset_y + (ghost_piece['y'] + row) * cell_size,
                                    cell_size, cell_size
                                ),
                                1  # Just the outline
                            )
        
        # Draw the grid blocks
        for row in range(self.height):
            for col in range(self.width):
                cell_value = self.grid[row][col]
                if cell_value != 0:
                    if cell_value == -1:  # Constraint block
                        color = GRAY
                    else:
                        color = COLORS[cell_value - 1]
                    
                    pygame.draw.rect(
                        surface,
                        color,
                        (
                            offset_x + col * cell_size,
                            offset_y + row * cell_size,
                            cell_size, cell_size
                        )
                    )
                    
                    # Draw block border
                    pygame.draw.rect(
                        surface,
                        tuple(max(0, c - 50) for c in color),  # Darker version of the color
                        (
                            offset_x + col * cell_size,
                            offset_y + row * cell_size,
                            cell_size, cell_size
                        ),
                        1
                    )
        
        # Draw the current piece
        if not self.game_over and not small:
            shape = self.current_piece['shape']
            for row in range(len(shape)):
                for col in range(len(shape[row])):
                    if shape[row][col]:
                        pygame.draw.rect(
                            surface,
                            self.current_piece['color'],
                            (
                                offset_x + (self.current_piece['x'] + col) * cell_size,
                                offset_y + (self.current_piece['y'] + row) * cell_size,
                                cell_size, cell_size
                            )
                        )
                        
                        # Draw block border
                        pygame.draw.rect(
                            surface,
                            tuple(max(0, c - 50) for c in self.current_piece['color']),
                            (
                                offset_x + (self.current_piece['x'] + col) * cell_size,
                                offset_y + (self.current_piece['y'] + row) * cell_size,
                                cell_size, cell_size
                            ),
                            1
                        )
    
    def draw_piece_preview(self, surface, piece, x, y, cell_size=15):
        """Draw a piece preview (for next pieces or held piece)"""
        if not piece:
            return
            
        shape = piece['shape']
        color = piece['color']
        
        # Calculate dimensions
        width = len(shape[0]) * cell_size
        height = len(shape) * cell_size
        
        # Center the piece
        center_x = x - width // 2
        center_y = y - height // 2
        
        # Draw the piece
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    pygame.draw.rect(
                        surface,
                        color,
                        (
                            center_x + col * cell_size,
                            center_y + row * cell_size,
                            cell_size, cell_size
                        )
                    )
                    
                    # Draw block border
                    pygame.draw.rect(
                        surface,
                        tuple(max(0, c - 50) for c in color),
                        (
                            center_x + col * cell_size,
                            center_y + row * cell_size,
                            cell_size, cell_size
                        ),
                        1
                    )
    
    def get_board_features(self):
        """Extract features from the current board state for AI evaluation"""
        # Get heights of each column
        heights = [0] * self.width
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    heights[col] = self.height - row
                    break
        
        # Calculate aggregate height
        aggregate_height = sum(heights)
        
        # Calculate bumpiness (sum of differences between adjacent columns)
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(self.width-1))
        
        # Count holes (empty cells with filled cells above them)
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if self.grid[row][col] != 0:
                    found_block = True
                elif found_block and self.grid[row][col] == 0:
                    holes += 1
        
        # Count wells (empty cells with filled cells on both sides)
        wells = 0
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col] == 0:
                    left_filled = col > 0 and self.grid[row][col-1] != 0
                    right_filled = col < self.width-1 and self.grid[row][col+1] != 0
                    if left_filled and right_filled:
                        wells += 1
        
        # Count row transitions (changes from filled to empty or vice versa)
        row_transitions = 0
        for row in range(self.height):
            for col in range(1, self.width):
                if (self.grid[row][col] == 0) != (self.grid[row][col-1] == 0):
                    row_transitions += 1
        
        # Count column transitions
        col_transitions = 0
        for col in range(self.width):
            for row in range(1, self.height):
                if (self.grid[row][col] == 0) != (self.grid[row-1][col] == 0):
                    col_transitions += 1
        
        # Count complete lines
        complete_lines = 0
        for row in range(self.height):
            if all(self.grid[row]):
                complete_lines += 1
        
        return {
            'heights': heights,
            'aggregate_height': aggregate_height,
            'bumpiness': bumpiness,
            'holes': holes,
            'wells': wells,
            'row_transitions': row_transitions,
            'col_transitions': col_transitions,
            'complete_lines': complete_lines
        }
    
    def evaluate_board(self, features=None):
        """Evaluate the current board state using heuristics or neural network"""
        if features is None:
            features = self.get_board_features()
        
        if self.nn_model and TORCH_AVAILABLE:
            # Use neural network for evaluation
            try:
                feature_vector = [
                    features['aggregate_height'] / (self.height * self.width),
                    features['bumpiness'] / self.width,
                    features['holes'] / (self.height * self.width),
                    features['wells'] / (self.height * self.width),
                    features['row_transitions'] / (self.height * self.width),
                    features['col_transitions'] / (self.height * self.width),
                    features['complete_lines'] / self.height,
                    self.level / 20  # Normalize level
                ]
                
                input_tensor = torch.FloatTensor(feature_vector)
                with torch.no_grad():
                    score = self.nn_model(input_tensor).item()
                return score
            except Exception as e:
                print(f"Neural network evaluation error: {e}")
                # Fall back to heuristic evaluation
        
        # Heuristic evaluation
        score = (
            self.ai_weights['height'] * features['aggregate_height'] +
            self.ai_weights['bumpiness'] * features['bumpiness'] +
            self.ai_weights['holes'] * features['holes'] +
            self.ai_weights['wells'] * features['wells'] +
            self.ai_weights['row_transitions'] * features['row_transitions'] +
            self.ai_weights['col_transitions'] * features['col_transitions'] +
            self.ai_weights['lines_cleared'] * features['complete_lines']
        )
        
        return score
    
    def simulate_move(self, piece, x, rotation):
        """Simulate a move and evaluate the resulting board state"""
        # Create a copy of the current state
        original_grid = self.grid.copy()
        original_piece = self.current_piece.copy()
        original_score = self.score
        original_lines = self.lines_cleared
        
        # Create a test piece
        test_piece = piece.copy()
        test_piece['x'] = x
        
        # Apply rotation
        for _ in range(rotation):
            shape = list(zip(*test_piece['shape']))
            shape = [list(row[::-1]) for row in shape]
            test_piece['shape'] = shape
        
        # Check if the position is valid
        if not self.valid_position(test_piece):
            # Restore original state
            self.grid = original_grid
            self.current_piece = original_piece
            self.score = original_score
            self.lines_cleared = original_lines
            return None
        
        # Drop the piece
        self.current_piece = test_piece
        while self.valid_position(y=self.current_piece['y'] + 1):
            self.current_piece['y'] += 1
        
        # Lock the piece
        shape = self.current_piece['shape']
        color_idx = COLORS.index(self.current_piece['color']) + 1
        
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    grid_y = self.current_piece['y'] + row
                    grid_x = self.current_piece['x'] + col
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        self.grid[grid_y][grid_x] = color_idx
        
        # Clear lines and get features
        lines_cleared = self.clear_lines()
        features = self.get_board_features()
        evaluation = self.evaluate_board(features)
        
        # Restore original state
        self.grid = original_grid
        self.current_piece = original_piece
        self.score = original_score
        self.lines_cleared = original_lines
        
        return {
            'x': x,
            'rotation': rotation,
            'features': features,
            'evaluation': evaluation,
            'lines_cleared': lines_cleared
        }
    
    def find_best_move(self):
        """Find the best move for the current piece using search algorithm"""
        best_evaluation = float('-inf')
        best_move = None
        all_evaluations = []
        
        # Try all possible rotations and positions
        for rotation in range(4):  # Maximum 4 rotations
            for x in range(-2, self.width + 2):  # Try different x positions with some margin
                result = self.simulate_move(self.current_piece, x, rotation)
                if result:
                    all_evaluations.append(result)
                    if result['evaluation'] > best_evaluation:
                        best_evaluation = result['evaluation']
                        best_move = result
        
        # Store evaluation history for visualization
        self.ai_evaluation_history = all_evaluations
        
        return best_move
    
    def apply_ai_move(self):
        """Apply the best move found by the AI"""
        if self.game_over or self.paused:
            return False
            
        best_move = self.find_best_move()
        if not best_move:
            return False
        
        # Store move for visualization
        self.ai_move_history.append(best_move)
        if len(self.ai_move_history) > 10:
            self.ai_move_history.pop(0)
        
        # Reset position
        self.current_piece['x'] = self.width // 2 - len(self.current_piece['shape'][0]) // 2
        self.current_piece['y'] = 0
        
        # Apply rotation
        for _ in range(best_move['rotation']):
            self.rotate_piece()
        
        # Move to the target x position
        target_x = best_move['x']
        dx = target_x - self.current_piece['x']
        if dx < 0:
            for _ in range(abs(dx)):
                self.move_piece(dx=-1)
        elif dx > 0:
            for _ in range(dx):
                self.move_piece(dx=1)
        
        # Drop the piece
        self.drop_piece()
        return True

class GeneticOptimizer:
    """Genetic algorithm for optimizing AI weights"""
    def __init__(self, population_size=50, generations=20, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # Weight ranges for initialization
        self.weight_ranges = {
            'holes': (-10.0, -0.1),
            'bumpiness': (-5.0, -0.1),
            'height': (-5.0, -0.1),
            'lines_cleared': (0.1, 10.0),
            'wells': (-5.0, -0.1),
            'row_transitions': (-5.0, -0.1),
            'col_transitions': (-5.0, -0.1)
        }
        
        # Statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def create_individual(self):
        """Create a random individual (set of weights)"""
        return {
            key: random.uniform(min_val, max_val)
            for key, (min_val, max_val) in self.weight_ranges.items()
        }
    
    def create_population(self):
        """Create an initial population of random individuals"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, weights, num_games=3, max_moves=200):
        """Evaluate the fitness of a set of weights by playing games"""
        game = Tetris()
        game.ai_weights = weights
        
        total_score = 0
        total_lines = 0
        
        for _ in range(num_games):
            game.reset()
            moves = 0
            
            while not game.game_over and moves < max_moves:
                game.apply_ai_move()
                moves += 1
            
            total_score += game.score
            total_lines += game.lines_cleared
        
        # Fitness is a combination of score and lines cleared
        return (total_score / num_games) + (total_lines / num_games) * 100
    
    def select_parents(self, population, fitnesses):
        """Select parents for reproduction using tournament selection"""
        tournament_size = 5
        
        def select_one():
            # Randomly select individuals for the tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            
            # Select the best individual from the tournament
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            return population[winner_idx]
        
        return select_one(), select_one()
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to create a child"""
        child = {}
        
        for key in parent1.keys():
            # Uniform crossover: randomly select from either parent
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return child
    
    def mutate(self, individual):
        """Apply mutation to an individual"""
        mutated = individual.copy()
        
        for key in mutated.keys():
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                min_val, max_val = self.weight_ranges[key]
                # Add a random value within ±20% of the range
                range_size = max_val - min_val
                mutation_amount = random.uniform(-0.2 * range_size, 0.2 * range_size)
                mutated[key] += mutation_amount
                
                # Ensure the value stays within the valid range
                mutated[key] = max(min_val, min(max_val, mutated[key]))
        
        return mutated
    
    def train(self, callback=None):
        """
        Train the AI using genetic algorithm
        
        Args:
            callback: Optional function to call after each generation with progress info
            
        Returns:
            dict: Best weights found
        """
        # Create initial population
        population = self.create_population()
        
        # Track the best individual across all generations
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness for each individual
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_fitness(individual)
                fitnesses.append(fitness)
                
                # Update best individual if needed
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Store statistics
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            # Call callback if provided
            if callback:
                progress = (generation + 1) / self.generations
                callback(progress, best_individual, best_fitness)
            
            # Create the next generation
            new_population = []
            
            # Elitism: keep the best individual
            best_idx = fitnesses.index(max(fitnesses))
            new_population.append(population[best_idx])
            
            # Create the rest of the population through selection, crossover, and mutation
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(population, fitnesses)
                
                # Create child through crossover
                child = self.crossover(parent1, parent2)
                
                # Apply mutation
                child = self.mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            # Update population
            population = new_population
        
        return best_individual

class NeuralNetworkTrainer:
    """Trainer for the neural network model"""
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network training")
        
        self.model = TetrisNN(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Training data
        self.states = []
        self.targets = []
        
        # Statistics
        self.loss_history = []
    
    def collect_training_data(self, num_games=10, max_moves=200):
        """Collect training data by playing games with the current best heuristic"""
        game = Tetris()
        
        for _ in range(num_games):
            game.reset()
            moves = 0
            
            while not game.game_over and moves < max_moves:
                # Find best move using heuristic
                best_move = game.find_best_move()
                if not best_move:
                    break
                
                # Extract features before making the move
                features = game.get_board_features()
                feature_vector = [
                    features['aggregate_height'] / (game.height * game.width),
                    features['bumpiness'] / game.width,
                    features['holes'] / (game.height * game.width),
                    features['wells'] / (game.height * game.width),
                    features['row_transitions'] / (game.height * game.width),
                    features['col_transitions'] / (game.height * game.width),
                    features['complete_lines'] / game.height,
                    game.level / 20  # Normalize level
                ]
                
                # Apply the move
                game.apply_ai_move()
                
                # Use the heuristic evaluation as the target
                target = best_move['evaluation']
                
                # Store the data
                self.states.append(feature_vector)
                self.targets.append(target)
                
                moves += 1
    
    def train(self, epochs=100, batch_size=32, callback=None):
        """Train the neural network on collected data"""
        if not self.states:
            raise ValueError("No training data available. Call collect_training_data first.")
        
        # Convert data to tensors
        X = torch.FloatTensor(self.states)
        y = torch.FloatTensor(self.targets).unsqueeze(1)
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(self.states))
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.states), batch_size):
                # Get batch
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = total_loss / num_batches
            self.loss_history.append(avg_loss)
            
            # Call callback if provided
            if callback:
                progress = (epoch + 1) / epochs
                callback(progress, avg_loss)
        
        # Save the trained model
        torch.save(self.model.state_dict(), 'tetris_nn_model.pth')
        return self.model

class GameUI:
    """User interface for the Tetris game"""
    def __init__(self):
        self.game = Tetris()
        self.current_screen = "MENU"  # MENU, GAME, SETTINGS, AI_TRAINING, HELP
        self.ai_training_progress = 0
        self.genetic_optimizer = None
        self.nn_trainer = None
        
        # UI elements
        self.buttons = {}
        self.sliders = {}
        self.checkboxes = {}
        
        # Performance metrics
        self.fps_history = deque(maxlen=60)
        self.last_time = time.time()
    
    def handle_events(self):
        """Handle pygame events"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
            
            if self.current_screen == "GAME" and not self.game.game_over and not self.game.paused and not self.game.ai_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.game.move_piece(dx=-1)
                    elif event.key == pygame.K_RIGHT:
                        self.game.move_piece(dx=1)
                    elif event.key == pygame.K_DOWN:
                        self.game.move_piece(dy=1)
                    elif event.key == pygame.K_UP:
                        self.game.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        self.game.drop_piece()
                    elif event.key == pygame.K_c:
                        self.game.hold_piece()
                    elif event.key == pygame.K_p:
                        self.game.paused = not self.game.paused
                    elif event.key == pygame.K_a:
                        self.game.ai_active = not self.game.ai_active
                    elif event.key == pygame.K_r:
                        self.game.reset()
                    elif event.key == pygame.K_ESCAPE:
                        self.current_screen = "MENU"
        
        # Update UI elements
        self.update_ui_elements(mouse_pos, mouse_clicked)
        
        return True
    
    def update_ui_elements(self, mouse_pos, mouse_clicked):
        """Update UI elements based on mouse interaction"""
        # Update buttons
        for button_id, button in self.buttons.items():
            if button['rect'].collidepoint(mouse_pos):
                button['hover'] = True
                if mouse_clicked and button['action']:
                    button['action']()
            else:
                button['hover'] = False
        
        # Update sliders
        for slider_id, slider in self.sliders.items():
            slider_rect = slider['rect']
            if slider_rect.collidepoint(mouse_pos) and mouse_clicked:
                # Calculate new value based on mouse position
                slider['value'] = (mouse_pos[0] - slider_rect.left) / slider_rect.width
                slider['value'] = max(0, min(1, slider['value']))
                
                # Call callback if provided
                if slider['callback']:
                    slider['callback'](slider['value'])
        
        # Update checkboxes
        for checkbox_id, checkbox in self.checkboxes.items():
            if checkbox['rect'].collidepoint(mouse_pos) and mouse_clicked:
                checkbox['checked'] = not checkbox['checked']
                
                # Call callback if provided
                if checkbox['callback']:
                    checkbox['callback'](checkbox['checked'])
    
    def add_button(self, button_id, text, rect, action=None):
        """Add a button to the UI"""
        self.buttons[button_id] = {
            'text': text,
            'rect': pygame.Rect(rect),
            'action': action,
            'hover': False
        }
    
    def add_slider(self, slider_id, rect, value=0.5, callback=None):
        """Add a slider to the UI"""
        self.sliders[slider_id] = {
            'rect': pygame.Rect(rect),
            'value': value,
            'callback': callback
        }
    
    def add_checkbox(self, checkbox_id, text, rect, checked=False, callback=None):
        """Add a checkbox to the UI"""
        self.checkboxes[checkbox_id] = {
            'text': text,
            'rect': pygame.Rect(rect),
            'checked': checked,
            'callback': callback
        }
    
    def draw_ui_elements(self, surface):
        """Draw all UI elements"""
        # Draw buttons
        for button in self.buttons.values():
            color = HIGHLIGHT_COLOR if button['hover'] else PANEL_COLOR
            pygame.draw.rect(surface, color, button['rect'], border_radius=5)
            pygame.draw.rect(surface, LIGHT_GRAY, button['rect'], 2, border_radius=5)
            
            text_surf = medium_font.render(button['text'], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=button['rect'].center)
            surface.blit(text_surf, text_rect)
        
        # Draw sliders
        for slider in self.sliders.values():
            # Draw background
            pygame.draw.rect(surface, DARK_GRAY, slider['rect'], border_radius=3)
            
            # Draw filled portion
            filled_rect = pygame.Rect(
                slider['rect'].left,
                slider['rect'].top,
                slider['rect'].width * slider['value'],
                slider['rect'].height
            )
            pygame.draw.rect(surface, ACCENT_COLOR, filled_rect, border_radius=3)
            
            # Draw border
            pygame.draw.rect(surface, LIGHT_GRAY, slider['rect'], 1, border_radius=3)
            
            # Draw handle
            handle_x = slider['rect'].left + slider['rect'].width * slider['value']
            handle_rect = pygame.Rect(
                handle_x - 5,
                slider['rect'].top - 5,
                10,
                slider['rect'].height + 10
            )
            pygame.draw.rect(surface, WHITE, handle_rect, border_radius=5)
        
        # Draw checkboxes
        for checkbox in self.checkboxes.values():
            # Draw box
            pygame.draw.rect(surface, DARK_GRAY, checkbox['rect'])
            pygame.draw.rect(surface, LIGHT_GRAY, checkbox['rect'], 1)
            
            # Draw check if checked
            if checkbox['checked']:
                inner_rect = pygame.Rect(
                    checkbox['rect'].left + 3,
                    checkbox['rect'].top + 3,
                    checkbox['rect'].width - 6,
                    checkbox['rect'].height - 6
                )
                pygame.draw.rect(surface, ACCENT_COLOR, inner_rect)
            
            # Draw text
            text_surf = small_font.render(checkbox['text'], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(
                left=checkbox['rect'].right + 10,
                centery=checkbox['rect'].centery
            )
            surface.blit(text_surf, text_rect)
    
    def draw_menu_screen(self):
        """Draw the main menu screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = title_font.render("Advanced Tetris AI", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Menu buttons
        button_width = 300
        button_height = 60
        button_spacing = 20
        start_y = 250
        
        # Clear previous buttons
        self.buttons = {}
        
        # Add menu buttons
        for i, (mode_key, mode_name) in enumerate(GAME_MODES.items()):
            button_rect = pygame.Rect(
                (SCREEN_WIDTH - button_width) // 2,
                start_y + i * (button_height + button_spacing),
                button_width,
                button_height
            )
            
            self.add_button(
                f"mode_{mode_key}",
                mode_name,
                button_rect,
                action=lambda m=mode_key: self.start_game(m)
            )
        
        # Settings and help buttons
        settings_rect = pygame.Rect(
            (SCREEN_WIDTH - button_width) // 2,
            start_y + len(GAME_MODES) * (button_height + button_spacing),
            button_width,
            button_height
        )
        
        help_rect = pygame.Rect(
            (SCREEN_WIDTH - button_width) // 2,
            start_y + (len(GAME_MODES) + 1) * (button_height + button_spacing),
            button_width,
            button_height
        )
        
        self.add_button("settings", "Settings", settings_rect, action=lambda: self.set_screen("SETTINGS"))
        self.add_button("help", "Help", help_rect, action=lambda: self.set_screen("HELP"))
        
        # Draw all buttons
        self.draw_ui_elements(screen)
        
        # Version info
        version_text = small_font.render("v1.0.0 - Final Project", True, LIGHT_GRAY)
        version_rect = version_text.get_rect(bottomright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20))
        screen.blit(version_text, version_rect)
    
    def draw_game_screen(self):
        """Draw the game screen"""
        screen.fill(BG_COLOR)
        
        # Calculate game board position
        board_width = self.game.width * GRID_SIZE
        board_height = self.game.height * GRID_SIZE
        board_x = (SCREEN_WIDTH - board_width - SIDEBAR_WIDTH) // 2
        board_y = (SCREEN_HEIGHT - board_height) // 2
        
        # Draw game board
        self.game.draw(screen, board_x, board_y)
        
        # Draw sidebar
        sidebar_x = board_x + board_width + 20
        sidebar_y = board_y
        sidebar_width = SIDEBAR_WIDTH
        sidebar_height = board_height
        
        # Sidebar background
        sidebar_rect = pygame.Rect(sidebar_x, sidebar_y, sidebar_width, sidebar_height)
        draw_panel(screen, sidebar_rect)
        
        # Game info
        y_offset = sidebar_y + 20
        
        # Game mode
        mode_text = medium_font.render(GAME_MODES[self.game.game_mode], True, WHITE)
        mode_rect = mode_text.get_rect(center=(sidebar_x + sidebar_width // 2, y_offset))
        screen.blit(mode_text, mode_rect)
        y_offset += 40
        
        # Score and level
        score_text = medium_font.render(f"Score: {self.game.score}", True, WHITE)
        level_text = small_font.render(f"Level: {self.game.level}", True, WHITE)
        lines_text = small_font.render(f"Lines: {self.game.lines_cleared}", True, WHITE)
        
        screen.blit(score_text, (sidebar_x + 10, y_offset))
        y_offset += 30
        screen.blit(level_text, (sidebar_x + 10, y_offset))
        y_offset += 25
        screen.blit(lines_text, (sidebar_x + 10, y_offset))
        y_offset += 40
        
        # Next pieces
        next_text = medium_font.render("Next:", True, WHITE)
        screen.blit(next_text, (sidebar_x + 10, y_offset))
        y_offset += 30
        
        for i, next_piece in enumerate(self.game.next_pieces):
            self.game.draw_piece_preview(
                screen,
                next_piece,
                sidebar_x + sidebar_width // 2,
                y_offset + i * 60 + 30
            )
        
        y_offset += 180
        
        # Held piece
        held_text = medium_font.render("Hold:", True, WHITE)
        screen.blit(held_text, (sidebar_x + 10, y_offset))
        y_offset += 30
        
        self.game.draw_piece_preview(
            screen,
            self.game.held_piece,
            sidebar_x + sidebar_width // 2,
            y_offset + 30
        )
        
        y_offset += 80
        
        # AI status
        ai_status = "ON" if self.game.ai_active else "OFF"
        ai_color = GREEN if self.game.ai_active else RED
        ai_text = medium_font.render(f"AI: {ai_status}", True, ai_color)
        screen.blit(ai_text, (sidebar_x + 10, y_offset))
        y_offset += 40
        
        # Constraint mode
        if self.game.constraint_mode:
            constraint_text = small_font.render(
                f"Constraint: {self.game.constraint_mode}",
                True,
                YELLOW
            )
            screen.blit(constraint_text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            # Constraint timer
            time_left = max(0, (self.game.constraint_duration - 
                              (pygame.time.get_ticks() - self.game.constraint_timer)) / 1000)
            timer_text = small_font.render(f"Time left: {time_left:.1f}s", True, YELLOW)
            screen.blit(timer_text, (sidebar_x + 10, y_offset))
            y_offset += 40
        
        # Game controls
        if y_offset < sidebar_y + sidebar_height - 150:

```python file="tetris_ai_game_part2.py" type="nodejs"
            controls_text = small_font.render("Controls:", True, LIGHT_GRAY)
            screen.blit(controls_text, (sidebar_x + 10, y_offset))
            y_offset += 25
            
            controls = [
                "←/→: Move",
                "↑: Rotate",
                "↓: Soft Drop",
                "Space: Hard Drop",
                "C: Hold Piece",
                "A: Toggle AI",
                "P: Pause",
                "R: Reset",
                "Esc: Menu"
            ]
            
            for control in controls:
                control_text = tiny_font.render(control, True, LIGHT_GRAY)
                screen.blit(control_text, (sidebar_x + 10, y_offset))
                y_offset += 20
        
        # Game over overlay
        if self.game.game_over:
            # Semi-transparent overlay
            overlay = pygame.Surface((board_width, board_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (board_x, board_y))
            
            # Game over text
            game_over_text = large_font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(
                center=(board_x + board_width // 2, board_y + board_height // 2 - 40)
            )
            screen.blit(game_over_text, game_over_rect)
            
            # Final score
            final_score_text = medium_font.render(f"Final Score: {self.game.score}", True, WHITE)
            final_score_rect = final_score_text.get_rect(
                center=(board_x + board_width // 2, board_y + board_height // 2 + 10)
            )
            screen.blit(final_score_text, final_score_rect)
            
            # Restart button
            restart_rect = pygame.Rect(
                board_x + board_width // 2 - 100,
                board_y + board_height // 2 + 60,
                200,
                50
            )
            
            self.add_button("restart", "Play Again", restart_rect, action=lambda: self.game.reset())
            
            # Menu button
            menu_rect = pygame.Rect(
                board_x + board_width // 2 - 100,
                board_y + board_height // 2 + 120,
                200,
                50
            )
            
            self.add_button("menu", "Main Menu", menu_rect, action=lambda: self.set_screen("MENU"))
        
        # Pause overlay
        elif self.game.paused:
            # Semi-transparent overlay
            overlay = pygame.Surface((board_width, board_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (board_x, board_y))
            
            # Paused text
            paused_text = large_font.render("PAUSED", True, WHITE)
            paused_rect = paused_text.get_rect(
                center=(board_x + board_width // 2, board_y + board_height // 2 - 40)
            )
            screen.blit(paused_text, paused_rect)
            
            # Resume button
            resume_rect = pygame.Rect(
                board_x + board_width // 2 - 100,
                board_y + board_height // 2 + 20,
                200,
                50
            )
            
            self.add_button("resume", "Resume", resume_rect, action=lambda: self.toggle_pause())
            
            # Menu button
            menu_rect = pygame.Rect(
                board_x + board_width // 2 - 100,
                board_y + board_height // 2 + 80,
                200,
                50
            )
            
            self.add_button("menu", "Main Menu", menu_rect, action=lambda: self.set_screen("MENU"))
        
        # Draw AI visualization if active
        if self.game.ai_active and self.game.ai_evaluation_history and not self.game.game_over and not self.game.paused:
            self.draw_ai_visualization(board_x, board_y + board_height + 20)
        
        # Draw UI elements
        self.draw_ui_elements(screen)
        
        # Draw FPS counter
        current_time = time.time()
        self.fps_history.append(1.0 / max(0.001, current_time - self.last_time))
        self.last_time = current_time
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        fps_text = tiny_font.render(f"FPS: {avg_fps:.1f}", True, LIGHT_GRAY)
        screen.blit(fps_text, (10, 10))
    
    def draw_ai_visualization(self, x, y):
        """Draw visualization of AI decision making"""
        width = self.game.width * GRID_SIZE
        height = 150
        
        # Background panel
        panel_rect = pygame.Rect(x, y, width, height)
        draw_panel(screen, panel_rect)
        
        # Title
        title_text = small_font.render("AI Decision Making", True, WHITE)
        title_rect = title_text.get_rect(center=(x + width // 2, y + 15))
        screen.blit(title_text, title_rect)
        
        # Draw evaluation bars
        if self.game.ai_evaluation_history:
            # Find min and max evaluations for normalization
            all_evals = [move['evaluation'] for move in self.game.ai_evaluation_history]
            min_eval = min(all_evals)
            max_eval = max(all_evals)
            eval_range = max(0.001, max_eval - min_eval)
            
            # Sort by evaluation
            sorted_evals = sorted(
                self.game.ai_evaluation_history,
                key=lambda m: m['evaluation'],
                reverse=True
            )
            
            # Draw top 5 evaluations
            for i, move in enumerate(sorted_evals[:5]):
                # Normalize evaluation to [0, 1]
                norm_eval = (move['evaluation'] - min_eval) / eval_range
                
                # Bar position and size
                bar_x = x + 10
                bar_y = y + 40 + i * 20
                bar_width = (width - 20) * norm_eval
                bar_height = 15
                
                # Draw bar
                bar_color = (
                    int(255 * (1 - norm_eval)),
                    int(255 * norm_eval),
                    50
                )
                pygame.draw.rect(
                    screen,
                    bar_color,
                    (bar_x, bar_y, bar_width, bar_height),
                    border_radius=3
                )
                
                # Draw border
                pygame.draw.rect(
                    screen,
                    LIGHT_GRAY,
                    (bar_x, bar_y, width - 20, bar_height),
                    1,
                    border_radius=3
                )
                
                # Draw text
                eval_text = tiny_font.render(
                    f"Rot: {move['rotation']}, X: {move['x']}, Score: {move['evaluation']:.2f}",
                    True,
                    WHITE
                )
                screen.blit(eval_text, (bar_x + 5, bar_y + 2))
    
    def draw_settings_screen(self):
        """Draw the settings screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = large_font.render("Settings", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Settings panel
        panel_width = 600
        panel_height = 500
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = 100
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_panel(screen, panel_rect)
        
        # Settings
        y_offset = panel_y + 30
        
        # AI Settings
        settings_title = medium_font.render("AI Settings", True, WHITE)
        settings_rect = settings_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(settings_title, settings_rect)
        y_offset += 50
        
        # AI Speed slider
        speed_text = small_font.render("AI Speed:", True, TEXT_COLOR)
        screen.blit(speed_text, (panel_x + 30, y_offset))
        
        speed_slider_rect = pygame.Rect(panel_x + 150, y_offset, 300, 20)
        speed_value = 1 - (self.game.ai_delay - 10) / 490  # Map 500ms-10ms to 0-1
        
        if "ai_speed" not in self.sliders:
            self.add_slider("ai_speed", speed_slider_rect, speed_value, self.set_ai_speed)
        else:
            self.sliders["ai_speed"]["rect"] = speed_slider_rect
            self.sliders["ai_speed"]["value"] = speed_value
        
        y_offset += 40
        
        # Ghost piece checkbox
        ghost_checkbox_rect = pygame.Rect(panel_x + 30, y_offset, 20, 20)
        
        if "ghost_piece" not in self.checkboxes:
            self.add_checkbox(
                "ghost_piece",
                "Show Ghost Piece",
                ghost_checkbox_rect,
                self.game.ghost_piece_enabled,
                self.toggle_ghost_piece
            )
        else:
            self.checkboxes["ghost_piece"]["rect"] = ghost_checkbox_rect
            self.checkboxes["ghost_piece"]["checked"] = self.game.ghost_piece_enabled
        
        y_offset += 40
        
        # Game Settings
        settings_title = medium_font.render("Game Settings", True, WHITE)
        settings_rect = settings_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(settings_title, settings_rect)
        y_offset += 50
        
        # Fall speed slider
        fall_text = small_font.render("Fall Speed:", True, TEXT_COLOR)
        screen.blit(fall_text, (panel_x + 30, y_offset))
        
        fall_slider_rect = pygame.Rect(panel_x + 150, y_offset, 300, 20)
        fall_value = 1 - (self.game.fall_speed - 50) / 450  # Map 500ms-50ms to 0-1
        
        if "fall_speed" not in self.sliders:
            self.add_slider("fall_speed", fall_slider_rect, fall_value, self.set_fall_speed)
        else:
            self.sliders["fall_speed"]["rect"] = fall_slider_rect
            self.sliders["fall_speed"]["value"] = fall_value
        
        y_offset += 40
        
        # Statistics
        stats_title = medium_font.render("Statistics", True, WHITE)
        stats_rect = stats_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(stats_title, stats_rect)
        y_offset += 40
        
        stats = [
            f"Games Played: {self.game.stats['total_games']}",
            f"Max Score: {self.game.stats['max_score']}",
            f"Max Level: {self.game.stats['max_level']}",
            f"Total Pieces: {self.game.stats['pieces_placed']}",
            f"Tetris Clears: {self.game.stats['tetris_clears']}",
            f"AI Wins: {self.game.stats['ai_wins']}",
            f"Player Wins: {self.game.stats['player_wins']}"
        ]
        
        for stat in stats:
            stat_text = small_font.render(stat, True, TEXT_COLOR)
            screen.blit(stat_text, (panel_x + 30, y_offset))
            y_offset += 25
        
        # Back button
        back_rect = pygame.Rect(
            panel_x + panel_width // 2 - 100,
            panel_y + panel_height - 60,
            200,
            50
        )
        
        self.add_button("back", "Back to Menu", back_rect, action=lambda: self.set_screen("MENU"))
        
        # Draw UI elements
        self.draw_ui_elements(screen)
    
    def draw_help_screen(self):
        """Draw the help screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = large_font.render("Help & Instructions", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Help panel
        panel_width = 800
        panel_height = 600
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = 100
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_panel(screen, panel_rect)
        
        # Content
        y_offset = panel_y + 30
        
        # Game modes
        modes_title = medium_font.render("Game Modes", True, WHITE)
        modes_rect = modes_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(modes_title, modes_rect)
        y_offset += 40
        
        modes_desc = [
            "Classic Tetris: Standard Tetris gameplay",
            "AI Player: Watch the AI play automatically",
            "Battle AI: Compete against the AI in a split-screen mode",
            "Train AI: Train the AI using genetic algorithms or neural networks",
            "Constraint Mode: Play with random constraints that change periodically"
        ]
        
        for desc in modes_desc:
            desc_text = small_font.render(desc, True, TEXT_COLOR)
            screen.blit(desc_text, (panel_x + 30, y_offset))
            y_offset += 25
        
        y_offset += 20
        
        # Controls
        controls_title = medium_font.render("Controls", True, WHITE)
        controls_rect = controls_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(controls_title, controls_rect)
        y_offset += 40
        
        controls = [
            "Left/Right Arrow: Move piece horizontally",
            "Up Arrow: Rotate piece clockwise",
            "Down Arrow: Soft drop (move down faster)",
            "Space: Hard drop (instantly place piece)",
            "C: Hold current piece",
            "P: Pause game",
            "A: Toggle AI assistance",
            "R: Reset game",
            "Esc: Return to main menu"
        ]
        
        for control in controls:
            control_text = small_font.render(control, True, TEXT_COLOR)
            screen.blit(control_text, (panel_x + 30, y_offset))
            y_offset += 25
        
        y_offset += 20
        
        # AI Techniques
        ai_title = medium_font.render("AI Techniques Used", True, WHITE)
        ai_rect = ai_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(ai_title, ai_rect)
        y_offset += 40
        
        techniques = [
            "Search Algorithms: Finding optimal piece placements",
            "Genetic Algorithms: Optimizing evaluation weights",
            "Neural Networks: Learning board evaluation (if PyTorch available)",
            "Adversarial Search: AI vs. Player competition",
            "Constraint Satisfaction: Special game modes with constraints"
        ]
        
        for technique in techniques:
            technique_text = small_font.render(technique, True, TEXT_COLOR)
            screen.blit(technique_text, (panel_x + 30, y_offset))
            y_offset += 25
        
        # Back button
        back_rect = pygame.Rect(
            panel_x + panel_width // 2 - 100,
            panel_y + panel_height - 60,
            200,
            50
        )
        
        self.add_button("back", "Back to Menu", back_rect, action=lambda: self.set_screen("MENU"))
        
        # Draw UI elements
        self.draw_ui_elements(screen)
    
    def draw_ai_training_screen(self):
        """Draw the AI training screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = large_font.render("AI Training", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Training panel
        panel_width = 800
        panel_height = 600
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = 100
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_panel(screen, panel_rect)
        
        # Content
        y_offset = panel_y + 30
        
        # Training options
        options_title = medium_font.render("Training Options", True, WHITE)
        options_rect = options_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(options_title, options_rect)
        y_offset += 50
        
        # Genetic Algorithm button
        genetic_rect = pygame.Rect(
            panel_x + panel_width // 4 - 100,
            y_offset,
            200,
            50
        )
        
        self.add_button(
            "genetic",
            "Genetic Algorithm",
            genetic_rect,
            action=self.start_genetic_training
        )
        
        # Neural Network button
        nn_rect = pygame.Rect(
            panel_x + panel_width * 3 // 4 - 100,
            y_offset,
            200,
            50
        )
        
        nn_enabled = TORCH_AVAILABLE
        nn_text = "Neural Network" if nn_enabled else "Neural Network (PyTorch required)"
        
        self.add_button(
            "neural",
            nn_text,
            nn_rect,
            action=self.start_nn_training if nn_enabled else None
        )
        
        y_offset += 80
        
        # Training progress
        if self.ai_training_progress > 0:
            progress_title = medium_font.render("Training Progress", True, WHITE)
            progress_rect = progress_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
            screen.blit(progress_title, progress_rect)
            y_offset += 40
            
            # Progress bar
            progress_bar_rect = pygame.Rect(
                panel_x + 50,
                y_offset,
                panel_width - 100,
                30
            )
            
            draw_progress_bar(screen, progress_bar_rect, self.ai_training_progress, ACCENT_COLOR)
            
            # Progress text
            progress_text = medium_font.render(f"{int(self.ai_training_progress * 100)}%", True, WHITE)
            progress_text_rect = progress_text.get_rect(center=(panel_x + panel_width // 2, y_offset + 15))
            screen.blit(progress_text, progress_text_rect)
            
            y_offset += 50
            
            # Training visualization
            if self.genetic_optimizer and self.genetic_optimizer.best_fitness_history:
                # Create matplotlib figure
                plt.figure(figsize=(8, 4))
                plt.plot(self.genetic_optimizer.best_fitness_history, label='Best Fitness')
                plt.plot(self.genetic_optimizer.avg_fitness_history, label='Average Fitness')
                plt.title('Genetic Algorithm Training Progress')
                plt.xlabel('Generation')
                plt.ylabel('Fitness')
                plt.legend()
                plt.grid(True)
                
                # Convert to pygame surface
                plot_surface = plot_to_surface(plt.gcf(), panel_width - 100, 200)
                screen.blit(plot_surface, (panel_x + 50, y_offset))
                plt.close()
                
                y_offset += 220
            
            elif self.nn_trainer and self.nn_trainer.loss_history:
                # Create matplotlib figure
                plt.figure(figsize=(8, 4))
                plt.plot(self.nn_trainer.loss_history)
                plt.title('Neural Network Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.grid(True)
                
                # Convert to pygame surface
                plot_surface = plot_to_surface(plt.gcf(), panel_width - 100, 200)
                screen.blit(plot_surface, (panel_x + 50, y_offset))
                plt.close()
                
                y_offset += 220
        
        # Back button
        back_rect = pygame.Rect(
            panel_x + panel_width // 2 - 100,
            panel_y + panel_height - 60,
            200,
            50
        )
        
        self.add_button("back", "Back to Menu", back_rect, action=lambda: self.set_screen("MENU"))
        
        # Draw UI elements
        self.draw_ui_elements(screen)
    
    def update(self):
        """Update game state"""
        current_time = pygame.time.get_ticks()
        
        # Update game
        if self.current_screen == "GAME":
            self.game.update(current_time)
            
            # Handle AI moves
            if self.game.ai_active and not self.game.game_over and not self.game.paused:
                if current_time - self.game.last_ai_move_time > self.game.ai_delay:
                    self.game.apply_ai_move()
                    self.game.last_ai_move_time = current_time
    
    def set_screen(self, screen_name):
        """Set the current screen"""
        self.current_screen = screen_name
        self.buttons = {}  # Clear buttons when changing screens
    
    def start_game(self, mode):
        """Start a new game with the specified mode"""
        self.game.reset()
        self.game.game_mode = mode
        
        # Set up game mode specific settings
        if mode == "AI_PLAY":
            self.game.ai_active = True
        elif mode == "AI_BATTLE":
            # TODO: Set up AI battle mode
            pass
        elif mode == "CONSTRAINT":
            self.game.activate_constraint_mode()
        
        self.set_screen("GAME")
    
    def toggle_pause(self):
        """Toggle game pause state"""
        self.game.paused = not self.game.paused
    
    def set_ai_speed(self, value):
        """Set AI speed from slider value"""
        # Map 0-1 to 500ms-10ms (inverse relationship)
        self.game.ai_delay = int(500 - value * 490)
    
    def set_fall_speed(self, value):
        """Set fall speed from slider value"""
        # Map 0-1 to 500ms-50ms (inverse relationship)
        self.game.fall_speed = int(500 - value * 450)
    
    def toggle_ghost_piece(self, enabled):
        """Toggle ghost piece visibility"""
        self.game.ghost_piece_enabled = enabled
    
    def start_genetic_training(self):
        """Start genetic algorithm training"""
        self.genetic_optimizer = GeneticOptimizer(
            population_size=30,
            generations=20,
            mutation_rate=0.1
        )
        
        # Start training in a separate thread
        import threading
        
        def training_thread():
            best_weights = self.genetic_optimizer.train(
                callback=self.update_training_progress
            )
            
            # Apply the best weights
            self.game.ai_weights = best_weights
            print("Genetic training complete. Best weights:", best_weights)
            
            # Reset progress
            self.ai_training_progress = 0
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def start_nn_training(self):
        """Start neural network training"""
        if not TORCH_AVAILABLE:
            return
        
        self.nn_trainer = NeuralNetworkTrainer()
        
        # Start training in a separate thread
        import threading
        
        def training_thread():
            # Collect training data
            print("Collecting training data...")
            self.nn_trainer.collect_training_data(num_games=5)
            
            # Train the model
            print("Training neural network...")
            model = self.nn_trainer.train(
                epochs=50,
                batch_size=32,
                callback=self.update_training_progress
            )
            
            # Apply the trained model
            self.game.nn_model = model
            print("Neural network training complete.")
            
            # Reset progress
            self.ai_training_progress = 0
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def update_training_progress(self, progress, *args):
        """Update training progress for UI"""
        self.ai_training_progress = progress
    
    def draw(self):
        """Draw the current screen"""
        if self.current_screen == "MENU":
            self.draw_menu_screen()
        elif self.current_screen == "GAME":
            self.draw_game_screen()
        elif self.current_screen == "SETTINGS":
            self.draw_settings_screen()
        elif self.current_screen == "HELP":
            self.draw_help_screen()
        elif self.current_screen == "AI_TRAINING":
            self.draw_ai_training_screen()

def main():
    """Main function"""
    # Create game UI
    ui = GameUI()
    
    # Game loop
    running = True
    while running:
        # Handle events
        running = ui.handle_events()
        
        # Update game state
        ui.update()
        
        # Draw the current screen
        ui.draw()
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
