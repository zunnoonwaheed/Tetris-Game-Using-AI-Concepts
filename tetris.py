"""
Core Tetris game logic.
"""
import pygame
import random
import numpy as np
import time
from collections import deque
from constants import *
from utils import play_sound

class Tetris:
    """Main Tetris game class"""
    def __init__(self, width=GRID_WIDTH, height=GRID_HEIGHT):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        
        # Initialize all attributes that might be accessed in methods called during initialization
        self.piece_stats = {name: 0 for name in SHAPE_NAMES}
        self.piece_sequence = deque(maxlen=10)  # Store last 10 pieces
        self.clear_history = deque(maxlen=10)   # Store last 10 line clears
        self.score_history = deque(maxlen=100)  # Store score history for graph
        self.heatmap_data = np.zeros((height, width))
        self.ai_weights = DEFAULT_AI_WEIGHTS.copy()
        
        # Now it's safe to call new_piece()
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
        self.difficulty = "MEDIUM"
        self.ai_active = False
        self.ai_delay = DIFFICULTY_LEVELS["MEDIUM"]["ai_delay"]  # ms
        self.last_ai_move_time = 0
        self.ai_move_history = []
        self.ai_evaluation_history = []
        self.constraint_mode = None
        self.constraint_timer = 0
        self.constraint_duration = CONSTRAINT_DURATION
        self.stats = {
            "pieces_placed": 0,
            "tetris_clears": 0,
            "max_score": 0,
            "max_level": 1,
            "total_games": 0,
            "ai_wins": 0,
            "player_wins": 0,
            "game_time": 0,
            "start_time": 0
        }
        
        # Start time tracking
        self.stats["start_time"] = time.time()
    
    def new_piece(self):
        """Generate a new random piece with defensive programming"""
        # Defensive checks for required attributes
        if not hasattr(self, 'piece_stats'):
            self.piece_stats = {name: 0 for name in SHAPE_NAMES}
            
        if not hasattr(self, 'piece_sequence'):
            self.piece_sequence = deque(maxlen=10)
            
        shape_idx = random.randint(0, len(SHAPES) - 1)
        
        # Update piece statistics
        self.piece_stats[SHAPE_NAMES[shape_idx]] += 1
        
        # Add to piece sequence
        self.piece_sequence.append(shape_idx)
        
        return {
            'shape': [row[:] for row in SHAPES[shape_idx]],  # Deep copy
            'color': COLORS[shape_idx],
            'x': self.width // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0,
            'rotation': 0,
            'type': shape_idx
        }
    
    def reset(self):
        """Reset the game state with defensive programming"""
        # Ensure all collections are initialized
        if not hasattr(self, 'piece_sequence'):
            self.piece_sequence = deque(maxlen=10)
        if not hasattr(self, 'clear_history'):
            self.clear_history = deque(maxlen=10)
        if not hasattr(self, 'score_history'):
            self.score_history = deque(maxlen=100)
        if not hasattr(self, 'heatmap_data'):
            self.heatmap_data = np.zeros((self.height, self.width))
            
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
        
        # Set fall speed based on difficulty
        self.fall_speed = DIFFICULTY_LEVELS[self.difficulty]["fall_speed"]
        self.ai_delay = DIFFICULTY_LEVELS[self.difficulty]["ai_delay"]
        
        self.last_fall_time = pygame.time.get_ticks()
        self.lock_timer = 0
        self.move_resets = 0
        self.ai_move_history = []
        self.ai_evaluation_history = []
        self.constraint_mode = None
        self.constraint_timer = 0
        self.heatmap_data = np.zeros((self.height, self.width))
        self.piece_sequence.clear()
        self.clear_history.clear()
        self.score_history.clear()
        
        # Update stats
        if not hasattr(self, 'stats'):
            self.stats = {
                "pieces_placed": 0,
                "tetris_clears": 0,
                "max_score": 0,
                "max_level": 1,
                "total_games": 0,
                "ai_wins": 0,
                "player_wins": 0,
                "game_time": 0,
                "start_time": 0
            }
        self.stats["total_games"] += 1
        self.stats["start_time"] = time.time()
        
        # Reset piece stats
        if not hasattr(self, 'piece_stats'):
            self.piece_stats = {name: 0 for name in SHAPE_NAMES}
        else:
            for name in SHAPE_NAMES:
                self.piece_stats[name] = 0
    
    def valid_position(self, piece=None, x=None, y=None):
        """Check if the piece is in a valid position"""
        if piece is None:
            piece = self.current_piece
        if x is None:
            x = piece['x']
        if y is None:
            y = piece['y']
        
        # Defensive check for grid
        if not hasattr(self, 'grid') or self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=int)
            
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
            play_sound("rotate")
            return True
        
        # Wall kick - try to move the piece left or right if rotation is blocked
        for offset in [-1, 1, -2, 2]:
            new_piece_offset = new_piece.copy()
            new_piece_offset['x'] += offset
            if self.valid_position(new_piece_offset):
                self.current_piece = new_piece_offset
                self._reset_lock_timer()
                play_sound("rotate")
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
                
            # Play sound for horizontal movement
            if dx != 0:
                play_sound("move")
                
            return True
        return False
    
    def _reset_lock_timer(self):
        """Reset the lock timer if move resets are available"""
        if not hasattr(self, 'move_resets'):
            self.move_resets = 0
        if not hasattr(self, 'move_reset_limit'):
            self.move_reset_limit = 15
            
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
            
            # Defensive check for level
            if not hasattr(self, 'level'):
                self.level = 1
                
            self.score += drop_distance * (self.level // 2 + 1)  # Bonus points for hard drop
            play_sound("drop")
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
        
        play_sound("hold")
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
                        play_sound("game_over")
                        return False
        
        # Update statistics
        if not hasattr(self, 'stats'):
            self.stats = {
                "pieces_placed": 0,
                "tetris_clears": 0,
                "max_score": 0,
                "max_level": 1,
                "total_games": 0,
                "ai_wins": 0,
                "player_wins": 0,
                "game_time": 0,
                "start_time": 0
            }
        self.stats["pieces_placed"] += 1
        
        # Update heatmap data
        if not hasattr(self, 'heatmap_data'):
            self.heatmap_data = np.zeros((self.height, self.width))
            
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    grid_y = self.current_piece['y'] + row
                    grid_x = self.current_piece['x'] + col
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        self.heatmap_data[grid_y][grid_x] += 1
        
        # Check for completed lines
        lines_cleared = self.clear_lines()
        
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
            play_sound("game_over")
            
            # Update max stats
            self.stats["max_score"] = max(self.stats["max_score"], self.score)
            self.stats["max_level"] = max(self.stats["max_level"], self.level)
            self.stats["game_time"] += time.time() - self.stats["start_time"]
            
            return False
            
        return True
    
    def clear_lines(self):
        """Clear completed lines and update score"""
        # Defensive check for grid
        if not hasattr(self, 'grid') or self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=int)
            
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
        
        # Add to clear history
        if not hasattr(self, 'clear_history'):
            self.clear_history = deque(maxlen=10)
        self.clear_history.append(lines_count)
        
        # Update combo
        self.combo += 1
        
        # Classic Tetris scoring with combo bonus
        line_scores = [100, 300, 500, 800]  # 1, 2, 3, 4 lines
        score_gain = line_scores[min(lines_count, 4) - 1] * self.level
        combo_bonus = (self.combo - 1) * 50 * self.level if self.combo > 1 else 0
        self.score += score_gain + combo_bonus
        
        # Update score history
        if not hasattr(self, 'score_history'):
            self.score_history = deque(maxlen=100)
        self.score_history.append(self.score)
        
        # Update level
        old_level = self.level
        self.level = max(1, self.lines_cleared // 10 + 1)
        
        # Play sound effects
        if lines_count == 4:
            play_sound("tetris")
            if hasattr(self, 'stats'):
                self.stats["tetris_clears"] += 1
        else:
            play_sound("clear")
            
        # Level up sound
        if self.level > old_level:
            play_sound("level_up")
        
        # Adjust fall speed based on level
        self.fall_speed = max(50, 500 - (self.level - 1) * 20)
        
        return lines_count
    
    def get_ghost_piece(self):
        """Get the position where the current piece would land"""
        if not hasattr(self, 'ghost_piece_enabled'):
            self.ghost_piece_enabled = True
            
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
        
        # Defensive checks
        if not hasattr(self, 'last_fall_time'):
            self.last_fall_time = current_time
        if not hasattr(self, 'fall_speed'):
            self.fall_speed = 500
        if not hasattr(self, 'lock_timer'):
            self.lock_timer = 0
        if not hasattr(self, 'lock_delay'):
            self.lock_delay = 500
        if not hasattr(self, 'constraint_mode'):
            self.constraint_mode = None
        if not hasattr(self, 'constraint_timer'):
            self.constraint_timer = 0
        if not hasattr(self, 'constraint_duration'):
            self.constraint_duration = CONSTRAINT_DURATION
            
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
        if not hasattr(self, 'constraint_mode') or not self.constraint_mode:
            return
            
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
        elif self.constraint_mode == "INVISIBLE":
            # Pieces become invisible after placement
            # This is handled in the drawing code
            pass
        elif self.constraint_mode == "MIRROR":
            # Controls are mirrored
            # This is handled in the input processing
            pass
    
    def activate_constraint_mode(self):
        """Activate a random constraint mode"""
        self.constraint_mode = random.choice(CONSTRAINT_MODES)
        self.constraint_timer = pygame.time.get_ticks()
    
    def draw(self, surface, offset_x=0, offset_y=0, small=False, invisible=False):
        """Draw the game board"""
        # Defensive checks
        if not hasattr(self, 'grid') or self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=int)
        if not hasattr(self, 'ghost_piece_enabled'):
            self.ghost_piece_enabled = True
        if not hasattr(self, 'constraint_mode'):
            self.constraint_mode = None
            
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
        if not small and not self.game_over and not self.paused and self.ghost_piece_enabled:
            ghost_piece = self.get_ghost_piece()
            if ghost_piece and self.constraint_mode != "INVISIBLE":
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
                    
                    # Skip drawing if in invisible mode and not a constraint block
                    if invisible and self.constraint_mode == "INVISIBLE" and cell_value != -1:
                        continue
                    
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
        if not self.game_over and not small and not (invisible and self.constraint_mode == "INVISIBLE"):
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
        # Defensive check for grid
        if not hasattr(self, 'grid') or self.grid is None:
            self.grid = np.zeros((self.height, self.width), dtype=int)
            
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
        """Evaluate the current board state using heuristics"""
        # Defensive check for ai_weights
        if not hasattr(self, 'ai_weights'):
            self.ai_weights = DEFAULT_AI_WEIGHTS.copy()
            
        if features is None:
            features = self.get_board_features()
        
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
        # Defensive checks
        if not hasattr(self, 'ai_evaluation_history'):
            self.ai_evaluation_history = []
            
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
        # Defensive checks
        if not hasattr(self, 'ai_move_history'):
            self.ai_move_history = []
            
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
        
        # Play AI move sound
        play_sound("ai_move")
        
        # Drop the piece
        self.drop_piece()
        return True