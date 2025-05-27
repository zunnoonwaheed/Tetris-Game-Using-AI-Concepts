import pygame
import random
import numpy as np
import time
from collections import deque
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 700
GRID_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SIDEBAR_WIDTH = 200

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Tetromino shapes and colors
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[1, 1, 0], [0, 1, 1]],  # Z
    [[0, 1, 1], [1, 1, 0]]   # S
]

COLORS = [CYAN, YELLOW, MAGENTA, ORANGE, BLUE, RED, GREEN]

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris with AI")
clock = pygame.time.Clock()

class Tetris:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.current_piece = self.new_piece()
        self.next_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.ai_mode = False
        self.ai_delay = 100  # ms between AI moves
        self.last_ai_move_time = 0
        self.fall_speed = 500  # ms
        self.last_fall_time = 0
    
    def new_piece(self):
        # Generate a new random piece
        shape_idx = random.randint(0, len(SHAPES) - 1)
        return {
            'shape': SHAPES[shape_idx],
            'color': COLORS[shape_idx],
            'x': self.width // 2 - len(SHAPES[shape_idx][0]) // 2,
            'y': 0
        }
    
    def valid_position(self, piece=None, x=None, y=None):
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
                    if y + row >= 0 and self.grid[y + row][x + col]:
                        return False
        return True
    
    def rotate_piece(self, clockwise=True):
        # Create a copy of the current piece
        new_piece = self.current_piece.copy()
        
        # Get the current shape
        shape = new_piece['shape']
        
        # Transpose the shape matrix
        shape = list(zip(*shape))
        
        # Reverse each row for clockwise rotation or each column for counter-clockwise
        if clockwise:
            shape = [list(row[::-1]) for row in shape]
        else:
            shape = list(reversed(shape))
            shape = [list(row) for row in shape]
        
        new_piece['shape'] = shape
        
        # Check if the rotated piece is in a valid position
        if self.valid_position(new_piece):
            self.current_piece = new_piece
    
    def move_piece(self, dx=0, dy=0):
        # Try to move the piece
        if self.valid_position(x=self.current_piece['x'] + dx, y=self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False
    
    def drop_piece(self):
        # Drop the piece as far as it can go
        while self.move_piece(dy=1):
            pass
        self.lock_piece()
    
    def lock_piece(self):
        # Lock the current piece in place and generate a new one
        shape = self.current_piece['shape']
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    # Add the piece to the grid
                    grid_y = self.current_piece['y'] + row
                    grid_x = self.current_piece['x'] + col
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        self.grid[grid_y][grid_x] = self.current_piece['color']
        
        # Check for completed lines
        self.clear_lines()
        
        # Get the next piece
        self.current_piece = self.next_piece
        self.next_piece = self.new_piece()
        
        # Check if the game is over
        if not self.valid_position():
            self.game_over = True
    
    def clear_lines(self):
        # Check for completed lines and remove them
        lines_to_clear = []
        for row in range(self.height):
            if all(self.grid[row]):
                lines_to_clear.append(row)
        
        # Remove the completed lines
        for row in lines_to_clear:
            del self.grid[row]
            self.grid.insert(0, [0 for _ in range(self.width)])
        
        # Update score
        if lines_to_clear:
            self.lines_cleared += len(lines_to_clear)
            self.score += (1, 2, 5, 10)[min(len(lines_to_clear) - 1, 3)] * 100 * self.level
            self.level = self.lines_cleared // 10 + 1
            self.fall_speed = max(100, 500 - (self.level - 1) * 50)
    
    def draw(self, screen):
        # Draw the grid
        for row in range(self.height):
            for col in range(self.width):
                pygame.draw.rect(
                    screen,
                    self.grid[row][col] if self.grid[row][col] else BLACK,
                    (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE),
                    0 if self.grid[row][col] else 1
                )
        
        # Draw the current piece
        shape = self.current_piece['shape']
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    pygame.draw.rect(
                        screen,
                        self.current_piece['color'],
                        ((self.current_piece['x'] + col) * GRID_SIZE,
                         (self.current_piece['y'] + row) * GRID_SIZE,
                         GRID_SIZE, GRID_SIZE),
                        0
                    )
        
        # Draw the sidebar
        sidebar_x = GRID_WIDTH * GRID_SIZE + 10
        
        # Draw the next piece preview
        next_piece_text = pygame.font.SysFont('Arial', 24).render('Next Piece:', True, WHITE)
        screen.blit(next_piece_text, (sidebar_x, 20))
        
        next_shape = self.next_piece['shape']
        for row in range(len(next_shape)):
            for col in range(len(next_shape[row])):
                if next_shape[row][col]:
                    pygame.draw.rect(
                        screen,
                        self.next_piece['color'],
                        (sidebar_x + col * GRID_SIZE, 60 + row * GRID_SIZE,
                         GRID_SIZE, GRID_SIZE),
                        0
                    )
        
        # Draw score, level, and lines cleared
        score_text = pygame.font.SysFont('Arial', 24).render(f'Score: {self.score}', True, WHITE)
        level_text = pygame.font.SysFont('Arial', 24).render(f'Level: {self.level}', True, WHITE)
        lines_text = pygame.font.SysFont('Arial', 24).render(f'Lines: {self.lines_cleared}', True, WHITE)
        ai_text = pygame.font.SysFont('Arial', 24).render(f'AI: {"ON" if self.ai_mode else "OFF"}', True, GREEN if self.ai_mode else RED)
        
        screen.blit(score_text, (sidebar_x, 150))
        screen.blit(level_text, (sidebar_x, 190))
        screen.blit(lines_text, (sidebar_x, 230))
        screen.blit(ai_text, (sidebar_x, 270))
        
        # Draw game over text if applicable
        if self.game_over:
            game_over_font = pygame.font.SysFont('Arial', 48)
            game_over_text = game_over_font.render('GAME OVER', True, RED)
            screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, 
                                         SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
    
    def get_grid_height(self):
        # Get the height of each column in the grid
        heights = [0] * self.width
        for col in range(self.width):
            for row in range(self.height):
                if self.grid[row][col]:
                    heights[col] = self.height - row
                    break
        return heights
    
    def count_holes(self):
        # Count the number of holes in the grid (empty cells with filled cells above them)
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if self.grid[row][col]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes
    
    def get_bumpiness(self, heights):
        # Calculate the bumpiness (sum of differences between adjacent columns)
        bumpiness = 0
        for i in range(self.width - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def count_complete_lines(self):
        # Count the number of complete lines
        complete_lines = 0
        for row in range(self.height):
            if all(self.grid[row]):
                complete_lines += 1
        return complete_lines
    
    def simulate_move(self, piece, x, rotation):
        # Create a copy of the current piece
        test_piece = piece.copy()
        test_piece['x'] = x
        
        # Apply rotation
        for _ in range(rotation):
            # Transpose
            shape = list(zip(*test_piece['shape']))
            # Reverse rows for clockwise rotation
            shape = [list(row[::-1]) for row in shape]
            test_piece['shape'] = shape
        
        # Check if the position is valid
        if not self.valid_position(test_piece):
            return None
        
        # Drop the piece
        while self.valid_position(test_piece, test_piece['x'], test_piece['y'] + 1):
            test_piece['y'] += 1
        
        # Create a copy of the grid
        new_grid = [row[:] for row in self.grid]
        
        # Add the piece to the grid
        shape = test_piece['shape']
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    grid_y = test_piece['y'] + row
                    grid_x = test_piece['x'] + col
                    if 0 <= grid_y < self.height and 0 <= grid_x < self.width:
                        new_grid[grid_y][grid_x] = test_piece['color']
        
        # Calculate metrics for the new grid
        heights = [0] * self.width
        for col in range(self.width):
            for row in range(self.height):
                if new_grid[row][col]:
                    heights[col] = self.height - row
                    break
        
        # Count holes
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height):
                if new_grid[row][col]:
                    found_block = True
                elif found_block:
                    holes += 1
        
        # Calculate bumpiness
        bumpiness = 0
        for i in range(self.width - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        # Count complete lines
        complete_lines = 0
        for row in range(self.height):
            if all(new_grid[row]):
                complete_lines += 1
        
        # Calculate aggregate height
        aggregate_height = sum(heights)
        
        # Calculate score using weights
        # These weights can be tuned for better AI performance
        weights = {
            'complete_lines': 0.760666,
            'holes': -0.35663,
            'bumpiness': -0.184483,
            'aggregate_height': -0.510066
        }
        
        score = (
            weights['complete_lines'] * complete_lines +
            weights['holes'] * holes +
            weights['bumpiness'] * bumpiness +
            weights['aggregate_height'] * aggregate_height
        )
        
        return {
            'piece': test_piece,
            'score': score,
            'metrics': {
                'complete_lines': complete_lines,
                'holes': holes,
                'bumpiness': bumpiness,
                'aggregate_height': aggregate_height
            }
        }
    
    def find_best_move(self):
        # Find the best move for the current piece using a search algorithm
        best_score = float('-inf')
        best_move = None
        
        # Try all possible positions and rotations
        for rotation in range(4):  # Maximum 4 rotations
            for x in range(-2, self.width + 2):  # Try different x positions
                result = self.simulate_move(self.current_piece, x, rotation)
                if result and result['score'] > best_score:
                    best_score = result['score']
                    best_move = {
                        'x': x,
                        'rotation': rotation,
                        'score': best_score,
                        'metrics': result['metrics']
                    }
        
        return best_move
    
    def apply_ai_move(self):
        # Find and apply the best move
        best_move = self.find_best_move()
        if not best_move:
            return
        
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

# Main game loop
def main():
    game = Tetris()
    
    # Game loop
    running = True
    while running:
        current_time = pygame.time.get_ticks()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if not game.game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        game.move_piece(dx=-1)
                    elif event.key == pygame.K_RIGHT:
                        game.move_piece(dx=1)
                    elif event.key == pygame.K_DOWN:
                        game.move_piece(dy=1)
                    elif event.key == pygame.K_UP:
                        game.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        game.drop_piece()
                    elif event.key == pygame.K_a:
                        # Toggle AI mode
                        game.ai_mode = not game.ai_mode
                    elif event.key == pygame.K_r:
                        # Reset game
                        game = Tetris()
        
        # AI move
        if game.ai_mode and not game.game_over and current_time - game.last_ai_move_time > game.ai_delay:
            game.apply_ai_move()
            game.last_ai_move_time = current_time
        
        # Natural falling
        if not game.game_over and current_time - game.last_fall_time > game.fall_speed:
            if not game.move_piece(dy=1):
                game.lock_piece()
            game.last_fall_time = current_time
        
        # Clear the screen
        screen.fill(BLACK)
        
        # Draw the game
        game.draw(screen)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "Left/Right: Move",
            "Up: Rotate",
            "Down: Soft Drop",
            "Space: Hard Drop",
            "A: Toggle AI",
            "R: Reset Game"
        ]
        
        for i, instruction in enumerate(instructions):
            text = pygame.font.SysFont('Arial', 18).render(instruction, True, WHITE)
            screen.blit(text, (GRID_WIDTH * GRID_SIZE + 10, 350 + i * 25))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
