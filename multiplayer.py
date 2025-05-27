"""
Multiplayer mode implementation for Tetris AI game.
"""
import pygame
import random
import time
from constants import *
from tetris import Tetris
from utils import play_sound

class MultiplayerManager:
    """Manager for multiplayer games"""
    def __init__(self, mode="VS_AI"):
        self.mode = mode
        self.player1_game = Tetris()
        self.player2_game = Tetris()
        self.player1_score = 0
        self.player2_score = 0
        self.game_active = True
        self.winner = None
        
        # Set up player 2 based on mode
        if mode != "VS_PLAYER":
            self.player2_game.ai_active = True
        
        # For cooperative mode
        self.shared_score = 0
        self.shared_level = 1
        self.shared_lines = 0
        
        # For battle mode
        self.attack_queue1 = []  # Lines to add to player 1
        self.attack_queue2 = []  # Lines to add to player 2
    
    def reset(self):
        """Reset the multiplayer games"""
        self.player1_game.reset()
        self.player2_game.reset()
        self.player1_score = 0
        self.player2_score = 0
        self.game_active = True
        self.winner = None
        self.attack_queue1 = []
        self.attack_queue2 = []
        
        # Set up player 2 based on mode
        if self.mode != "VS_PLAYER":
            self.player2_game.ai_active = True
    
    def update(self, current_time):
        """Update both games"""
        if not self.game_active:
            return
            
        # Update games
        self.player1_game.update(current_time)
        self.player2_game.update(current_time)
        
        # Handle game over conditions
        if self.player1_game.game_over and self.player2_game.game_over:
            self.game_active = False
            
            # Determine winner
            if self.player1_game.score > self.player2_game.score:
                self.winner = "PLAYER1"
            elif self.player2_game.score > self.player1_game.score:
                self.winner = "PLAYER2"
            else:
                self.winner = "TIE"
        
        # Handle battle mode line attacks
        if self.mode == "VS_AI" or self.mode == "VS_PLAYER":
            # Process attack queues
            self._process_attack_queues()
    
    def _process_attack_queues(self):
        """Process line attack queues for battle mode"""
        # Process attacks for player 1
        if self.attack_queue1 and not self.player1_game.game_over:
            lines = self.attack_queue1.pop(0)
            self._add_garbage_lines(self.player1_game, lines)
        
        # Process attacks for player 2
        if self.attack_queue2 and not self.player2_game.game_over:
            lines = self.attack_queue2.pop(0)
            self._add_garbage_lines(self.player2_game, lines)
    
    def _add_garbage_lines(self, game, num_lines):
        """Add garbage lines to the bottom of the game board"""
        if num_lines <= 0:
            return
            
        # Shift existing grid up
        game.grid = game.grid[num_lines:, :]
        
        # Add garbage lines at the bottom
        for _ in range(num_lines):
            # Create a garbage line with one hole
            garbage_line = np.ones(game.width, dtype=int)
            hole_pos = random.randint(0, game.width - 1)
            garbage_line[hole_pos] = 0
            
            # Add the line to the bottom
            game.grid = np.vstack([game.grid, garbage_line])
        
        # Play sound effect
        play_sound("clear")
    
    def handle_line_clear(self, player, lines_cleared):
        """Handle line clear events in battle mode"""
        if lines_cleared <= 0 or self.mode == "COOPERATIVE":
            return
            
        # Calculate attack lines (typically lines cleared - 1)
        attack_lines = max(0, lines_cleared - 1)
        
        # Add attack to the appropriate queue
        if player == "PLAYER1" and attack_lines > 0:
            self.attack_queue2.append(attack_lines)
        elif player == "PLAYER2" and attack_lines > 0:
            self.attack_queue1.append(attack_lines)
    
    def handle_cooperative_clear(self, lines_cleared):
        """Handle line clear events in cooperative mode"""
        if lines_cleared <= 0 or self.mode != "COOPERATIVE":
            return
            
        # Update shared score and level
        self.shared_lines += lines_cleared
        self.shared_level = max(1, self.shared_lines // 10 + 1)
        
        # Calculate score
        line_scores = [100, 300, 500, 800]  # 1, 2, 3, 4 lines
        score_gain = line_scores[min(lines_cleared, 4) - 1] * self.shared_level
        self.shared_score += score_gain
        
        # Update both games
        self.player1_game.level = self.shared_level
        self.player2_game.level = self.shared_level
        self.player1_game.score = self.shared_score
        self.player2_game.score = self.shared_score
