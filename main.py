"""
Main entry point for the Tetris AI game.
"""
import pygame
import sys
from constants import *
from utils import *
from ui import GameUI

def main():
    """Main function"""
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    pygame.mixer.init()
    
    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Advanced Tetris AI")
    clock = pygame.time.Clock()
    
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
        ui.draw(screen)
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
