"""
Sound management for the Tetris AI game.
"""
import pygame
import os
from constants import *

# Initialize pygame mixer
pygame.mixer.init()

# Dictionary to store loaded sound effects
sound_effects = {}

def load_sounds():
    """Load all sound effects"""
    # Create sounds directory if it doesn't exist
    os.makedirs("sounds", exist_ok=True)
    
    for name, path in SOUND_EFFECTS.items():
        try:
            # Create an empty sound file if it doesn't exist
            if not os.path.exists(path):
                with open(path, 'wb') as f:
                    f.write(b'')
            sound_effects[name] = pygame.mixer.Sound(path)
        except:
            print(f"Could not load sound: {path}")

def play_sound(sound_name, volume=0.5):
    """Play a sound effect"""
    if sound_name in sound_effects:
        sound_effects[sound_name].set_volume(volume)
        sound_effects[sound_name].play()

def play_music(volume=0.3):
    """Play background music"""
    try:
        if not os.path.exists(BACKGROUND_MUSIC):
            # Create an empty music file if it doesn't exist
            with open(BACKGROUND_MUSIC, 'wb') as f:
                f.write(b'')
        pygame.mixer.music.load(BACKGROUND_MUSIC)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1)  # Loop indefinitely
    except:
        print(f"Could not load music: {BACKGROUND_MUSIC}")

def stop_music():
    """Stop background music"""
    pygame.mixer.music.stop()
