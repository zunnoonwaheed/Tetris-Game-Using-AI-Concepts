"""
Utility functions for the Tetris AI game.
"""
import pygame
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from constants import *

# Initialize pygame
pygame.init()
pygame.font.init()

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

# Create sounds directory if it doesn't exist
os.makedirs("sounds", exist_ok=True)

# Dictionary to store loaded sound effects
sound_effects = {}

def load_sounds():
    """Load all sound effects"""
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
        play_sound("menu_select")
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

def create_heatmap(data, width, height, colormap='hot'):
    """Create a heatmap surface from data"""
    # Normalize data to 0-1
    if np.max(data) > np.min(data):
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        normalized_data = np.zeros_like(data)
    
    # Create a surface
    surf = pygame.Surface((width, height))
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Draw each cell
    cell_width = width / data.shape[1]
    cell_height = height / data.shape[0]
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = normalized_data[y, x]
            color = cmap(value)
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            
            rect = pygame.Rect(
                x * cell_width,
                y * cell_height,
                cell_width,
                cell_height
            )
            pygame.draw.rect(surf, color, rect)
    
    return surf

def get_fps():
    """Get current FPS"""
    return int(pygame.time.Clock().get_fps())

def format_time(milliseconds):
    """Format time in milliseconds to MM:SS format"""
    seconds = milliseconds / 1000
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def create_shadow_text(surface, text, font, color, shadow_color, x, y, align="left", shadow_offset=2):
    """Create text with a shadow effect"""
    # Draw shadow
    shadow_rect = draw_text(surface, text, font, shadow_color, x + shadow_offset, y + shadow_offset, align)
    
    # Draw text
    text_rect = draw_text(surface, text, font, color, x, y, align)
    
    return text_rect
