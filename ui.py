"""
User interface components for the Tetris AI game.
"""
import pygame
import numpy as np
import time
import os
from collections import deque
import matplotlib.pyplot as plt
from constants import *
from utils import *
from tetris import Tetris
from ai import *

class GameUI:
    """User interface for the Tetris game"""
    def __init__(self):
        self.game = Tetris()
        self.current_screen = "MENU"  # MENU, GAME, SETTINGS, AI_TRAINING, HELP, MULTIPLAYER
        self.ai_training_progress = 0
        self.genetic_optimizer = None
        self.nn_trainer = None
        self.nn_model = load_nn_model()
        self.sound_enabled = True
        self.music_enabled = True
        self.show_ai_visualization = True
        self.show_heatmap = False
        self.show_stats = True
        
        # UI elements
        self.buttons = {}
        self.sliders = {}
        self.checkboxes = {}
        self.dropdowns = {}
        self.active_dropdown = None
        
        # Performance metrics
        self.fps_history = deque(maxlen=60)
        self.last_time = time.time()
        
        # Multiplayer
        self.multiplayer_mode = "VS_AI"
        self.player1_score = 0
        self.player2_score = 0
        self.player1_game = self.game
        self.player2_game = Tetris()
        self.player2_game.ai_active = True
        
        # Load sounds
        load_sounds()
        
        # Start background music
        if self.music_enabled:
            play_music()
    
    def handle_events(self):
        """Handle pygame events"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_clicked = True
                # Close any open dropdown when clicking elsewhere
                if self.active_dropdown and not self.dropdowns[self.active_dropdown]['rect'].collidepoint(mouse_pos):
                    self.active_dropdown = None
            
            if self.current_screen == "GAME" and not self.game.game_over and not self.game.paused and not self.game.ai_active:
                if event.type == pygame.KEYDOWN:
                    # Handle mirrored controls if constraint mode is active
                    if self.game.constraint_mode == "MIRROR":
                        if event.key == pygame.K_LEFT:
                            self.game.move_piece(dx=1)
                        elif event.key == pygame.K_RIGHT:
                            self.game.move_piece(dx=-1)
                        elif event.key == pygame.K_DOWN:
                            self.game.move_piece(dy=1)
                        elif event.key == pygame.K_UP:
                            self.game.rotate_piece(clockwise=False)
                    else:
                        if event.key == pygame.K_LEFT:
                            self.game.move_piece(dx=-1)
                        elif event.key == pygame.K_RIGHT:
                            self.game.move_piece(dx=1)
                        elif event.key == pygame.K_DOWN:
                            self.game.move_piece(dy=1)
                        elif event.key == pygame.K_UP:
                            self.game.rotate_piece()
                    
                    # Common controls regardless of constraint mode
                    if event.key == pygame.K_SPACE:
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
                    elif event.key == pygame.K_h:
                        self.show_heatmap = not self.show_heatmap
                    elif event.key == pygame.K_v:
                        self.show_ai_visualization = not self.show_ai_visualization
                    elif event.key == pygame.K_s:
                        self.show_stats = not self.show_stats
            
            elif self.current_screen == "MULTIPLAYER" and not self.player1_game.paused:
                if event.type == pygame.KEYDOWN:
                    # Player 1 controls
                    if event.key == pygame.K_LEFT:
                        self.player1_game.move_piece(dx=-1)
                    elif event.key == pygame.K_RIGHT:
                        self.player1_game.move_piece(dx=1)
                    elif event.key == pygame.K_DOWN:
                        self.player1_game.move_piece(dy=1)
                    elif event.key == pygame.K_UP:
                        self.player1_game.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        self.player1_game.drop_piece()
                    elif event.key == pygame.K_c:
                        self.player1_game.hold_piece()
                    
                    # Player 2 controls (if not AI)
                    if self.multiplayer_mode == "VS_PLAYER":
                        if event.key == pygame.K_a:
                            self.player2_game.move_piece(dx=-1)
                        elif event.key == pygame.K_d:
                            self.player2_game.move_piece(dx=1)
                        elif event.key == pygame.K_s:
                            self.player2_game.move_piece(dy=1)
                        elif event.key == pygame.K_w:
                            self.player2_game.rotate_piece()
                        elif event.key == pygame.K_q:
                            self.player2_game.drop_piece()
                        elif event.key == pygame.K_e:
                            self.player2_game.hold_piece()
                    
                    # Common controls
                    if event.key == pygame.K_p:
                        self.player1_game.paused = not self.player1_game.paused
                        self.player2_game.paused = self.player1_game.paused
                    elif event.key == pygame.K_r:
                        self.reset_multiplayer()
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
        
        # Update dropdowns
        for dropdown_id, dropdown in self.dropdowns.items():
            dropdown_rect = dropdown['rect']
            
            # Check if dropdown header is clicked
            if dropdown_rect.collidepoint(mouse_pos) and mouse_clicked:
                if self.active_dropdown == dropdown_id:
                    self.active_dropdown = None
                else:
                    self.active_dropdown = dropdown_id
                    play_sound("menu_navigate")
            
            # Check if dropdown is open and an option is clicked
            if self.active_dropdown == dropdown_id and mouse_clicked:
                for i, option in enumerate(dropdown['options']):
                    option_rect = pygame.Rect(
                        dropdown_rect.left,
                        dropdown_rect.bottom + i * 30,
                        dropdown_rect.width,
                        30
                    )
                    
                    if option_rect.collidepoint(mouse_pos):
                        dropdown['selected'] = option
                        self.active_dropdown = None
                        
                        # Call callback if provided
                        if dropdown['callback']:
                            dropdown['callback'](option)
                        
                        play_sound("menu_select")
                        break
    
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
    
    def add_dropdown(self, dropdown_id, text, rect, options, selected=None, callback=None):
        """Add a dropdown to the UI"""
        if selected is None and options:
            selected = options[0]
            
        self.dropdowns[dropdown_id] = {
            'text': text,
            'rect': pygame.Rect(rect),
            'options': options,
            'selected': selected,
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
        
        # Draw dropdowns
        for dropdown_id, dropdown in self.dropdowns.items():
            # Draw dropdown header
            color = HIGHLIGHT_COLOR if self.active_dropdown == dropdown_id else PANEL_COLOR
            pygame.draw.rect(surface, color, dropdown['rect'], border_radius=5)
            pygame.draw.rect(surface, LIGHT_GRAY, dropdown['rect'], 2, border_radius=5)
            
            # Draw selected option
            text = f"{dropdown['text']}: {dropdown['selected']}"
            text_surf = small_font.render(text, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(
                left=dropdown['rect'].left + 10,
                centery=dropdown['rect'].centery
            )
            surface.blit(text_surf, text_rect)
            
            # Draw dropdown arrow
            arrow_points = [
                (dropdown['rect'].right - 20, dropdown['rect'].centery - 5),
                (dropdown['rect'].right - 10, dropdown['rect'].centery + 5),
                (dropdown['rect'].right - 30, dropdown['rect'].centery + 5)
            ]
            pygame.draw.polygon(surface, TEXT_COLOR, arrow_points)
            
            # Draw dropdown options if active
            if self.active_dropdown == dropdown_id:
                for i, option in enumerate(dropdown['options']):
                    option_rect = pygame.Rect(
                        dropdown['rect'].left,
                        dropdown['rect'].bottom + i * 30,
                        dropdown['rect'].width,
                        30
                    )
                    
                    # Highlight selected option
                    bg_color = HIGHLIGHT_COLOR if option == dropdown['selected'] else PANEL_COLOR
                    pygame.draw.rect(surface, bg_color, option_rect)
                    pygame.draw.rect(surface, LIGHT_GRAY, option_rect, 1)
                    
                    # Draw option text
                    option_text = small_font.render(option, True, TEXT_COLOR)
                    option_text_rect = option_text.get_rect(
                        left=option_rect.left + 10,
                        centery=option_rect.centery
                    )
                    surface.blit(option_text, option_text_rect)
    
    def draw_menu_screen(self, screen):
        """Draw the main menu screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = title_font.render("Advanced Tetris AI", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = medium_font.render("With Multiple AI Techniques", True, ACCENT_COLOR)
        subtitle_rect = subtitle_text.get_rect(center=(SCREEN_WIDTH // 2, 150))
        screen.blit(subtitle_text, subtitle_rect)
        
        # Menu buttons
        button_width = 300
        button_height = 60
        button_spacing = 20
        start_y = 220
        
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
        version_text = small_font.render("v1.0.0 - AI Final Project", True, LIGHT_GRAY)
        version_rect = version_text.get_rect(bottomright=(SCREEN_WIDTH - 20, SCREEN_HEIGHT - 20))
        screen.blit(version_text, version_rect)
        
        # Credits
        credits_text = small_font.render("Created by: Your Team Names", True, LIGHT_GRAY)
        credits_rect = credits_text.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20))
        screen.blit(credits_text, credits_rect)
    
    def draw_game_screen(self, screen):
        """Draw the game screen"""
        screen.fill(BG_COLOR)
        
        # Calculate game board position
        board_width = self.game.width * GRID_SIZE
        board_height = self.game.height * GRID_SIZE
        board_x = (SCREEN_WIDTH - board_width - SIDEBAR_WIDTH) // 2
        board_y = (SCREEN_HEIGHT - board_height) // 2
        
        # Draw game board
        self.game.draw(screen, board_x, board_y, invisible=False)
        
        # Draw heatmap overlay if enabled
        if self.show_heatmap:
            heatmap_surf = create_heatmap(self.game.heatmap_data, board_width, board_height)
            heatmap_surf.set_alpha(128)  # Semi-transparent
            screen.blit(heatmap_surf, (board_x, board_y))
        
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
        
        # Difficulty
        difficulty_text = small_font.render(f"Difficulty: {self.game.difficulty}", True, ACCENT_COLOR)
        difficulty_rect = difficulty_text.get_rect(center=(sidebar_x + sidebar_width // 2, y_offset))
        screen.blit(difficulty_text, difficulty_rect)
        y_offset += 30
        
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
                "H: Toggle Heatmap",
                "V: Toggle AI Viz",
                "S: Toggle Stats",
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
        if self.show_ai_visualization and self.game.ai_evaluation_history and not self.game.game_over and not self.game.paused:
            self.draw_ai_visualization(screen, board_x, board_y + board_height + 20)
        
        # Draw statistics if enabled
        if self.show_stats and not self.game.game_over and not self.game.paused:
            self.draw_game_statistics(screen, board_x - 220, board_y)
        
        # Draw UI elements
        self.draw_ui_elements(screen)
        
        # Draw FPS counter
        current_time = time.time()
        self.fps_history.append(1.0 / max(0.001, current_time - self.last_time))
        self.last_time = current_time
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        fps_text = tiny_font.render(f"FPS: {avg_fps:.1f}", True, LIGHT_GRAY)
        screen.blit(fps_text, (10, 10))
    
    def draw_ai_visualization(self, screen, x, y):
        """Draw visualization of AI decision making"""
        width = self.game.width * GRID_SIZE
        height = 200
        
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
                bar_y = y + 40 + i * 25
                bar_width = (width - 20) * norm_eval
                bar_height = 20
                
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
            
            # Draw feature importance visualization
            y_offset = y + 40 + 5 * 25 + 10
            
            # Title
            features_text = small_font.render("Feature Importance", True, WHITE)
            features_rect = features_text.get_rect(center=(x + width // 2, y_offset))
            screen.blit(features_text, features_rect)
            y_offset += 25
            
            # Draw feature bars
            if sorted_evals and 'features' in sorted_evals[0]:
                best_move = sorted_evals[0]
                features = best_move['features']
                
                # Calculate weighted features
                weighted_features = {
                    'Height': self.game.ai_weights['height'] * features['aggregate_height'],
                    'Bumpiness': self.game.ai_weights['bumpiness'] * features['bumpiness'],
                    'Holes': self.game.ai_weights['holes'] * features['holes'],
                    'Lines': self.game.ai_weights['lines_cleared'] * features['complete_lines']
                }
                
                # Find min and max for normalization
                values = list(weighted_features.values())
                min_val = min(values)
                max_val = max(values)
                val_range = max(0.001, max_val - min_val)
                
                # Draw bars
                for i, (name, value) in enumerate(weighted_features.items()):
                    # Normalize value to [0, 1]
                    norm_val = (value - min_val) / val_range
                    
                    # Bar position and size
                    bar_x = x + 10
                    bar_y = y_offset + i * 20
                    bar_width = (width - 20) * abs(norm_val)
                    bar_height = 15
                    
                    # Bar color based on positive/negative
                    bar_color = GREEN if value >= 0 else RED
                    
                    # Draw bar
                    pygame.draw.rect(
                        screen,
                        bar_color,
                        (bar_x, bar_y, bar_width, bar_height),
                        border_radius=3
                    )
                    
                    # Draw text
                    text = tiny_font.render(f"{name}: {value:.2f}", True, WHITE)
                    screen.blit(text, (bar_x + 5, bar_y + 1))
    
    def draw_game_statistics(self, screen, x, y):
        """Draw game statistics panel"""
        width = 200
        height = self.game.height * GRID_SIZE
        
        # Background panel
        panel_rect = pygame.Rect(x, y, width, height)
        draw_panel(screen, panel_rect)
        
        # Title
        title_text = small_font.render("Game Statistics", True, WHITE)
        title_rect = title_text.get_rect(center=(x + width // 2, y + 15))
        screen.blit(title_text, title_rect)
        
        y_offset = y + 40
        
        # Game time
        game_time = time.time() - self.game.stats["start_time"]
        time_text = small_font.render(f"Time: {format_time(int(game_time * 1000))}", True, WHITE)
        screen.blit(time_text, (x + 10, y_offset))
        y_offset += 30
        
        # Piece statistics
        pieces_text = small_font.render("Piece Distribution:", True, WHITE)
        screen.blit(pieces_text, (x + 10, y_offset))
        y_offset += 25
        
        # Calculate total pieces
        total_pieces = sum(self.game.piece_stats.values())
        if total_pieces > 0:
            # Draw piece distribution bars
            for i, (name, count) in enumerate(self.game.piece_stats.items()):
                # Bar position and size
                bar_x = x + 10
                bar_y = y_offset + i * 20
                bar_width = (width - 20) * (count / total_pieces)
                bar_height = 15
                
                # Get piece color
                piece_idx = SHAPE_NAMES.index(name)
                bar_color = COLORS[piece_idx]
                
                # Draw bar
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
                text = tiny_font.render(f"{name}: {count}", True, WHITE)
                screen.blit(text, (bar_x + 5, bar_y + 1))
            
            y_offset += len(self.game.piece_stats) * 20 + 10
        
        # Line clear history
        if self.game.clear_history:
            clears_text = small_font.render("Recent Line Clears:", True, WHITE)
            screen.blit(clears_text, (x + 10, y_offset))
            y_offset += 25
            
            # Draw line clear history
            history_width = width - 20
            history_height = 40
            history_x = x + 10
            history_y = y_offset
            
            # Draw background
            pygame.draw.rect(
                screen,
                DARK_GRAY,
                (history_x, history_y, history_width, history_height),
                border_radius=3
            )
            
            # Draw line clear bars
            bar_width = history_width / len(self.game.clear_history)
            for i, lines in enumerate(self.game.clear_history):
                # Bar height based on lines cleared (1-4)
                bar_height = (lines / 4) * history_height
                
                # Bar position
                bar_x = history_x + i * bar_width
                bar_y = history_y + history_height - bar_height
                
                # Bar color based on lines cleared
                if lines == 4:
                    bar_color = MAGENTA  # Tetris
                elif lines == 3:
                    bar_color = ORANGE
                elif lines == 2:
                    bar_color = YELLOW
                else:
                    bar_color = GREEN
                
                # Draw bar
                pygame.draw.rect(
                    screen,
                    bar_color,
                    (bar_x, bar_y, bar_width - 1, bar_height)
                )
            
            # Draw border
            pygame.draw.rect(
                screen,
                LIGHT_GRAY,
                (history_x, history_y, history_width, history_height),
                1,
                border_radius=3
            )
            
            y_offset += history_height + 20
        
        # Score history graph
        if len(self.game.score_history) > 1:
            score_text = small_font.render("Score Progress:", True, WHITE)
            screen.blit(score_text, (x + 10, y_offset))
            y_offset += 25
            
            # Draw score graph
            graph_width = width - 20
            graph_height = 60
            graph_x = x + 10
            graph_y = y_offset
            
            # Draw background
            pygame.draw.rect(
                screen,
                DARK_GRAY,
                (graph_x, graph_y, graph_width, graph_height),
                border_radius=3
            )
            
            # Draw score line
            scores = list(self.game.score_history)
            max_score = max(scores)
            min_score = min(scores)
            score_range = max(1, max_score - min_score)
            
            points = []
            for i, score in enumerate(scores):
                point_x = graph_x + (i / (len(scores) - 1)) * graph_width
                point_y = graph_y + graph_height - ((score - min_score) / score_range) * graph_height
                points.append((point_x, point_y))
            
            if len(points) > 1:
                pygame.draw.lines(
                    screen,
                    ACCENT_COLOR,
                    False,
                    points,
                    2
                )
            
            # Draw border
            pygame.draw.rect(
                screen,
                LIGHT_GRAY,
                (graph_x, graph_y, graph_width, graph_height),
                1,
                border_radius=3
            )
    
    def draw_settings_screen(self, screen):
        """Draw the settings screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = large_font.render("Settings", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Settings panel
        panel_width = 700
        panel_height = 600
        panel_x = (SCREEN_WIDTH - panel_width) // 2
        panel_y = 100
        
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        draw_panel(screen, panel_rect)
        
        # Settings
        y_offset = panel_y + 30
        
        # Game Settings
        settings_title = medium_font.render("Game Settings", True, WHITE)
        settings_rect = settings_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(settings_title, settings_rect)
        y_offset += 50
        
        # Difficulty dropdown
        difficulty_dropdown_rect = pygame.Rect(panel_x + 150, y_offset, 300, 30)
        if "difficulty" not in self.dropdowns:
            self.add_dropdown(
                "difficulty",
                "Difficulty",
                difficulty_dropdown_rect,
                list(DIFFICULTY_LEVELS.keys()),
                self.game.difficulty,
                self.set_difficulty
            )
        else:
            self.dropdowns["difficulty"]["rect"] = difficulty_dropdown_rect
        
        # Difficulty label
        difficulty_text = small_font.render("Difficulty:", True, TEXT_COLOR)
        screen.blit(difficulty_text, (panel_x + 30, y_offset + 15))
        
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
        
        # AI Settings
        ai_title = medium_font.render("AI Settings", True, WHITE)
        ai_rect = ai_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(ai_title, ai_rect)
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
        
        # Show AI visualization checkbox
        ai_viz_checkbox_rect = pygame.Rect(panel_x + 30, y_offset, 20, 20)
        
        if "ai_visualization" not in self.checkboxes:
            self.add_checkbox(
                "ai_visualization",
                "Show AI Visualization",
                ai_viz_checkbox_rect,
                self.show_ai_visualization,
                self.toggle_ai_visualization
            )
        else:
            self.checkboxes["ai_visualization"]["rect"] = ai_viz_checkbox_rect
            self.checkboxes["ai_visualization"]["checked"] = self.show_ai_visualization
        
        y_offset += 40
        
        # Sound Settings
        sound_title = medium_font.render("Sound Settings", True, WHITE)
        sound_rect = sound_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
        screen.blit(sound_title, sound_rect)
        y_offset += 50
        
        # Sound effects checkbox
        sound_checkbox_rect = pygame.Rect(panel_x + 30, y_offset, 20, 20)
        
        if "sound_effects" not in self.checkboxes:
            self.add_checkbox(
                "sound_effects",
                "Sound Effects",
                sound_checkbox_rect,
                self.sound_enabled,
                self.toggle_sound
            )
        else:
            self.checkboxes["sound_effects"]["rect"] = sound_checkbox_rect
            self.checkboxes["sound_effects"]["checked"] = self.sound_enabled
        
        y_offset += 40
        
        # Music checkbox
        music_checkbox_rect = pygame.Rect(panel_x + 30, y_offset, 20, 20)
        
        if "music" not in self.checkboxes:
            self.add_checkbox(
                "music",
                "Background Music",
                music_checkbox_rect,
                self.music_enabled,
                self.toggle_music
            )
        else:
            self.checkboxes["music"]["rect"] = music_checkbox_rect
            self.checkboxes["music"]["checked"] = self.music_enabled
        
        y_offset += 60
        
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
            f"Player Wins: {self.game.stats['player_wins']}",
            f"Total Game Time: {format_time(int(self.game.stats['game_time'] * 1000))}"
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
    
    def draw_help_screen(self, screen):
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
            "Constraint Mode: Play with random constraints that change periodically",
            "Multiplayer: Play against a friend or the AI"
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
            "H: Toggle heatmap visualization",
            "V: Toggle AI visualization",
            "S: Toggle statistics panel",
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
    
    def draw_ai_training_screen(self, screen):
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
        
        genetic_text = "Genetic Algorithm"
        if self.genetic_optimizer and self.genetic_optimizer.running:
            genetic_text = "Stop Genetic Training"
            
        self.add_button(
            "genetic",
            genetic_text,
            genetic_rect,
            action=self.toggle_genetic_training
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
        
        if nn_enabled and self.nn_trainer and self.nn_trainer.running:
            nn_text = "Stop NN Training"
            
        self.add_button(
            "neural",
            nn_text,
            nn_rect,
            action=self.toggle_nn_training if nn_enabled else None
        )
        
        y_offset += 80
        
        # Training progress
        if (self.genetic_optimizer and self.genetic_optimizer.running) or (self.nn_trainer and self.nn_trainer.running):
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
            
            # Calculate progress
            if self.genetic_optimizer and self.genetic_optimizer.running:
                progress = self.genetic_optimizer.current_generation / self.genetic_optimizer.generations
                status_text = f"Generation: {self.genetic_optimizer.current_generation}/{self.genetic_optimizer.generations}"
                if self.genetic_optimizer.best_individual:
                    status_text += f" - Best Fitness: {self.genetic_optimizer.best_fitness:.2f}"
            elif self.nn_trainer and self.nn_trainer.running:
                if self.nn_trainer.total_epochs > 0:
                    progress = self.nn_trainer.current_epoch / self.nn_trainer.total_epochs
                    status_text = f"Epoch: {self.nn_trainer.current_epoch}/{self.nn_trainer.total_epochs}"
                    if self.nn_trainer.loss_history:
                        status_text += f" - Loss: {self.nn_trainer.loss_history[-1]:.4f}"
                else:
                    progress = 0
                    status_text = "Collecting training data..."
            else:
                progress = 0
                status_text = ""
            
            draw_progress_bar(screen, progress_bar_rect, progress, ACCENT_COLOR)
            
            # Progress text
            progress_text = medium_font.render(f"{int(progress * 100)}%", True, WHITE)
            progress_text_rect = progress_text.get_rect(center=(panel_x + panel_width // 2, y_offset + 15))
            screen.blit(progress_text, progress_text_rect)
            
            # Status text
            status_text_surf = small_font.render(status_text, True, WHITE)
            status_text_rect = status_text_surf.get_rect(center=(panel_x + panel_width // 2, y_offset + 50))
            screen.blit(status_text_surf, status_text_rect)
            
            y_offset += 70
            
            # Training visualization
            if self.genetic_optimizer and self.genetic_optimizer.best_fitness_history:
                # Get plot surface
                plot_surface = self.genetic_optimizer.get_training_plot(panel_width - 100, 200)
                if plot_surface:
                    screen.blit(plot_surface, (panel_x + 50, y_offset))
                    y_offset += 220
            
            elif self.nn_trainer and self.nn_trainer.loss_history:
                # Get plot surface
                plot_surface = self.nn_trainer.get_training_plot(panel_width - 100, 200)
                if plot_surface:
                    screen.blit(plot_surface, (panel_x + 50, y_offset))
                    y_offset += 220
        
        # AI weights display
        if not ((self.genetic_optimizer and self.genetic_optimizer.running) or (self.nn_trainer and self.nn_trainer.running)):
            weights_title = medium_font.render("Current AI Weights", True, WHITE)
            weights_rect = weights_title.get_rect(center=(panel_x + panel_width // 2, y_offset))
            screen.blit(weights_title, weights_rect)
            y_offset += 40
            
            # Display current weights
            for i, (key, value) in enumerate(self.game.ai_weights.items()):
                weight_text = small_font.render(f"{key}: {value:.4f}", True, TEXT_COLOR)
                screen.blit(weight_text, (panel_x + 30, y_offset))
                y_offset += 25
            
            # Test AI button
            test_rect = pygame.Rect(
                panel_x + panel_width // 2 - 100,
                y_offset + 20,
                200,
                50
            )
            
            self.add_button(
                "test_ai",
                "Test AI",
                test_rect,
                action=self.test_ai
            )
        
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
    
    def draw_multiplayer_screen(self, screen):
        """Draw the multiplayer screen"""
        screen.fill(BG_COLOR)
        
        # Title
        title_text = large_font.render("Multiplayer", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Mode subtitle
        mode_text = medium_font.render(f"Mode: {self.multiplayer_mode}", True, ACCENT_COLOR)
        mode_rect = mode_text.get_rect(center=(SCREEN_WIDTH // 2, 90))
        screen.blit(mode_text, mode_rect)
        
        # Calculate game board positions
        board_width = self.player1_game.width * GRID_SIZE
        board_height = self.player1_game.height * GRID_SIZE
        
        # Player 1 board
        player1_x = SCREEN_WIDTH // 4 - board_width // 2
        player1_y = (SCREEN_HEIGHT - board_height) // 2
        
        # Player 2 board
        player2_x = 3 * SCREEN_WIDTH // 4 - board_width // 2
        player2_y = (SCREEN_HEIGHT - board_height) // 2
        
        # Draw player labels
        player1_label = medium_font.render("Player 1", True, WHITE)
        player1_label_rect = player1_label.get_rect(center=(player1_x + board_width // 2, player1_y - 40))
        screen.blit(player1_label, player1_label_rect)
        
        player2_label = medium_font.render("Player 2" if self.multiplayer_mode == "VS_PLAYER" else "AI", True, WHITE)
        player2_label_rect = player2_label.get_rect(center=(player2_x + board_width // 2, player2_y - 40))
        screen.blit(player2_label, player2_label_rect)
        
        # Draw scores
        score1_text = small_font.render(f"Score: {self.player1_game.score}", True, WHITE)
        score1_rect = score1_text.get_rect(left=player1_x, top=player1_y - 20)
        screen.blit(score1_text, score1_rect)
        
        score2_text = small_font.render(f"Score: {self.player2_game.score}", True, WHITE)
        score2_rect = score2_text.get_rect(left=player2_x, top=player2_y - 20)
        screen.blit(score2_text, score2_rect)
        
        # Draw game boards
        self.player1_game.draw(screen, player1_x, player1_y)
        self.player2_game.draw(screen, player2_x, player2_y)
        
        # Draw next pieces
        next_y = player1_y + board_height + 20
        next1_text = small_font.render("Next:", True, WHITE)
        screen.blit(next1_text, (player1_x, next_y))
        
        next2_text = small_font.render("Next:", True, WHITE)
        screen.blit(next2_text, (player2_x, next_y))
        
        # Draw next piece previews
        for i, next_piece in enumerate(self.player1_game.next_pieces[:1]):  # Show only first next piece
            self.player1_game.draw_piece_preview(
                screen,
                next_piece,
                player1_x + 60,
                next_y + 30
            )
            
        for i, next_piece in enumerate(self.player2_game.next_pieces[:1]):
            self.player2_game.draw_piece_preview(
                screen,
                next_piece,
                player2_x + 60,
                next_y + 30
            )
        
        # Draw held pieces
        held_y = next_y + 60
        held1_text = small_font.render("Hold:", True, WHITE)
        screen.blit(held1_text, (player1_x, held_y))
        
        held2_text = small_font.render("Hold:", True, WHITE)
        screen.blit(held2_text, (player2_x, held_y))
        
        # Draw held piece previews
        self.player1_game.draw_piece_preview(
            screen,
            self.player1_game.held_piece,
            player1_x + 60,
            held_y + 30
        )
        
        self.player2_game.draw_piece_preview(
            screen,
            self.player2_game.held_piece,
            player2_x + 60,
            held_y + 30
        )
        
        # Draw controls
        controls_y = held_y + 60
        controls1_text = small_font.render("Controls: Arrow Keys, Space, C", True, LIGHT_GRAY)
        screen.blit(controls1_text, (player1_x, controls_y))
        
        if self.multiplayer_mode == "VS_PLAYER":
            controls2_text = small_font.render("Controls: WASD, Q (drop), E (hold)", True, LIGHT_GRAY)
            screen.blit(controls2_text, (player2_x, controls_y))
        else:
            controls2_text = small_font.render("AI is playing...", True, LIGHT_GRAY)
            screen.blit(controls2_text, (player2_x, controls_y))
        
        # Game over overlay
        if self.player1_game.game_over and self.player2_game.game_over:
            # Determine winner
            if self.player1_game.score > self.player2_game.score:
                winner_text = large_font.render("Player 1 Wins!", True, GREEN)
                self.game.stats["player_wins"] += 1
            elif self.player2_game.score > self.player1_game.score:
                winner_text = large_font.render(
                    "Player 2 Wins!" if self.multiplayer_mode == "VS_PLAYER" else "AI Wins!",
                    True,
                    RED
                )
                self.game.stats["ai_wins"] += 1
            else:
                winner_text = large_font.render("It's a Tie!", True, YELLOW)
            
            winner_rect = winner_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
            screen.blit(winner_text, winner_rect)
            
            # Restart button
            restart_rect = pygame.Rect(
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2 + 20,
                200,
                50
            )
            
            self.add_button("restart", "Play Again", restart_rect, action=self.reset_multiplayer)
            
            # Menu button
            menu_rect = pygame.Rect(
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2 + 80,
                200,
                50
            )
            
            self.add_button("menu", "Main Menu", menu_rect, action=lambda: self.set_screen("MENU"))
        
        # Pause overlay
        elif self.player1_game.paused:
            # Semi-transparent overlay
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            
            # Paused text
            paused_text = large_font.render("PAUSED", True, WHITE)
            paused_rect = paused_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))
            screen.blit(paused_text, paused_rect)
            
            # Resume button
            resume_rect = pygame.Rect(
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2 + 20,
                200,
                50
            )
            
            self.add_button("resume", "Resume", resume_rect, action=lambda: self.toggle_multiplayer_pause())
            
            # Menu button
            menu_rect = pygame.Rect(
                SCREEN_WIDTH // 2 - 100,
                SCREEN_HEIGHT // 2 + 80,
                200,
                50
            )
            
            self.add_button("menu", "Main Menu", menu_rect, action=lambda: self.set_screen("MENU"))
        
        # Draw UI elements
        self.draw_ui_elements(screen)
        
        # Draw FPS counter
        current_time = time.time()
        self.fps_history.append(1.0 / max(0.001, current_time - self.last_time))
        self.last_time = current_time
        
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        fps_text = tiny_font.render(f"FPS: {avg_fps:.1f}", True, LIGHT_GRAY)
        screen.blit(fps_text, (10, 10))
    
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
        
        # Update multiplayer
        elif self.current_screen == "MULTIPLAYER":
            self.player1_game.update(current_time)
            self.player2_game.update(current_time)
            
            # Handle AI moves for player 2 if in AI mode
            if self.multiplayer_mode != "VS_PLAYER" and self.player2_game.ai_active and not self.player2_game.game_over and not self.player2_game.paused:
                if current_time - self.player2_game.last_ai_move_time > self.player2_game.ai_delay:
                    self.player2_game.apply_ai_move()
                    self.player2_game.last_ai_move_time = current_time
    
    def set_screen(self, screen_name):
        """Set the current screen"""
        self.current_screen = screen_name
        self.buttons = {}  # Clear buttons when changing screens
        self.active_dropdown = None
    
    def start_game(self, mode):
        """Start a new game with the specified mode"""
        self.game.reset()
        self.game.game_mode = mode
        
        # Set up game mode specific settings
        if mode == "AI_PLAY":
            self.game.ai_active = True
        elif mode == "AI_BATTLE":
            self.set_screen("MULTIPLAYER")
            self.multiplayer_mode = "VS_AI"
            self.reset_multiplayer()
            return
        elif mode == "CONSTRAINT":
            self.game.activate_constraint_mode()
        elif mode == "MULTIPLAYER":
            self.set_screen("MULTIPLAYER")
            self.multiplayer_mode = "VS_PLAYER"
            self.reset_multiplayer()
            return
        elif mode == "AI_TRAIN":
            self.set_screen("AI_TRAINING")
            return
        
        self.set_screen("GAME")
    
    def toggle_pause(self):
        """Toggle game pause state"""
        self.game.paused = not self.game.paused
    
    def toggle_multiplayer_pause(self):
        """Toggle multiplayer pause state"""
        self.player1_game.paused = not self.player1_game.paused
        self.player2_game.paused = self.player1_game.paused
    
    def reset_multiplayer(self):
        """Reset multiplayer games"""
        self.player1_game.reset()
        self.player2_game.reset()
        
        # Set up player 2 based on mode
        if self.multiplayer_mode != "VS_PLAYER":
            self.player2_game.ai_active = True
        else:
            self.player2_game.ai_active = False
    
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
    
    def toggle_ai_visualization(self, enabled):
        """Toggle AI visualization"""
        self.show_ai_visualization = enabled
    
    def toggle_sound(self, enabled):
        """Toggle sound effects"""
        self.sound_enabled = enabled
        
        # Set volume to 0 if disabled
        for sound in sound_effects.values():
            sound.set_volume(0.5 if enabled else 0)
    
    def toggle_music(self, enabled):
        """Toggle background music"""
        self.music_enabled = enabled
        
        if enabled:
            play_music()
        else:
            stop_music()
    
    def set_difficulty(self, difficulty):
        """Set game difficulty"""
        self.game.difficulty = difficulty
        
        # Update game settings based on difficulty
        self.game.fall_speed = DIFFICULTY_LEVELS[difficulty]["fall_speed"]
        self.game.ai_delay = DIFFICULTY_LEVELS[difficulty]["ai_delay"]
    
    def toggle_genetic_training(self):
        """Toggle genetic algorithm training"""
        if self.genetic_optimizer and self.genetic_optimizer.running:
            self.genetic_optimizer.stop_training()
        else:
            self.genetic_optimizer = start_genetic_training(
                self.game,
                self.update_training_progress
            )
    
    def toggle_nn_training(self):
        """Toggle neural network training"""
        if not TORCH_AVAILABLE:
            return
            
        if self.nn_trainer and self.nn_trainer.running:
            self.nn_trainer.stop_training()
        else:
            self.nn_trainer = start_nn_training(
                self.game,
                self.update_training_progress
            )
    
    def update_training_progress(self, progress, *args):
        """Update training progress for UI"""
        self.ai_training_progress = progress
    
    def test_ai(self):
        """Test the current AI by starting a game with AI active"""
        self.game.reset()
        self.game.ai_active = True
        self.set_screen("GAME")
    
    def draw(self, screen):
        """Draw the current screen"""
        if self.current_screen == "MENU":
            self.draw_menu_screen(screen)
        elif self.current_screen == "GAME":
            self.draw_game_screen(screen)
        elif self.current_screen == "SETTINGS":
            self.draw_settings_screen(screen)
        elif self.current_screen == "HELP":
            self.draw_help_screen(screen)
        elif self.current_screen == "AI_TRAINING":
            self.draw_ai_training_screen(screen)
        elif self.current_screen == "MULTIPLAYER":
            self.draw_multiplayer_screen(screen)