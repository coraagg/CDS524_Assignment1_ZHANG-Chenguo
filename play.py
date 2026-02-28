# Main game launcher with introduction screen
import pygame
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())
from space_env import SpaceDodgeEnv


def draw_intro_screen(screen, screen_size):
    """Draw introduction screen with game instructions"""
    # Background - dark space
    screen.fill((5, 5, 25))

    # Draw stars in background
    for _ in range(100):
        x = np.random.randint(0, screen_size[0])
        y = np.random.randint(0, screen_size[1])
        size = np.random.randint(1, 3)
        brightness = np.random.uniform(0.5, 1.0)
        color = (int(200 * brightness), int(200 * brightness), int(255 * brightness))
        pygame.draw.circle(screen, color, (x, y), size)

    # Create fonts with fallback
    try:
        title_font = pygame.font.SysFont('arial', 64, bold=True)
        subtitle_font = pygame.font.SysFont('arial', 32)
        heading_font = pygame.font.SysFont('arial', 28, bold=True)
        normal_font = pygame.font.SysFont('arial', 22)
        small_font = pygame.font.SysFont('arial', 18)
    except:
        title_font = pygame.font.Font(None, 64)
        subtitle_font = pygame.font.Font(None, 32)
        heading_font = pygame.font.Font(None, 28)
        normal_font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 18)

    # Title section
    title = title_font.render("SPACE DODGE", True, (100, 200, 255))
    subtitle = subtitle_font.render("Asteroid Avoidance Game", True, (150, 220, 255))

    screen.blit(title, (screen_size[0] // 2 - title.get_width() // 2, 40))
    screen.blit(subtitle, (screen_size[0] // 2 - subtitle.get_width() // 2, 110))

    # Game description
    desc_y = 180
    description = [
        "You are piloting a spaceship through a dangerous asteroid field.",
        "Your mission is to survive as long as possible while avoiding asteroids.",
        "Collect powerups for special abilities and achieve the highest score!"
    ]

    for i, line in enumerate(description):
        text = normal_font.render(line, True, (220, 230, 255))
        screen.blit(text, (screen_size[0] // 2 - text.get_width() // 2, desc_y + i * 32))

    # Controls section
    controls_y = desc_y + len(description) * 32 + 30
    controls_title = heading_font.render("CONTROLS", True, (255, 200, 100))
    screen.blit(controls_title, (screen_size[0] // 2 - controls_title.get_width() // 2, controls_y))

    controls = [
        "UP ARROW : Move Up",
        "DOWN ARROW : Move Down",
        "LEFT ARROW : Move Left",
        "RIGHT ARROW : Move Right",
        "R : Restart Game",
        "ESC : Exit Game"
    ]

    for i, control in enumerate(controls):
        text = normal_font.render(control, True, (240, 240, 200))
        screen.blit(text, (screen_size[0] // 2 - text.get_width() // 2, controls_y + 40 + i * 30))

    # Start button (moved up since POWERUPS section is removed)
    start_y = controls_y + len(controls) * 30 + 40  # Adjusted position
    start_bg = pygame.Rect(screen_size[0] // 2 - 150, start_y, 300, 50)
    pygame.draw.rect(screen, (50, 150, 50), start_bg, border_radius=10)
    pygame.draw.rect(screen, (100, 255, 100), start_bg, 3, border_radius=10)

    start_text = heading_font.render("PRESS SPACE TO START", True, (255, 255, 255))
    screen.blit(start_text, (screen_size[0] // 2 - start_text.get_width() // 2, start_y + 12))

def show_intro_screen():
    """Display introduction screen and wait for user input"""
    pygame.init()
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Space Dodge")
    clock = pygame.time.Clock()
 
    intro_running = True
    while intro_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_y, pygame.K_SPACE):
                    pygame.quit()
                    return True
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False

        # Draw introduction screen
        draw_intro_screen(screen, screen_size)

        # Update display
        pygame.display.flip()

        # Control frame rate
        clock.tick(60)

    pygame.quit()
    return False


def main_game():
    """Main game loop"""
    print("=" * 70)
    print("SPACE DODGE - Starting Game")
    print("=" * 70)

    # Create game environment
    env = SpaceDodgeEnv(render_mode="human")
    state = env.reset()

    # Game state variables
    running = True
    game_over = False
    clock = pygame.time.Clock()

    # Score tracking
    high_score = 0
    last_score = 0
    last_time = 0
    last_level = 1
    last_near_misses = 0

    # Font initialization
    try:
        font_large = pygame.font.SysFont('arial', 72)
        font_medium = pygame.font.SysFont('arial', 36)
        font_small = pygame.font.SysFont('arial', 24)
    except:
        font_large = pygame.font.Font(None, 72)
        font_medium = pygame.font.Font(None, 36)
        font_small = pygame.font.Font(None, 24)

    # Error handling
    error_count = 0
    max_errors = 3

    # Main game loop
    while running and error_count < max_errors:
        try:
            # Check if game window exists
            if not hasattr(env, 'screen') or env.screen is None:
                print("Game window closed, restarting...")
                state = env.reset()
                env._init_pygame()
                continue

            # Default action: No operation
            action = 4

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        if game_over:
                            # Restart from game over state
                            state = env.reset()
                            game_over = False
                            print("Game restarted!")
                        else:
                            # Restart during gameplay
                            state = env.reset()
                            print("Game reset!")

            # Game logic when not game over
            if not game_over:
                # Continuous key detection for movement
                keys = pygame.key.get_pressed()

                # Action mapping: 0=Up, 1=Down, 2=Left, 3=Right, 4=No-op
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    action = 0
                elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    action = 1
                elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    action = 2
                elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    action = 3

                # Execute action and get results
                next_state, reward, done, info = env.step(action)
                state = next_state

                # Update high score
                if info['score'] > high_score:
                    high_score = info['score']

                # Check if game is over
                if done:
                    game_over = True
                    last_score = info['score']
                    last_time = info['time_alive']
                    last_level = info['level']
                    last_near_misses = info['near_misses']

                    # Print game over info to console
                    print("\n" + "=" * 70)
                    print("GAME OVER!")
                    print(f"Final Score: {last_score:.1f}")
                    print(f"Survival Time: {last_time} frames ({last_time // 60}.{last_time % 60:02d} seconds)")
                    print(f"Level Reached: {last_level}")
                    print(f"Near Misses: {last_near_misses}")
                    print("Press R to restart or ESC to quit")
                    print("=" * 70)

            # Render game
            env.render()

            # Draw game over screen if game is over
            if game_over:
                # Create semi-transparent overlay
                overlay = pygame.Surface(env.screen_size, pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                env.screen.blit(overlay, (0, 0))

                # Game Over text
                game_over_text = font_large.render("GAME OVER", True, (255, 50, 50))
                env.screen.blit(game_over_text,
                                (env.screen_size[0] // 2 - game_over_text.get_width() // 2, 100))

                # Score information
                score_text = font_medium.render(f"Score: {last_score:.1f}", True, (255, 255, 255))
                time_text = font_medium.render(f"Time: {last_time // 60}:{last_time % 60:02d}", True, (255, 255, 255))
                level_text = font_medium.render(f"Level: {last_level}", True, (255, 255, 255))
                misses_text = font_medium.render(f"Near Misses: {last_near_misses}", True, (255, 255, 255))
                high_score_text = font_medium.render(f"High Score: {high_score:.1f}", True, (255, 255, 100))

                # Position calculation
                y_start = 200
                line_height = 50

                env.screen.blit(score_text, (env.screen_size[0] // 2 - score_text.get_width() // 2, y_start))
                env.screen.blit(time_text,
                                (env.screen_size[0] // 2 - time_text.get_width() // 2, y_start + line_height))
                env.screen.blit(level_text,
                                (env.screen_size[0] // 2 - level_text.get_width() // 2, y_start + line_height * 2))
                env.screen.blit(misses_text,
                                (env.screen_size[0] // 2 - misses_text.get_width() // 2, y_start + line_height * 3))
                env.screen.blit(high_score_text,
                                (env.screen_size[0] // 2 - high_score_text.get_width() // 2, y_start + line_height * 4))

                # Restart instructions
                restart_text = font_small.render("Press R to Restart, ESC to Quit", True, (200, 200, 255))
                env.screen.blit(restart_text,
                                (env.screen_size[0] // 2 - restart_text.get_width() // 2,
                                 env.screen_size[1] - 100))

                # Performance rating
                performance_score = last_score * 0.5 + last_time * 0.3 + last_level * 10

                if performance_score < 1000:
                    rating = "BEGINNER PILOT"
                    tip = "Practice predicting asteroid paths"
                elif performance_score < 2500:
                    rating = "SKILLED PILOT"
                    tip = "Good! Try collecting more powerups"
                elif performance_score < 5000:
                    rating = "ELITE PILOT"
                    tip = "Excellent! You've mastered evasion"
                else:
                    rating = "LEGENDARY PILOT"
                    tip = "Outstanding! You're a space master!"

                rating_text = font_medium.render(f"Rating: {rating}", True, (100, 255, 100))

                env.screen.blit(rating_text,
                                (env.screen_size[0] // 2 - rating_text.get_width() // 2, y_start + line_height * 5))

                # Update display
                pygame.display.flip()

            # Update window title with game info
            title = "Space Dodge | "
            if game_over:
                title += "GAME OVER | "

            title += f"Score: {env.score:.1f} | "
            title += f"High Score: {high_score:.1f} | "
            title += f"Time: {env.time_alive // 60}:{env.time_alive % 60:02d} | "
            title += f"Level: {env.level} | "
            title += f"Asteroids: {len(env.asteroids)}"

            pygame.display.set_caption(title)

            # Control frame rate
            clock.tick(60)

        except Exception as e:
            error_count += 1
            print(f"Game error ({error_count}/{max_errors}): {e}")
            import traceback
            traceback.print_exc()

            # Try to recover the game
            if error_count < max_errors:
                try:
                    env.close()
                    env = SpaceDodgeEnv(render_mode="human")
                    state = env.reset()
                    game_over = False
                    print("Game recovered, restarting")
                except Exception as e2:
                    print(f"Unable to recover game: {e2}")

    # Cleanup
    try:
        env.close()
    except:
        pass

    print(f"\n{'=' * 70}")
    print("Thank you for playing SPACE DODGE!")
    print(f"Final High Score: {high_score:.1f}")
    print("See you next time!")
    print(f"{'=' * 70}")


def main():
    """Main function: Start with intro screen, then launch game"""
    print("=" * 70)
    print("SPACE DODGE - Asteroid Avoidance Game")
    print("=" * 70)

    # Show introduction screen
    if not show_intro_screen():
        print("Game cancelled by user")
        return

    # Start main game
    main_game()


if __name__ == "__main__":
    main()