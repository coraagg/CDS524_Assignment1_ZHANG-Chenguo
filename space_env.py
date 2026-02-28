# Game environment
import pygame
import numpy as np
import math
import random
import sys


class SpaceDodgeEnv:
    """Space Dodge Game Environment - Fixed Version"""

    def __init__(self, render_mode="human", screen_size=(800, 600)):
        self.screen_size = screen_size
        self.render_mode = render_mode

        # Ship parameters
        self.ship_size = 20
        self.ship_pos = np.array([screen_size[0] // 2, screen_size[1] // 2], dtype=float)
        self.ship_vel = np.array([0.0, 0.0], dtype=float)
        self.ship_max_speed = 6.0
        self.ship_acceleration = 0.5
        self.ship_friction = 0.95

        # Asteroid parameters
        self.asteroids = []
        self.max_asteroids = 15
        self.spawn_rate = 0.02
        self.asteroid_min_speed = 1.5
        self.asteroid_max_speed = 3.5

        # Powerup parameters
        self.powerups = []
        self.powerup_types = ['shield', 'laser', 'slow']
        self.powerup_spawn_rate = 0.002
        self.ship_has_shield = False
        self.shield_time = 0
        self.max_shield_time = 300

        # Particle system
        self.particles = []

        # Game state
        self.score = 0
        self.time_alive = 0
        self.done = False
        self.level = 1

        # Near miss counter
        self.near_misses = 0

        # Colors
        self.colors = {
            'ship': (100, 200, 255),
            'ship_shield': (100, 255, 200),
            'asteroid_small': (180, 180, 200),
            'asteroid_medium': (160, 160, 190),
            'asteroid_large': (140, 140, 180),
            'powerup_shield': (0, 255, 255),
            'powerup_laser': (255, 100, 100),
            'powerup_slow': (100, 255, 100),
            'background': (10, 10, 30),
            'star': (200, 200, 255),
            'text': (220, 220, 255)
        }

        # Starfield background
        self.stars = []
        self._generate_stars(100)

        # Initialize Pygame
        if self.render_mode == "human":
            self._init_pygame()

    def _init_pygame(self):
        """Initialize Pygame safely"""
        try:
            if not pygame.get_init():
                pygame.init()

            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Space Dodge")
            self.clock = pygame.time.Clock()

            # Fonts
            try:
                self.font = pygame.font.SysFont('arial', 20)
                self.title_font = pygame.font.SysFont('arial', 32, bold=True)
                self.small_font = pygame.font.SysFont('arial', 16)
            except:
                self.font = pygame.font.Font(None, 20)
                self.title_font = pygame.font.Font(None, 32)
                self.small_font = pygame.font.Font(None, 16)

            print("✓ Pygame initialized successfully")

        except Exception as e:
            print(f"⚠️ Pygame initialization failed: {e}")
            self.render_mode = None

    def _generate_stars(self, count):
        """Generate starfield background"""
        for _ in range(count):
            self.stars.append({
                'pos': np.array([random.randint(0, self.screen_size[0]),
                                 random.randint(0, self.screen_size[1])], dtype=float),
                'size': random.randint(1, 3),
                'brightness': random.uniform(0.3, 1.0)
            })

    def reset(self, seed=None):
        """Reset game environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset ship
        self.ship_pos = np.array([self.screen_size[0] // 2, self.screen_size[1] // 2], dtype=float)
        self.ship_vel = np.array([0.0, 0.0], dtype=float)

        # Clear asteroids and powerups
        self.asteroids = []
        self.powerups = []
        self.particles = []

        # Reset state
        self.score = 0
        self.time_alive = 0
        self.done = False
        self.level = 1
        self.near_misses = 0
        self.ship_has_shield = False
        self.shield_time = 0

        # Generate initial asteroids
        for _ in range(5):
            self._spawn_asteroid()

        return self._get_state()

    def _spawn_asteroid(self):
        """Spawn a new asteroid"""
        side = random.randint(0, 3)

        if side == 0:  # Top
            x = random.randint(0, self.screen_size[0])
            y = -50
        elif side == 1:  # Right
            x = self.screen_size[0] + 50
            y = random.randint(0, self.screen_size[1])
        elif side == 2:  # Bottom
            x = random.randint(0, self.screen_size[0])
            y = self.screen_size[1] + 50
        else:  # Left
            x = -50
            y = random.randint(0, self.screen_size[1])

        size = random.choice([15, 25, 35])
        speed = random.uniform(self.asteroid_min_speed, self.asteroid_max_speed) * (1 + self.level * 0.1)

        # Calculate direction toward ship (with some randomness)
        dx = self.ship_pos[0] - x
        dy = self.ship_pos[1] - y

        angle = math.atan2(dy, dx) + random.uniform(-0.5, 0.5)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        rotation_speed = random.uniform(-0.05, 0.05)

        self.asteroids.append({
            'pos': np.array([x, y], dtype=float),
            'vel': np.array([vx, vy], dtype=float),
            'size': size,
            'rotation': 0.0,
            'rotation_speed': rotation_speed
        })

    def _spawn_powerup(self):
        """Spawn a powerup"""
        x = random.randint(50, self.screen_size[0] - 50)
        y = random.randint(50, self.screen_size[1] - 50)

        powerup_type = random.choice(self.powerup_types)

        self.powerups.append({
            'pos': np.array([x, y], dtype=float),
            'type': powerup_type,
            'size': 15,
            'pulse': 0.0
        })

    def _create_particles(self, pos, count=20, color=(255, 255, 200)):
        """Create particle effects"""
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            speed = random.uniform(1.0, 5.0)
            lifetime = random.randint(20, 60)

            self.particles.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed], dtype=float),
                'color': color,
                'size': random.randint(2, 5),
                'lifetime': lifetime,
                'max_lifetime': lifetime
            })

    def step(self, action):
        """Execute one game step"""
        # 1. Apply action
        self._apply_action(action)

        # 2. Update physics
        self._update_physics()

        # 3. Spawn new objects
        self._spawn_objects()

        # 4. Update particles
        self._update_particles()

        # 5. Check collisions
        self._check_collisions()

        # 6. Calculate reward
        reward = self._calculate_reward()

        # 7. Update game state
        self.score += reward
        self.time_alive += 1

        # 8. Increase difficulty
        if self.time_alive % 600 == 0:
            self.level += 1

        # 9. Return results
        next_state = self._get_state()
        info = {
            'score': self.score,
            'time_alive': self.time_alive,
            'level': self.level,
            'asteroids': len(self.asteroids),
            'near_misses': self.near_misses,
            'has_shield': self.ship_has_shield
        }

        return next_state, reward, self.done, info

    def _apply_action(self, action):
        """Apply action to ship"""
        if action == 0:  # Up
            self.ship_vel[1] -= self.ship_acceleration
        elif action == 1:  # Down
            self.ship_vel[1] += self.ship_acceleration
        elif action == 2:  # Left
            self.ship_vel[0] -= self.ship_acceleration
        elif action == 3:  # Right
            self.ship_vel[0] += self.ship_acceleration

        # Limit speed
        speed = np.linalg.norm(self.ship_vel)
        if speed > self.ship_max_speed:
            self.ship_vel = self.ship_vel / speed * self.ship_max_speed

        # Apply friction
        self.ship_vel *= self.ship_friction

        # Update position
        self.ship_pos += self.ship_vel

        # Boundary check (with bounce)
        if self.ship_pos[0] < self.ship_size:
            self.ship_pos[0] = self.ship_size
            self.ship_vel[0] = abs(self.ship_vel[0]) * 0.5
        elif self.ship_pos[0] > self.screen_size[0] - self.ship_size:
            self.ship_pos[0] = self.screen_size[0] - self.ship_size
            self.ship_vel[0] = -abs(self.ship_vel[0]) * 0.5

        if self.ship_pos[1] < self.ship_size:
            self.ship_pos[1] = self.ship_size
            self.ship_vel[1] = abs(self.ship_vel[1]) * 0.5
        elif self.ship_pos[1] > self.screen_size[1] - self.ship_size:
            self.ship_pos[1] = self.screen_size[1] - self.ship_size
            self.ship_vel[1] = -abs(self.ship_vel[1]) * 0.5

        # Update shield
        if self.ship_has_shield:
            self.shield_time -= 1
            if self.shield_time <= 0:
                self.ship_has_shield = False
                self._create_particles(self.ship_pos, 10, self.colors['powerup_shield'])

    def _update_physics(self):
        """Update physics system"""
        # Update asteroids
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            # Update position
            asteroid['pos'] += asteroid['vel']
            asteroid['rotation'] += asteroid['rotation_speed']

            # Check if out of bounds
            if (asteroid['pos'][0] < -100 or asteroid['pos'][0] > self.screen_size[0] + 100 or
                    asteroid['pos'][1] < -100 or asteroid['pos'][1] > self.screen_size[1] + 100):
                asteroids_to_remove.append(i)

        # Remove out-of-bounds asteroids
        for i in sorted(asteroids_to_remove, reverse=True):
            if i < len(self.asteroids):
                self.asteroids.pop(i)

        # Update powerups
        for powerup in self.powerups:
            powerup['pulse'] = (powerup['pulse'] + 0.05) % (math.pi * 2)

    def _spawn_objects(self):
        """Spawn new objects"""
        # Spawn asteroids
        if len(self.asteroids) < self.max_asteroids and random.random() < self.spawn_rate:
            self._spawn_asteroid()

        # Spawn powerups
        if len(self.powerups) < 2 and random.random() < self.powerup_spawn_rate:
            self._spawn_powerup()

    def _update_particles(self):
        """Update particle system"""
        particles_to_remove = []
        for i, particle in enumerate(self.particles):
            particle['pos'] += particle['vel']
            particle['lifetime'] -= 1

            if particle['lifetime'] <= 0:
                particles_to_remove.append(i)

        # Remove dead particles
        for i in sorted(particles_to_remove, reverse=True):
            if i < len(self.particles):
                self.particles.pop(i)

    def _check_collisions(self):
        """Check for collisions - FIXED VERSION"""
        asteroids_to_remove = []
        powerups_to_remove = []

        # Ship-asteroid collision
        for i, asteroid in enumerate(self.asteroids):
            dx = self.ship_pos[0] - asteroid['pos'][0]
            dy = self.ship_pos[1] - asteroid['pos'][1]
            distance = math.sqrt(dx * dx + dy * dy)

            collision_distance = self.ship_size + asteroid['size']

            if distance < collision_distance:
                if self.ship_has_shield:
                    # Shield protects, destroy asteroid
                    asteroids_to_remove.append(i)
                    self._create_particles(asteroid['pos'], 30, self.colors['asteroid_large'])
                    self.score += 5
                else:
                    # Game over
                    self.done = True
                    self._create_particles(self.ship_pos, 50, (255, 100, 100))
                    break  # No need to check more collisions if game over

        # Ship-powerup collision
        for i, powerup in enumerate(self.powerups):
            dx = self.ship_pos[0] - powerup['pos'][0]
            dy = self.ship_pos[1] - powerup['pos'][1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < self.ship_size + powerup['size']:
                powerups_to_remove.append(i)
                self._collect_powerup(powerup)

        # Remove asteroids and powerups
        for i in sorted(asteroids_to_remove, reverse=True):
            if i < len(self.asteroids):
                self.asteroids.pop(i)

        for i in sorted(powerups_to_remove, reverse=True):
            if i < len(self.powerups):
                self.powerups.pop(i)

    def _collect_powerup(self, powerup):
        """Collect a powerup - FIXED VERSION"""
        self._create_particles(powerup['pos'], 20, self.colors[f'powerup_{powerup["type"]}'])

        if powerup['type'] == 'shield':
            self.ship_has_shield = True
            self.shield_time = self.max_shield_time
        elif powerup['type'] == 'laser':
            # Clear all asteroids (create a copy of the list to avoid modification during iteration)
            asteroid_positions = []
            for asteroid in self.asteroids:
                asteroid_positions.append(asteroid['pos'].copy())

            # Clear the asteroids list
            self.asteroids.clear()

            # Create explosion particles for each destroyed asteroid
            for pos in asteroid_positions:
                self._create_particles(pos, 15, (255, 200, 100))
        elif powerup['type'] == 'slow':
            # Slow down all asteroids
            for asteroid in self.asteroids:
                asteroid['vel'] *= 0.5

    def _calculate_reward(self):
        """Calculate reward for current frame"""
        reward = 0.0

        # 1. Basic survival reward
        reward += 0.1

        # 2. Distance to nearest asteroid penalty (FIXED)
        if self.asteroids:
            nearest_distance = float('inf')
            for asteroid in self.asteroids:
                dx = self.ship_pos[0] - asteroid['pos'][0]
                dy = self.ship_pos[1] - asteroid['pos'][1]
                distance = math.sqrt(dx * dx + dy * dy)
                nearest_distance = min(nearest_distance, distance)

            safe_distance = self.ship_size + 50

            if nearest_distance < safe_distance:
                penalty = (safe_distance - nearest_distance) / safe_distance * 0.3
                reward -= penalty

                # Count near misses
                if nearest_distance < safe_distance * 0.5:
                    self.near_misses += 1

        # 3. Powerup proximity reward
        for powerup in self.powerups:
            dx = self.ship_pos[0] - powerup['pos'][0]
            dy = self.ship_pos[1] - powerup['pos'][1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < 100:
                reward += 0.01

        # 4. Speed penalty
        speed = np.linalg.norm(self.ship_vel)
        if speed > self.ship_max_speed * 0.8:
            reward -= 0.05

        # 5. Center of screen reward
        center_x = self.screen_size[0] // 2
        center_y = self.screen_size[1] // 2
        dx = self.ship_pos[0] - center_x
        dy = self.ship_pos[1] - center_y
        distance_to_center = math.sqrt(dx * dx + dy * dy)
        max_distance = math.sqrt(center_x * center_x + center_y * center_y)

        if max_distance > 0:
            center_reward = (1.0 - distance_to_center / max_distance) * 0.1
            reward += center_reward

        return reward

    def _get_state(self):
        """Get state vector (42 dimensions)"""
        state = []

        # Ship state (normalized)
        state.extend([self.ship_pos[0] / self.screen_size[0],
                      self.ship_pos[1] / self.screen_size[1]])
        state.extend([self.ship_vel[0] / self.ship_max_speed,
                      self.ship_vel[1] / self.ship_max_speed])

        # Nearest 5 asteroids
        if self.asteroids:
            sorted_asteroids = sorted(self.asteroids,
                                      key=lambda a: np.linalg.norm(a['pos'] - self.ship_pos))
            for i in range(5):
                if i < len(sorted_asteroids):
                    asteroid = sorted_asteroids[i]
                    state.extend([asteroid['pos'][0] / self.screen_size[0],
                                  asteroid['pos'][1] / self.screen_size[1],
                                  asteroid['vel'][0] / self.asteroid_max_speed,
                                  asteroid['vel'][1] / self.asteroid_max_speed,
                                  asteroid['size'] / 50.0])
                else:
                    state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            state.extend([0.0, 0.0, 0.0, 0.0, 0.0] * 5)

        # Nearest powerup
        if self.powerups:
            nearest_powerup = min(self.powerups,
                                  key=lambda p: np.linalg.norm(p['pos'] - self.ship_pos))
            state.extend([nearest_powerup['pos'][0] / self.screen_size[0],
                          nearest_powerup['pos'][1] / self.screen_size[1],
                          self.powerup_types.index(nearest_powerup['type']) / len(self.powerup_types)])
        else:
            state.extend([0.0, 0.0, 0.0])

        # Boundary distances
        state.extend([self.ship_pos[0] / self.screen_size[0],
                      (self.screen_size[0] - self.ship_pos[0]) / self.screen_size[0],
                      self.ship_pos[1] / self.screen_size[1],
                      (self.screen_size[1] - self.ship_pos[1]) / self.screen_size[1]])

        # Game state
        state.append(self.level / 10.0)
        state.append(len(self.asteroids) / self.max_asteroids)
        state.append(float(self.ship_has_shield))
        state.append(self.shield_time / self.max_shield_time if self.ship_has_shield else 0.0)

        return np.array(state, dtype=np.float32)

    def render(self):
        """Render game screen"""
        if self.render_mode != "human" or not hasattr(self, 'screen'):
            return

        try:
            # Draw background
            self.screen.fill(self.colors['background'])

            # Draw stars
            for star in self.stars:
                brightness = star['brightness'] * (0.8 + 0.2 * math.sin(self.time_alive * 0.01))
                color = tuple(int(c * brightness) for c in self.colors['star'])
                pygame.draw.circle(self.screen, color,
                                   (int(star['pos'][0]), int(star['pos'][1])), star['size'])

            # Draw particles
            for particle in self.particles:
                alpha = particle['lifetime'] / particle['max_lifetime']
                color = tuple(int(c * alpha) for c in particle['color'])
                pygame.draw.circle(self.screen, color,
                                   (int(particle['pos'][0]), int(particle['pos'][1])),
                                   particle['size'])

            # Draw powerups
            for powerup in self.powerups:
                pulse_size = powerup['size'] + 3 * math.sin(powerup['pulse'])
                color = self.colors[f'powerup_{powerup["type"]}']

                # Draw glow
                for i in range(3):
                    radius = pulse_size + i * 2
                    alpha = 100 - i * 30
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (*color[:3], alpha), (radius, radius), radius)
                    self.screen.blit(s, (powerup['pos'][0] - radius, powerup['pos'][1] - radius))

                # Draw powerup
                pygame.draw.circle(self.screen, color,
                                   (int(powerup['pos'][0]), int(powerup['pos'][1])),
                                   powerup['size'])

                # Draw symbol
                if powerup['type'] == 'shield':
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (int(powerup['pos'][0]), int(powerup['pos'][1])),
                                       powerup['size'] // 2, 2)
                elif powerup['type'] == 'laser':
                    pygame.draw.line(self.screen, (255, 255, 255),
                                     (int(powerup['pos'][0] - powerup['size'] // 2),
                                      int(powerup['pos'][1])),
                                     (int(powerup['pos'][0] + powerup['size'] // 2),
                                      int(powerup['pos'][1])), 3)

            # Draw asteroids
            for asteroid in self.asteroids:
                color = self.colors[
                    f'asteroid_{"large" if asteroid["size"] > 30 else "medium" if asteroid["size"] > 20 else "small"}']
                pygame.draw.circle(self.screen, color,
                                   (int(asteroid['pos'][0]), int(asteroid['pos'][1])),
                                   asteroid['size'])

                # Draw texture
                for _ in range(3):
                    angle = random.uniform(0, math.pi * 2)
                    dist = random.uniform(0, asteroid['size'] * 0.7)
                    x = asteroid['pos'][0] + math.cos(angle) * dist
                    y = asteroid['pos'][1] + math.sin(angle) * dist
                    pygame.draw.circle(self.screen, (color[0] - 20, color[1] - 20, color[2] - 20),
                                       (int(x), int(y)), asteroid['size'] // 6)

            # Draw ship
            ship_color = self.colors['ship_shield'] if self.ship_has_shield else self.colors['ship']

            # Draw trail
            speed = np.linalg.norm(self.ship_vel)
            if speed > 0.1:
                for i in range(3):
                    trail_length = 10 + i * 5
                    normalized_vel = self.ship_vel / speed if speed > 0 else np.array([0, 0])
                    trail_pos = self.ship_pos - normalized_vel * trail_length
                    alpha = 100 - i * 30
                    pygame.draw.circle(self.screen, (*ship_color[:3], alpha),
                                       (int(trail_pos[0]), int(trail_pos[1])),
                                       self.ship_size - i * 3)

            # Draw ship body
            pygame.draw.polygon(self.screen, ship_color, [
                (int(self.ship_pos[0]), int(self.ship_pos[1] - self.ship_size)),
                (int(self.ship_pos[0] - self.ship_size), int(self.ship_pos[1] + self.ship_size)),
                (int(self.ship_pos[0]), int(self.ship_pos[1] + self.ship_size // 2)),
                (int(self.ship_pos[0] + self.ship_size), int(self.ship_pos[1] + self.ship_size))
            ])

            # Draw shield
            if self.ship_has_shield:
                shield_radius = self.ship_size + 10
                for i in range(3):
                    radius = shield_radius + i * 2
                    alpha = 50 - i * 15
                    s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(s, (100, 255, 200, alpha), (radius, radius), radius, 2)
                    self.screen.blit(s, (self.ship_pos[0] - radius, self.ship_pos[1] - radius))

            # Draw UI
            info_lines = [
                f"Score: {self.score:.1f}",
                f"Time: {self.time_alive // 60}:{self.time_alive % 60:02d}",
                f"Level: {self.level}",
                f"Asteroids: {len(self.asteroids)}/{self.max_asteroids}",
                f"Near Misses: {self.near_misses}",
                f"Shield: {'ON' if self.ship_has_shield else 'OFF'}",
                f"Shield Time: {self.shield_time // 60}.{self.shield_time % 60:02d}" if self.ship_has_shield else ""
            ]

            for i, line in enumerate(info_lines):
                if line:
                    try:
                        text = self.font.render(line, True, self.colors['text'])
                        self.screen.blit(text, (10, 10 + i * 25))
                    except:
                        pass

            # Draw title
            try:
                title = self.title_font.render("SPACE DODGE", True, (200, 220, 255))
                self.screen.blit(title, (self.screen_size[0] // 2 - title.get_width() // 2, 10))
            except:
                pass

            # Draw controls
            controls = [
                "Controls: WASD/Arrow Keys to move",
                "Goal: Avoid asteroids, collect powerups",
                "Powerups: Shield (cyan), Laser (red), Slow (green)"
            ]

            for i, line in enumerate(controls):
                try:
                    text = self.small_font.render(line, True, (150, 170, 200))
                    self.screen.blit(text, (self.screen_size[0] // 2 - text.get_width() // 2,
                                            self.screen_size[1] - 100 + i * 20))
                except:
                    pass

            # Update display
            pygame.display.flip()
            self.clock.tick(60)

        except Exception as e:
            print(f"Rendering error: {e}")

    def close(self):
        """Close environment"""
        if hasattr(self, 'screen'):
            pygame.quit()

    def get_action_meanings(self):
        """Get action meanings"""
        return ["Up", "Down", "Left", "Right", "No-op"]