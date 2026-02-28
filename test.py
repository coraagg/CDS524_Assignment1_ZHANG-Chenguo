# Testing Script for Space Dodge Game
import pygame
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from space_env import SpaceDodgeEnv
from space_agent import SpaceDQN


def test_trained_agent(model_path="models/best_space_model.pkl", num_episodes=5):
    """Test trained AI agent"""
    print("Testing trained agent...")
    print("=" * 60)

    # Create environment
    env = SpaceDodgeEnv(render_mode="human")

    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 5

    print(f"State dimension: {state_dim}")
    print(f"Action space: {action_dim} actions")
    print(f"Action meanings: {env.get_action_meanings()}")
    print("-" * 60)

    # Create agent
    agent = SpaceDQN(state_dim, action_dim)

    # Load trained model
    if not agent.load(model_path):
        print(f"Cannot load model: {model_path}")
        print("Please run training script first: python train_space.py")
        return

    print(f"Successfully loaded model: {model_path}")
    print(f"Exploration rate: {agent.epsilon:.4f}")
    print(f"Training steps: {agent.training_steps}")
    print("-" * 60)

    # Disable exploration (use greedy policy)
    agent.epsilon = 0.0

    # Test statistics
    test_rewards = []
    test_scores = []
    test_lengths = []
    test_near_misses = []

    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nTest episode {episode + 1}/{num_episodes}")
        print("-" * 30)

        while not done:
            # Use trained policy to select action
            action = agent.select_action(state, eval_mode=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Render game
            env.render()

            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                    elif event.key == pygame.K_r:
                        done = True  # Restart

        # Record results
        test_rewards.append(episode_reward)
        test_scores.append(info['score'])
        test_lengths.append(episode_length)
        test_near_misses.append(info['near_misses'])

        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Game score: {info['score']:.2f}")
        print(f"Survival time: {episode_length} frames ({episode_length // 60}.{episode_length % 60:02d} seconds)")
        print(f"Level: {info['level']}")
        print(f"Near misses: {info['near_misses']}")
        print(f"Remaining asteroids: {info['asteroids']}")

    # Print test summary
    print("\n" + "=" * 60)
    print("Test results summary:")
    print(f"Average reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average score: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
    print(f"Average length: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"Average near misses: {np.mean(test_near_misses):.1f} ± {np.std(test_near_misses):.1f}")
    print(f"Max score: {np.max(test_scores):.2f}")
    print(f"Min score: {np.min(test_scores):.2f}")
    print("=" * 60)

    env.close()

    return test_rewards, test_scores, test_lengths


def human_play():
    """Human control mode for playing the game"""
    print("Human control mode - Space Dodge")
    print("=" * 60)
    print("Control instructions:")
    print(" ↑: Move up")
    print(" ↓: Move down")
    print(" ←: Move left")
    print(" →: Move right")
    print(" R: Restart game")
    print(" ESC: Exit game")
    print("=" * 60)
    print("Game goals:")
    print(" 1. Avoid asteroids and survive as long as possible")
    print(" 2. Collect powerups for special abilities")
    print(" 3. Get high score")
    print("=" * 60)
    print("Powerup descriptions:")
    print(" 🛡️ Shield (cyan): Temporary invincibility, can crash into asteroids")
    print(" 🔥 Laser (red): Clear all asteroids on screen")
    print(" 🐌 Slow (green): Slow down all asteroids")
    print("=" * 60)

    # Create environment
    env = SpaceDodgeEnv(render_mode="human")
    state = env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:
        # Default action: No-op
        action = 4

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = env.reset()
                    print("Game reset!")

        # Continuous key detection
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 0  # Up
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 1  # Down
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = 2  # Left
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = 3  # Right

        # Execute action
        next_state, reward, done, info = env.step(action)
        state = next_state

        # Render game
        env.render()

        # Control frame rate
        clock.tick(60)

        if done:
            print(f"\n{'=' * 50}")
            print("Game over!")
            print(f"Final score: {info['score']:.2f}")
            print(f"Survival time: {info['time_alive']} frames ({info['time_alive'] // 60}.{info['time_alive'] % 60:02d} seconds)")
            print(f"Highest level: {info['level']}")
            print(f"Near misses: {info['near_misses']}")
            print(f"{'=' * 50}")

            # Show rating based on performance
            score = info['score']
            if score < 20:
                rating = "Beginner"
                tip = "Try staying in the center of the screen, watch asteroid trajectories."
            elif score < 50:
                rating = "Skilled"
                tip = "Good! Try predicting asteroid movement, collect powerups."
            elif score < 100:
                rating = "Expert"
                tip = "Excellent! You've mastered dodging skills."
            else:
                rating = "Master"
                tip = "Amazing! You're a true space pilot!"

            print(f"Rating: {rating}")
            print(f"Tip: {tip}")
            print(f"{'=' * 50}")

            response = input("\nPlay again? (y/n): ")
            if response.lower() == 'y':
                state = env.reset()
                print("\nNew game started!")
            else:
                running = False

    env.close()
    print("\nGame over, thanks for playing!")


def benchmark_agent(model_path="models/best_space_model.pkl", num_episodes=20):
    """Run benchmark test without rendering for faster evaluation"""
    print("Running benchmark test...")
    print("=" * 60)

    # Create environment without rendering
    env = SpaceDodgeEnv(render_mode=None)

    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 5

    # Create agent
    agent = SpaceDQN(state_dim, action_dim)

    # Load model
    if not agent.load(model_path):
        print(f"Cannot load model: {model_path}")
        return

    # Disable exploration
    agent.epsilon = 0.0

    # Test statistics
    all_scores = []
    all_times = []
    all_levels = []

    print("Running benchmark test...")

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        episode_time = 0

        while not done:
            # Select action using trained policy
            action = agent.select_action(state, eval_mode=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Update state
            state = next_state
            episode_score = info['score']
            episode_time = info['time_alive']

        # Record results
        all_scores.append(episode_score)
        all_times.append(episode_time)
        all_levels.append(info['level'])

        # Show progress
        if (episode + 1) % 5 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")

    # Print benchmark results
    print("\nBenchmark results:")
    print("-" * 40)
    print(f"Test episodes: {num_episodes}")
    print(f"Average score: {np.mean(all_scores):.2f} ± {np.std(all_scores):.2f}")
    print(f"Average survival time: {np.mean(all_times):.1f} frames ({np.mean(all_times) // 60:.0f}.{np.mean(all_times) % 60:.0f} seconds)")
    print(f"Average highest level: {np.mean(all_levels):.1f}")
    print(f"Max score: {np.max(all_scores):.2f}")
    print(f"Min score: {np.min(all_scores):.2f}")
    print(f"Median score: {np.median(all_scores):.2f}")
    print("-" * 40)

    # Rating standards
    avg_score = np.mean(all_scores)
    if avg_score < 20:
        grade = "D (Needs more training)"
    elif avg_score < 40:
        grade = "C (Basic qualification)"
    elif avg_score < 70:
        grade = "B (Good)"
    elif avg_score < 100:
        grade = "A (Excellent)"
    else:
        grade = "S (Outstanding)"

    print(f"Agent grade: {grade}")
    print("=" * 60)

    env.close()

    return all_scores, all_times, all_levels


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Space Dodge Game Testing")
    parser.add_argument("--mode", type=str, default="ai",
                        choices=["ai", "human", "benchmark"],
                        help="Test mode: ai=AI test, human=human control, benchmark=benchmark test")
    parser.add_argument("--model", type=str, default="models/best_space_model.pkl",
                        help="Model path")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of test episodes")

    args = parser.parse_args()

    if args.mode == "ai":
        test_trained_agent(
            model_path=args.model,
            num_episodes=args.episodes
        )
    elif args.mode == "human":
        human_play()
    elif args.mode == "benchmark":
        benchmark_agent(
            model_path=args.model,
            num_episodes=args.episodes
        )