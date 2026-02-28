# DQN Training Script for Space Dodge Game
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.insert(0, os.getcwd())

from space_env import SpaceDodgeEnv
from space_agent import SpaceDQN


def train_space_dqn(num_episodes=500, save_interval=50, render_interval=100):
    """
    Train DQN agent for Space Dodge game

    Parameters:
    num_episodes: Number of training episodes
    save_interval: Interval for saving models
    render_interval: Interval for rendering display
    """
    print("=" * 70)
    print("Space Dodge Game - DQN Training")
    print("=" * 70)

    # Create environment
    env = SpaceDodgeEnv(render_mode=None)  # No rendering during training

    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 5  # Up, Down, Left, Right, No-op

    print(f"State dimension: {state_dim}")
    print(f"Action space: {action_dim} actions")
    print(f"Action meanings: {env.get_action_meanings()}")
    print("-" * 70)

    # Create agent
    agent = SpaceDQN(state_dim, action_dim)

    # Create save directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_logs", exist_ok=True)

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    epsilon_history = []
    best_score = -float('inf')

    print("Starting training...")
    print("-" * 70)

    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training progress", unit="episode")

    for episode in pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Occasionally render to observe training progress
        render_this_episode = (episode % render_interval == 0) and render_interval > 0

        if render_this_episode:
            env.close()
            env = SpaceDodgeEnv(render_mode="human")
            state = env.reset()

        while not done:
            # Select action using epsilon-greedy policy
            action = agent.select_action(state, eval_mode=False)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)

            # Train agent with experience replay
            loss = agent.replay()

            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Render if this episode is being visualized
            if render_this_episode:
                env.render()

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_scores.append(info['score'])
        epsilon_history.append(agent.epsilon)

        # Update progress bar description
        avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_length_10 = np.mean(episode_lengths[-10:]) if len(episode_lengths) >= 10 else np.mean(episode_lengths)

        pbar.set_postfix({
            'Reward': f"{episode_reward:.1f}",
            'Avg10R': f"{avg_reward_10:.1f}",
            'Length': episode_length,
            'Epsilon': f"{agent.epsilon:.3f}",
            'Memory': len(agent.memory)
        })

        # Print detailed information every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\nEpisode {episode + 1:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Length: {episode_length:5d} | "
                  f"Score: {info['score']:6.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Experiences: {len(agent.memory):5d} | "
                  f"Level: {info['level']}")

        # Save best model
        if info['score'] > best_score:
            best_score = info['score']
            agent.save("models/best_space_model.pkl")

            if info['score'] > 50:  # Only print when score is high
                print(f"\n⭐ New best model! Score: {best_score:.1f}")

        # Regular model saving
        if (episode + 1) % save_interval == 0:
            agent.save(f"models/space_model_episode_{episode + 1}.pkl")

            # Save training log
            log_data = {
                'episode': episode + 1,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'episode_scores': episode_scores,
                'epsilon_history': epsilon_history,
                'best_score': best_score,
                'agent_stats': agent.get_stats()
            }

            with open(f"training_logs/training_log_episode_{episode + 1}.json", 'w') as f:
                json.dump(log_data, f)

            # Generate training charts
            if episode >= 10:
                plot_training_progress(episode_rewards, episode_lengths, episode_scores, epsilon_history, episode + 1)

        # Restore non-rendering mode
        if render_this_episode:
            env.close()
            env = SpaceDodgeEnv(render_mode=None)

    # Save final model
    agent.save("models/final_space_model.pkl")

    # Plot final training results
    plot_final_results(episode_rewards, episode_lengths, episode_scores, epsilon_history)

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best score: {best_score:.1f}")
    print(f"Final exploration rate: {agent.epsilon:.4f}")
    print(f"Total training steps: {agent.training_steps}")
    print("=" * 70)

    env.close()

    return agent, episode_rewards


def plot_training_progress(rewards, lengths, scores, epsilons, episode):
    """Plot training progress at specified episode"""
    if episode % 100 != 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Smoothing function
    def smooth(data, window_size=10):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data) - window_size + 1):
            smoothed.append(np.mean(data[i:i + window_size]))
        return smoothed

    # 1. Reward curve
    axes[0, 0].plot(rewards, alpha=0.3, label='Raw', color='blue')
    if len(rewards) > 10:
        smoothed_rewards = smooth(rewards, 10)
        axes[0, 0].plot(range(9, len(rewards)), smoothed_rewards,
                        linewidth=2, label='Smoothed', color='red')
    axes[0, 0].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Episode Rewards (Episode {episode})', fontsize=12, fontweight='bold', pad=20)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode length
    axes[0, 1].plot(lengths, alpha=0.3, label='Raw', color='green')
    if len(lengths) > 10:
        smoothed_lengths = smooth(lengths, 10)
        axes[0, 1].plot(range(9, len(lengths)), smoothed_lengths,
                        linewidth=2, label='Smoothed', color='orange')
    axes[0, 1].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Steps', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Episode Length', fontsize=12, fontweight='bold', pad=20)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score curve
    axes[1, 0].plot(scores, alpha=0.3, label='Raw', color='purple')
    if len(scores) > 10:
        smoothed_scores = smooth(scores, 10)
        axes[1, 0].plot(range(9, len(scores)), smoothed_scores,
                        linewidth=2, label='Smoothed', color='magenta')
    axes[1, 0].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Game Scores', fontsize=12, fontweight='bold', pad=20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Exploration rate decay
    axes[1, 1].plot(epsilons, linewidth=2, color='brown')
    axes[1, 1].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Exploration Rate (ε)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Exploration Rate Decay', fontsize=12, fontweight='bold', pad=20)
    axes[1, 1].grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0.35, wspace=0.25)

    plt.savefig(f'training_logs/training_progress_episode_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training chart saved: training_logs/training_progress_episode_{episode}.png")


def plot_final_results(rewards, lengths, scores, epsilons):
    """Plot final training results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Smoothing function
    def smooth(data, window_size=20):
        if len(data) < window_size:
            return data
        smoothed = []
        for i in range(len(data) - window_size + 1):
            smoothed.append(np.mean(data[i:i + window_size]))
        return smoothed

    # 1. Reward curve
    axes[0, 0].plot(rewards, alpha=0.2, color='blue', label='Raw')
    if len(rewards) > 20:
        smoothed_rewards = smooth(rewards, 20)
        axes[0, 0].plot(range(19, len(rewards)), smoothed_rewards,
                        linewidth=2, color='red', label='Smoothed(20)')
    axes[0, 0].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Episode Reward Curve', fontsize=12, fontweight='bold', pad=20)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Episode length distribution
    axes[0, 1].hist(lengths, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Average: {np.mean(lengths):.0f}')
    axes[0, 1].set_xlabel('Episode Length', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Episode Length Distribution', fontsize=12, fontweight='bold', pad=20)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Score vs episode length relationship
    scatter = axes[1, 0].scatter(lengths, scores, c=range(len(scores)), cmap='viridis', alpha=0.6)
    axes[1, 0].set_xlabel('Episode Length', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Score vs Episode Length', fontsize=12, fontweight='bold', pad=20)
    colorbar = plt.colorbar(scatter, ax=axes[1, 0])
    colorbar.set_label('Episode Index', fontsize=10, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Exploration rate decay
    axes[1, 1].plot(epsilons, linewidth=2, color='brown')
    axes[1, 1].set_xlabel('Episode', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Exploration Rate (ε)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Exploration Rate Decay Curve', fontsize=12, fontweight='bold', pad=20)
    axes[1, 1].grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0.35, wspace=0.25)

    plt.savefig('training_final_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Final training chart saved: training_final_results.png")


if __name__ == "__main__":
    NUM_EPISODES = 300
    SAVE_INTERVAL = 50
    RENDER_INTERVAL = 100

    print("Space Dodge Game - DQN Training")
    print("=" * 50)
    print(f"Training episodes: {NUM_EPISODES}")
    print(f"Save interval: {SAVE_INTERVAL} episodes")
    print(f"Render interval: {RENDER_INTERVAL} episodes")
    print("=" * 50)

    # Start training
    try:
        agent, rewards = train_space_dqn(
            num_episodes=NUM_EPISODES,
            save_interval=SAVE_INTERVAL,
            render_interval=RENDER_INTERVAL
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current model...")
        agent.save("models/interrupted_space_model.pkl")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()