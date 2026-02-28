# DQN Agent for Space Dodge Game
import numpy as np
import random
import pickle
import os
from collections import deque


class SpaceDQN:
    """Deep Q-Network agent for Space Dodge game"""

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Neural network architecture: Input -> Hidden(128) -> Hidden(64) -> Output
        self.hidden_size1 = 128
        self.hidden_size2 = 64

        # He initialization for weights
        self.W1 = np.random.randn(state_dim, self.hidden_size1) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros((1, self.hidden_size1))
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2) * np.sqrt(2.0 / self.hidden_size1)
        self.b2 = np.zeros((1, self.hidden_size2))
        self.W3 = np.random.randn(self.hidden_size2, action_dim) * np.sqrt(2.0 / self.hidden_size2)
        self.b3 = np.zeros((1, action_dim))

        # Experience replay buffer
        self.memory = deque(maxlen=20000)

        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = 0.001  # Learning rate
        self.batch_size = 64  # Batch size for training

        # Training statistics
        self.training_steps = 0
        self.total_loss = 0

        print(f"Initializing SpaceDQN (State dimension: {state_dim}, Action dimension: {action_dim})")
        print(f"Network architecture: {state_dim} -> {self.hidden_size1} -> {self.hidden_size2} -> {action_dim}")

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)

    def forward(self, state):
        """Forward propagation through the network"""
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # First hidden layer
        self.z1 = np.dot(state, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Second hidden layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        # Output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def predict(self, state):
        """Predict Q-values for given state"""
        q_values = self.forward(state)
        return q_values.flatten()

    def select_action(self, state, eval_mode=False):
        """Select action using epsilon-greedy policy"""
        if not eval_mode and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.action_dim)
        else:
            # Exploitation: action with highest Q-value
            q_values = self.predict(state)
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Experience replay training"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # Randomly sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to NumPy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Calculate target Q-values using Bellman equation
        next_q_values = np.array([self.predict(s) for s in next_states])
        max_next_q = np.max(next_q_values, axis=1)

        targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Forward pass to get current Q-values
        current_q_values = self.forward(states)

        # Create target array
        target_q = current_q_values.copy()
        for i in range(self.batch_size):
            target_q[i, actions[i]] = targets[i]

        # Backpropagation to update weights
        loss = self.backward(states, current_q_values, target_q)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Track training statistics
        self.training_steps += 1
        self.total_loss += loss

        return loss

    def backward(self, states, current_q, target_q):
        """Backpropagation to update network weights"""
        batch_size = states.shape[0]

        # Calculate output layer error
        error = current_q - target_q
        dloss_dz3 = error / batch_size

        # Third layer gradients
        dW3 = np.dot(self.a2.T, dloss_dz3)
        db3 = np.sum(dloss_dz3, axis=0, keepdims=True)

        # Second layer error
        dloss_da2 = np.dot(dloss_dz3, self.W3.T)
        dloss_dz2 = dloss_da2 * self.relu_derivative(self.z2)

        # Second layer gradients
        dW2 = np.dot(self.a1.T, dloss_dz2)
        db2 = np.sum(dloss_dz2, axis=0, keepdims=True)

        # First layer error
        dloss_da1 = np.dot(dloss_dz2, self.W2.T)
        dloss_dz1 = dloss_da1 * self.relu_derivative(self.z1)

        # First layer gradients
        dW1 = np.dot(states.T, dloss_dz1)
        db1 = np.sum(dloss_dz1, axis=0, keepdims=True)

        # Update weights using gradient descent
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

        # Calculate mean squared error loss
        loss = np.mean(np.square(error))
        return loss

    def save(self, filename):
        """Save model to file"""
        data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'total_loss': self.total_loss
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved: {filename}")
        print(f"Exploration rate: {self.epsilon:.4f}, Training steps: {self.training_steps}")

    def load(self, filename):
        """Load model from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.W3 = data['W3']
            self.b3 = data['b3']
            self.epsilon = data.get('epsilon', 0.1)
            self.training_steps = data.get('training_steps', 0)
            self.total_loss = data.get('total_loss', 0)

            print(f"Model loaded: {filename}")
            print(f"Exploration rate: {self.epsilon:.4f}, Training steps: {self.training_steps}")
            return True

        print(f"Model file not found: {filename}")
        return False

    def get_stats(self):
        """Get training statistics"""
        avg_loss = self.total_loss / max(1, self.training_steps)

        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.training_steps,
            'avg_loss': avg_loss
        }