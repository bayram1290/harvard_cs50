import gymnasium as gym
import numpy as np


class QLearning:
    def __init__(self,
                env,
                learning_rate: float=0.1,
                discount_factor: float=0.95,
                exploration_rate: float=1.0,
                exploration_decay: float=0.995,
                min_exploration: float=0.01
                ) -> None:
        self.env = env
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration

        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))


    def _getAction(self, state: int) -> np.int64:
        # Exploration: random action
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        # Exploitation: best known action
        else:
            return np.argmax(self.q_table[state])


    def _updateQTable(self, state:int, action: np.int64, reward:int, next_state:int, done:bool) -> None:
        # Current Q-value
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # Maximum Q-value for next state
            max_next_q = np.max(self.q_table[next_state])
            target = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[state, action] = current_q + self.alpha * (target - current_q)


    def _decayExploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def modelTraining(self, episodes=10000) -> list:
        rewards = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self._getAction(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self._updateQTable(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            self._decayExploration()
            rewards.append(total_reward)

            if episode % 1000 == 0:
                avg_reward = np.mean(rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return rewards


    def modelTesting(self, episodes: int=10, render: bool=False) -> list:
        rewards = []

        for episode in range(episodes):
            state, _ = self.env.reset()
            reward_sum = 0
            done = False

            while not done:
                if render:
                    self.env.render()

                # Always choose best action during testing
                action = np.argmax(self.q_table[state])
                _, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                reward_sum += reward

            rewards.append(reward_sum)
            print(f"Test Episode {episode + 1}: Reward = {reward_sum}")

        print(f"Average test reward: {np.mean(rewards):.2f}")
        return rewards


def main():
    train_episode_count=10000
    test_episode_count=10

    env = gym.make('FrozenLake-v1', is_slippery=True)
    # Initialize agent
    agent = QLearning(env)

    # Train the agent
    print("Training the agent...")
    rewards = agent.modelTraining(train_episode_count)

    # Test the trained agent
    print("\nTesting the trained agent...")
    test_rewards = agent.modelTesting(test_episode_count)

    # Display final Q-table
    print("\nFinal Q-Table:")
    print(agent.q_table)

    env.close()


if __name__ == '__main__':
    main() 