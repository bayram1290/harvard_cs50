import numpy as np

# Simple 3x3 grid world
# States: 0,1,2,3,4,5,6,7,8 (goal at state 8)
# Actions: 0=up, 1=right, 2=down, 3=left

class SimpleGridWorld:
    def __init__(self):
        self.n_states = 9
        self.n_actions = 4
        self.goal_state = 8
        
    def reset(self):
        return 0  # Start at state 0
    
    def step(self, state, action):
        # Define transitions
        transitions = {
            0: [0, 1, 3, 0],  # From state 0
            1: [1, 2, 4, 0],  # From state 1
            2: [2, 2, 5, 1],  # From state 2
            3: [0, 4, 6, 3],  # From state 3
            4: [1, 5, 7, 3],  # From state 4
            5: [2, 5, 8, 4],  # From state 5
            6: [3, 7, 6, 6],  # From state 6
            7: [4, 8, 7, 6],  # From state 7
            8: [8, 8, 8, 8]   # Goal state (absorbing)
        }
        
        next_state = transitions[state][action]
        
        # Reward: +10 for reaching goal, -1 otherwise
        reward = 10 if next_state == self.goal_state else -1
        
        done = (next_state == self.goal_state)
        
        return next_state, reward, done

# Simplified Q-learning
def simple_q_learning():
    env = SimpleGridWorld()
    
    # Initialize Q-table
    q_table = np.zeros((env.n_states, env.n_actions))
    
    # Hyperparameters
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    episodes = 1000
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Choose action (epsilon-greedy)
            if np.random.random() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(q_table[state])
            
            # Take action
            next_state, reward, done = env.step(state, action)
            
            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            state = next_state
    
    print("Learned Q-table:")
    print(q_table)
    
    # Test the policy
    state = env.reset()
    path = [state]
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        state, _, done = env.step(state, action)
        path.append(state)
    
    print(f"Optimal path: {path}")

# Run the simple example
simple_q_learning()