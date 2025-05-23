import math
import random
import numpy as np
import matplotlib.pyplot as plt

class RTDP:
    def __init__(self, state_space, actions):
        self.V = {s:0 for s in state_space}
        self.actions = actions
        self.state_space = state_space

    def terminal(self, state):
        x,y = state
        return x == 0 and y == 0

    def argmax(self, eval_function):
        if not self.actions:
            return None
        best_score_found_so_far = -math.inf
        best_action_found_so_far = None
        for current_action in self.actions:
            current_score = eval_function(current_action)
            if current_score > best_score_found_so_far:
                best_score_found_so_far = current_score
                best_action_found_so_far = current_action
        return best_action_found_so_far

    def random_action(self):
        return random.choice(self.actions)

    def get_next_state(self, state: tuple, action):
        x,y = state
        if action == "right" and x < 4:
            return (x + 1, y)
        if action == "left" and x > -4:
            return (x - 1, y)
        if action == "up" and y < 4:
            return (x, y + 1)
        if action == "down" and y > -4:
            return (x, y - 1)
        return state
    
    def reward(self, state, next_action, actual_next_state):
        if actual_next_state == (0,0):
            return 100
        if actual_next_state == state:
            return -10
        return -1

    def rtdp(self, state_space, actions, initial_state):

        gamma = 0.99 #discount factor from bellman equation
        epsilon = 0.9 # exploration rate
        max_episodes = 1000
        epsilon_decay = 0.995

        # Initialize the terminal state value
        self.V[(0,0)] = 100

        for episode in range(max_episodes):
            state = initial_state
            while not self.terminal(state):
                # epsilon-greedy method
                best_action = self.argmax(lambda a: self.V[self.get_next_state(state, a)])
                next_action = self.random_action() if random.random() < epsilon else best_action
                actual_next_state = self.get_next_state(state, next_action)
                # Bellman update
                max_future_value = max(self.V[self.get_next_state(state, a)] for a in actions)
                reward_value = self.reward(state, next_action, actual_next_state)
                self.V[state] = reward_value + gamma * max_future_value
                state = actual_next_state
            epsilon *= epsilon_decay
        return self.V

    def plot_value_function(self, show_values=True, highlight_target=True, cmap='viridis', figsize=(10, 10)):
        # Fixed dimensions for 9x9 grid
        min_coord = -4
        max_coord = 4
        grid_size = 9
        
        # Create a 2D array to hold the values
        value_grid = np.zeros((grid_size, grid_size))
        
        # Populate the grid with values from self.V
        for (x, y), value in self.V.items():
            # Convert world coordinates (-4 to 4) to array indices (0 to 8)
            row_index = y + 4
            col_index = x + 4
            value_grid[row_index, col_index] = value
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create the heatmap
        im = plt.imshow(value_grid, cmap=cmap, origin='lower', interpolation='nearest')
        plt.colorbar(im, label="State Value", shrink=0.8)
        
        # Set ticks and labels
        tick_positions = np.arange(grid_size)
        tick_labels = np.arange(min_coord, max_coord + 1)
        
        plt.xticks(tick_positions, labels=tick_labels)
        plt.yticks(tick_positions, labels=tick_labels)
        
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.title("Learned Value Function (V) - 9x9 Grid", fontsize=14)
        
        # Highlight the target cell if requested
        if highlight_target:
            target_x_idx = 4
            target_y_idx = 4
            plt.scatter(target_x_idx, target_y_idx, c='red', s=300, marker='*', 
                        edgecolors='white', linewidth=3, label='Target (0,0)', zorder=5)
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add text annotations if requested
        if show_values:
            for y_idx in range(grid_size):
                for x_idx in range(grid_size):
                    val = value_grid[y_idx, x_idx]
                    
                    # Choose text color for better contrast
                    text_color = "white" if val < np.mean(value_grid) else "black"
                    
                    # Special formatting for target cell
                    if x_idx == 4 and y_idx == 4:
                        plt.text(x_idx, y_idx, f"{val:.2f}", 
                                ha="center", va="center", color="yellow", 
                                fontsize=10, fontweight='bold')
                    else:
                        plt.text(x_idx, y_idx, f"{val:.2f}", 
                                ha="center", va="center", color=text_color, fontsize=9)
        
        # Add grid lines
        plt.grid(True, color='gray', linewidth=0.5, alpha=0.7)
        
        # Set equal aspect ratio
        plt.gca().set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Value function statistics:")
        print(f"Min value: {np.min(value_grid):.3f}")
        print(f"Max value: {np.max(value_grid):.3f}")
        print(f"Mean value: {np.mean(value_grid):.3f}")
        print(f"Target cell (0,0) value: {value_grid[4, 4]:.3f}")

if __name__ == "__main__":
    actions = ["right", "left", "up","down"]
    state_space = []
    for x in range(-4,5):
        for y in range(-4,5):
            state_space.append((x,y))
    rtdp_agent = RTDP(state_space, actions)
    response = rtdp_agent.rtdp(state_space, actions, (2,2))
    for i in range(-4,5):
        for j in range(-4,5):
            state = (i,j)
            if state in response:
                print(f"{response[state]:8.2f}", end=" ")
            else:
                print("  NA  ", end=" ")
        print()
    rtdp_agent.plot_value_function()