import math
import random
import numpy as np

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