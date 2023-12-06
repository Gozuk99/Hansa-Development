import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 1234

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

#current input size is 2309
#current output size is 681
class HansaNN(nn.Module):
    def __init__(self, input_size, output_size, model_file=None):
        super(HansaNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 2048).to(device)
        self.layer2 = nn.Linear(2048, 1024).to(device)
        self.layer3 = nn.Linear(1024, output_size).to(device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        if model_file and os.path.isfile(model_file):
            self.load_state_dict(torch.load(model_file, map_location=device))
            print(f"Model loaded from {model_file}")
        else:
            if model_file:
                print(f"No saved model found at {model_file}. Initializing new model.")

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.softmax(x)  # Apply softmax to the output layer
        return x

    def print_weights(self, layer_name, n=50, precision=4):
        """Print the first n weights of a specified layer of the model, with limited precision."""
        with torch.no_grad():  # Ensure no gradients are calculated
            layer = getattr(self, layer_name)
            weights = layer.weight.data.flatten()[:n]
            formatted_weights = torch.round(weights * (10 ** precision)) / (10 ** precision)
            print(f"{layer_name} first {n} weights: {formatted_weights.cpu().numpy()}")  # Convert to CPU and NumPy array for printing

# # Step 1: Displacement Decision
# state = get_current_state(game)
# displace_action = choose_displace_action(state)  # AI selects piece to displace
# game.execute_displacement(displace_action)
# intermediate_state = get_current_state(game)  # State after displacement

# # Step 2: Placement Decision
# placement_action = choose_placement_action(intermediate_state)  # AI or displaced player selects placement
# game.execute_placement(placement_action)
# new_state = get_current_state(game)  # Final state after placement

# # Rewards and Learning
# reward = calculate_reward(state, displace_action, placement_action, new_state)
# update_learning_algorithm(state, displace_action, intermediate_state, placement_action, reward, new_state)


# for episode in range(total_episodes):
#     state = env.reset()  # Reset the game to a starting state
#     done = False
#     total_reward = 0

#     while not done:
#         action = model.select_action(state)  # Select an action
#         next_state, reward, done, _ = env.step(action)  # Perform the action
#         model.remember(state, action, reward, next_state, done)  # Store the experience

#         model.learn()  # Train the model with experiences

#         state = next_state
#         total_reward += reward

#     print(f"Episode {episode}: Total Reward: {total_reward}")
