import torch
import torch.nn as nn
import torch.nn.functional as F

class HansaNet(nn.Module):
    def __init__(self, num_channels, board_height, board_width, num_actions):
        super(HansaNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * board_height * board_width, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# class HansaNN(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(HansaNN, self).__init__()
#         self.layer1 = nn.Linear(input_size, 128)
#         self.layer2 = nn.Linear(128, 64)
#         self.layer3 = nn.Linear(64, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.layer1(x))
#         x = self.relu(self.layer2(x))
#         x = self.layer3(x)  # No activation here as this is your output layer
#         return x


# # # Assuming you've defined the dimensions and number of actions
# # num_channels = ...  # Number of channels in input (e.g., different aspects of game state)
# # board_height = ...
# # board_width = ...
# # num_actions = ...

# # model = HansaNet(num_channels, board_height, board_width, num_actions)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # for epoch in range(num_epochs):
# #     for batch_idx, (data, target) in enumerate(train_loader):
# #         optimizer.zero_grad()  # Clear gradients
# #         output = model(data)  # Forward pass
# #         loss = F.cross_entropy(output, target)  # Compute loss
# #         loss.backward()  # Backpropagation
# #         optimizer.step()  # Update weights

# #         if batch_idx % log_interval == 0:
# #             print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

# # Initialize neural network instances for each AI agent
# nn_agents = [NeuralNetwork() for _ in range(5)]

# # Game loop
# while not game_over:
#     current_player_index = game.get_current_player_index()
#     current_nn = nn_agents[current_player_index]

#     action = current_nn.decide_action(game_state)
#     game.execute_action(action)

#     reward = calculate_reward_for_agent(current_player_index)
#     current_nn.update(action, game_state, reward)

#     game_state = get_current_state(game)

# # Initialize a single neural network for shared learning
# shared_nn = NeuralNetwork()

# # Game loop
# while not game_over:
#     if current_player in [agent1, agent2]:
#         action = shared_nn.decide_action(game_state)
#         game.execute_action(action)
#         reward = calculate_reward(current_player)
#         shared_nn.update(action, game_state, reward)

#     game_state = get_current_state(game)


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



# #create action space
# #1 INCOME - 5 valid options: 0-4 circles, and squares leftover - mask out options based on circles
# #2 CLAIM - all posts claim with circle, or square - mask out options with incorrect region(blue or brown priv), incorrect personal_supply, occupied, incorrect shape
# #3 DISPLACE - all posts owned by opponents - mask out if invalid personal supply,
# #4 MOVE - query all posts -> add least desirable owned piece to array, loop for player.book -> when met, loop for player.book -> ONLY VALID ACTIONS are CLAIM, same rules.
# #5 CLAIM - query all routes owned by player - add claim city office, upgrade X, claim route for points
# #6 BM - ?
# #7 PERM BM - Use immediately
# #8 REPLACE BM - Choose a route?
# #9 PICK UP - all posts owned by me, mask out not owned, add to array.
# #10 END TURN - use BM or End Turn