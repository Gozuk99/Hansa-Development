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


# # Assuming you've defined the dimensions and number of actions
# num_channels = ...  # Number of channels in input (e.g., different aspects of game state)
# board_height = ...
# board_width = ...
# num_actions = ...

# model = HansaNet(num_channels, board_height, board_width, num_actions)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for epoch in range(num_epochs):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         optimizer.zero_grad()  # Clear gradients
#         output = model(data)  # Forward pass
#         loss = F.cross_entropy(output, target)  # Compute loss
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Update weights

#         if batch_idx % log_interval == 0:
#             print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
