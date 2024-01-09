import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch


class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        self.num_symbols, self.num_features, self.frame_stack_size = observation_space.shape

        # 1D CNN layers for processing each feature across the frame stack
        # These layers will be applied to each symbol separately
        # They will also act as an encoder for the features
        # Input is [batch, features, frame_stack] (frame_stack is the time dimension or history of features)
        self.conv1 = nn.Conv1d(
            in_channels=self.num_features, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate the output size after convolutions
        # update pooling here if added or removed
        dummy_input = torch.zeros(1, self.num_features, self.frame_stack_size)
        dummy_output = self.pool(self.conv3(
            self.conv2(self.pool(self.conv1(dummy_input)))))

        # Multiply by the number of symbols
        conv_output_size = dummy_output.numel() * self.num_symbols

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.fc2 = nn.Linear(64, features_dim)
        self.relu = nn.ReLU()
        # Use sigmoid for the last layer to get values between 0 and 1
        self.sigmoid = nn.Sigmoid()
        # Use tanh for the last layer to get values between -1 and 1
        self.tanh = nn.Tanh()

    def forward(self, observations):
        batch_size = observations.shape[0]
        # Rearrange dimensions: [batch, features, symbols, frame_stack]
        observations = observations.permute(0, 2, 1, 3).contiguous()

        # Process observations through the 1D CNN layers
        x = observations.view(batch_size * self.num_symbols,
                              self.num_features, self.frame_stack_size)

        # note add or remove pooling layers if large or small data (frame stack vs features)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # Reshape to separate symbols
        x = x.view(batch_size, self.num_symbols, -1)

        # Flatten and concatenate outputs for all symbols
        x = x.view(batch_size, -1)

        # Fully connected layers
        x = self.relu(self.fc1(x))

        # Use sigmoid for the last layer to get values between 0 and 1
        # or use tanh for the last layer to get values between -1 and 1
        x = self.tanh(self.fc2(x))

        return x


# another architecture to try

# class CustomNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim):
#         super(CustomNetwork, self).__init__(observation_space, features_dim)

#         # Extract the number of stock symbols, features, and frame stack size from the observation space
#         num_symbols, num_features, frame_stack_size = observation_space.shape

#         # CNN layers
#         self.conv1 = nn.Conv2d(in_channels=num_symbols, out_channels=32, kernel_size=(
#             3, 3), stride=1, padding=1)

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)

#         # Additional convolutional layer with max pooling
#         self.conv3 = nn.Conv2d(
#             64, 128, kernel_size=(3, 3), stride=1, padding=1)

#         # Max pooling layer
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Update dummy input calculation
#         dummy_input = torch.zeros(
#             1, num_symbols, num_features, frame_stack_size)

#         dummy_output = self.pool(self.conv3(
#             self.pool(self.conv2(self.conv1(dummy_input)))))

#         conv_output_size = torch.prod(
#             torch.tensor(dummy_output.shape[1:])).item()

#         # Update fully connected layers
#         self.fc1 = nn.Linear(conv_output_size, 128)
#         self.fc2 = nn.Linear(128, features_dim)  # Adjust output_dim as needed
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, observations):
#         # Apply convolutions
#         x = nn.functional.relu(self.conv1(observations))
#         x = self.pool(x)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.relu(self.conv3(x))
#         x = self.pool(x)

#         # Flatten the output for the fully connected layers
#         x = x.view(x.size(0), -1)

#         # Fully connected layers
#         x = nn.functional.relu(self.fc1(x))
#         x = self.tanh(self.fc2(x))
#         return x
