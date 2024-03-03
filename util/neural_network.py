import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomNetwork(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomNetwork, self).__init__(observation_space, features_dim)

        # Extract shapes from the observation_space dict
        stacked_obs_space = observation_space.spaces['stacked_obs']
        self.num_symbols, self.num_features, self.frame_stack_size = stacked_obs_space.shape

        # 1D CNN layers
        self.conv1 = nn.Conv1d(in_channels=self.num_features,
                               out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 8, kernel_size=3, stride=1, padding=1)

        # Pooling layer to reduce the dimensionality
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dummy input for output size calculation
        dummy_input = torch.zeros(1, self.num_features, self.frame_stack_size)
        dummy_output = self.conv2(self.pool(self.conv1(dummy_input)))
        conv_output_size = dummy_output.numel() * self.num_symbols

        # Adjust the fully connected layer input size to include holdings and unrealized PL
        total_fc_input_size = conv_output_size

        # Fully connected layers
        self.fc1 = nn.Linear(total_fc_input_size, features_dim*2)
        self.fc2 = nn.Linear(features_dim*2, features_dim)

        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, observations):
        # Extract each component from the dictionary
        stacked_obs = observations['stacked_obs']

        batch_size = stacked_obs.shape[0]
        x = stacked_obs.view(batch_size * self.num_symbols,
                             self.num_features, self.frame_stack_size)

        # Apply 1D CNN
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))

        # Flatten CNN output
        x = x.view(batch_size, -1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))

        return x


# # another architecture to try

# class CustomNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim):
#         super(CustomNetwork, self).__init__(observation_space, features_dim)

#         # Extract the number of stock symbols, features, and frame stack size from the observation space
#         num_symbols, num_features, frame_stack_size = observation_space.shape

#         # CNN layers
#         self.conv1 = nn.Conv2d(in_channels=num_symbols, out_channels=16, kernel_size=(
#             3, 3), stride=1, padding=1)

#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)

#         # Additional convolutional layer with max pooling
#         self.conv3 = nn.Conv2d(
#             32, 32, kernel_size=(3, 3), stride=1, padding=1)

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


# class CustomNetwork(BaseFeaturesExtractor):
#     def __init__(self, observation_space, features_dim):
#         super(CustomNetwork, self).__init__(observation_space, features_dim)

#         self.num_symbols, self.num_features, self.frame_stack_size = observation_space.shape

#         # Adjust the in_channels of the first 1D convolution to match the number of features
#         self.temporal_conv1 = nn.Conv1d(
#             self.num_features, 16, kernel_size=3, stride=1, padding=1)
#         self.temporal_conv2 = nn.Conv1d(
#             16, 32, kernel_size=3, stride=1, padding=1)

#         # The number of input channels to the first 2D convolution should match the output channels of the last 1D convolution
#         self.spatial_conv1 = nn.Conv2d(
#             32, 64, kernel_size=(3, 3), stride=1, padding=1)
#         self.spatial_conv2 = nn.Conv2d(
#             64, 64, kernel_size=(3, 3), stride=1, padding=1)

#         # Pooling layers
#         self.pool1d = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Dummy input for output size calculations
#         dummy_input_1d = torch.zeros(
#             1, self.num_features, self.frame_stack_size)
#         dummy_output_1d = self.pool1d(self.temporal_conv2(
#             self.pool1d(self.temporal_conv1(dummy_input_1d))))
#         # Adjust the number of channels to match the output of temporal_conv2
#         dummy_input_2d = torch.zeros(
#             1, 32, self.num_symbols, dummy_output_1d.shape[2])
#         dummy_output_2d = self.pool2d(self.spatial_conv2(
#             self.pool2d(self.spatial_conv1(dummy_input_2d))))

#         conv_output_size = torch.prod(
#             torch.tensor(dummy_output_2d.shape[1:])).item()

#         # Fully connected layers
#         self.fc1 = nn.Linear(conv_output_size, 128)
#         self.fc2 = nn.Linear(128, features_dim)
#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()

#     def forward(self, observations):
#         batch_size = observations.shape[0]

#         # Temporal feature extraction
#         x = observations.view(
#             batch_size * self.num_symbols, self.num_features, -1)
#         x = self.relu(self.temporal_conv1(x))
#         x = self.pool1d(x)
#         x = self.relu(self.temporal_conv2(x))
#         x = self.pool1d(x)

#         # Rearranging for spatial processing
#         x = x.view(batch_size, 32, self.num_symbols, -1)

#         # Spatial feature extraction
#         x = self.relu(self.spatial_conv1(x))
#         x = self.pool2d(x)
#         x = self.relu(self.spatial_conv2(x))
#         x = self.pool2d(x)

#         # Flatten and pass through fully connected layers
#         x = x.view(batch_size, -1)
#         x = self.relu(self.fc1(x))
#         x = self.tanh(self.fc2(x))

#         return x
