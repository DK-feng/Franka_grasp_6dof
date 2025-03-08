import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import gymnasium as gym
from torchvision.models import resnet18


class FCN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        arch: list = [128,256],
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        layers = []

        for out_dim in arch:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, features_dim))
        layers.append(nn.ReLU())

        self.linear = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(observations) # use current frame



class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x


class DeformableCNNT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "DeformCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.deform_cnn = nn.Sequential(
            DeformableConv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            DeformableConv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,-1,]
            n_flatten = self.deform_cnn(torch.as_tensor(each_T).float()).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.deform_cnn(observations[:,-1,])) # use current frame


class NatureCNNT(BaseFeaturesExtractor):
    """
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNNT must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,-1,]
            n_flatten = self.cnn(torch.as_tensor(each_T).float()).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations[:,-1,])) # use current frame
    

class DualCNNT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "DualCNNT must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        each_network_channels = int(n_input_channels/2)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            sample_array = observation_space.sample()[None][:,-1,]
            channel_dim = sample_array.shape[1]
            each_channel_dim = int(channel_dim/2)
            sample_tensor1 = torch.tensor(sample_array[:, :each_channel_dim, :, :]).float()
            sample_tensor2 = torch.tensor(sample_array[:, each_channel_dim:, :, :]).float()

            n_flatten1 = self.cnn1(sample_tensor1).shape[1]
            n_flatten2 = self.cnn2(sample_tensor2).shape[1]
        
        self.linear1 = nn.Sequential(nn.Linear(n_flatten1, int(features_dim/2)), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(n_flatten2, int(features_dim/2)), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        channel_dim = observations[:,-1,].shape[1]
        each_channel_dim = int(channel_dim/2)
        observations1 = observations[:,-1,][:, :each_channel_dim, :, :]
        observations2 = observations[:,-1,][:, each_channel_dim:, :, :]
        forward1 = self.linear1(self.cnn1(observations1))
        forward2 = self.linear2(self.cnn2(observations2))
        return torch.cat((forward1, forward2), dim=1)


class DeformDualCNNT(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "DualCNNT must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        each_network_channels = int(n_input_channels/2)

        self.deformcnn1 = nn.Sequential(
            DeformableConv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            DeformableConv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.deformcnn2 = nn.Sequential(
            DeformableConv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            DeformableConv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            sample_array = observation_space.sample()[None][:,-1,]
            channel_dim = sample_array.shape[1]
            each_channel_dim = int(channel_dim/2)
            sample_tensor1 = torch.tensor(sample_array[:, :each_channel_dim, :, :]).float()
            sample_tensor2 = torch.tensor(sample_array[:, each_channel_dim:, :, :]).float()

            n_flatten1 = self.deformcnn1(sample_tensor1).shape[1]
            n_flatten2 = self.deformcnn2(sample_tensor2).shape[1]
        
        self.linear1 = nn.Sequential(nn.Linear(n_flatten1, int(features_dim/2)), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(n_flatten2, int(features_dim/2)), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        channel_dim = observations[:,-1,].shape[1]
        each_channel_dim = int(channel_dim/2)
        observations1 = observations[:,-1,][:, :each_channel_dim, :, :]
        observations2 = observations[:,-1,][:, each_channel_dim:, :, :]
        forward1 = self.linear1(self.deformcnn1(observations1))
        forward2 = self.linear2(self.deformcnn2(observations2))
        return torch.cat((forward1, forward2), dim=1)


class DualLSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "DualCNNT must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        each_network_channels = int(n_input_channels/2)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(each_network_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten shape by doing one forward pass
        with torch.no_grad():
            sample_array = observation_space.sample()[None][:,-1,]
            channel_dim = sample_array.shape[1]
            each_channel_dim = int(channel_dim/2)
            sample_tensor1 = torch.tensor(sample_array[:, :each_channel_dim, :, :]).float()
            sample_tensor2 = torch.tensor(sample_array[:, each_channel_dim:, :, :]).float()

            n_flatten1 = self.cnn1(sample_tensor1).shape[1]
            n_flatten2 = self.cnn2(sample_tensor2).shape[1]
        
        self.linear1 = nn.Sequential(nn.Linear(n_flatten1, int(features_dim/2)), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(n_flatten2, int(features_dim/2)), nn.ReLU())
        
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        channel_dim = observations[:,-1,].shape[1]
        each_channel_dim = int(channel_dim/2)
        for t in range(observations.size(1)):
            observations1 = observations[:,t,][:, :each_channel_dim, :, :]
            observations2 = observations[:,t,][:, each_channel_dim:, :, :]
            forward1 = self.linear1(self.cnn1(observations1))
            forward2 = self.linear2(self.cnn2(observations2))
            x = torch.cat((forward1, forward2), dim=1)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc(out[-1, :, :])
        x = F.relu(x)
        return x


class CNNLSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.cnn(torch.as_tensor(each_T).float()).shape[1]

        self.fc_lstm = nn.Linear(n_flatten, features_dim)
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.cnn(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x


class CNNLSTM6Layers(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.cnn(torch.as_tensor(each_T).float()).shape[1]

        self.fc_lstm = nn.Linear(n_flatten, features_dim)
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.cnn(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x


class DeformCNNLSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "DeformCNNLSTM must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.deform_cnn = nn.Sequential(
            DeformableConv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            DeformableConv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            DeformableConv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.deform_cnn(torch.as_tensor(each_T).float()).shape[1]

        self.fc_lstm = nn.Linear(n_flatten, features_dim)
        self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.deform_cnn(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x


class Resnet18LSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        resnet = resnet18(pretrained=False)
        down_blocks = []
        self.input_block = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            
        )
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.Sequential(*down_blocks)
        self.resnet = nn.Sequential(self.input_block,
                                    self.down_blocks,
                                    nn.Flatten()
                                    )

        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.resnet(torch.as_tensor(each_T).float()).shape[1]

        self.fc_lstm = nn.Linear(n_flatten, features_dim)
        # self.lstm = nn.LSTM(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.fc1 = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.resnet(observations[:, t, :, :, :]) 
            x = self.fc_lstm(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)         

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        return x


class CNNGRU(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "CNNGRU must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[1] # [T, C, H, W]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            each_T = observation_space.sample()[None][:,0,]
            n_flatten = self.cnn(torch.as_tensor(each_T).float()).shape[1]

        self.fc_gru = nn.Linear(n_flatten, features_dim)
        self.gru = nn.GRU(input_size=features_dim, hidden_size=features_dim, num_layers=3)
        self.gru_fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = None
        for t in range(observations.size(1)):
            x = self.cnn(observations[:, t, :, :, :]) 
            x = self.fc_gru(x)
            out, hidden = self.gru(x.unsqueeze(0), hidden)         

        x = self.gru_fc(out[-1, :, :])
        x = F.relu(x)
        return x

