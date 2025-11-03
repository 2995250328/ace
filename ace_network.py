# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ace_util import get_pixel_grid

_logger = logging.getLogger(__name__)


class IntrinsicRayEncoding(nn.Module):
    """Sinusoidal encoding of camera rays using image intrinsics."""

    def __init__(self, feature_dim, subsampling_factor=8, scale=400.0):
        super().__init__()

        if feature_dim % 4 != 0:
            raise ValueError("feature_dim must be divisible by 4 for sine/cosine encoding")

        self.feature_dim = feature_dim
        self.subsampling_factor = subsampling_factor
        self.scale = scale

        frequency = torch.exp(
            torch.arange(0, feature_dim // 2, 2, dtype=torch.float32)
            * (-math.log(10000.0) / (feature_dim // 2))
        )
        self.register_buffer('frequency', frequency.view(1, -1), persistent=False)

        pixel_grid = get_pixel_grid(subsampling_factor)
        self.register_buffer('pixel_grid_2HW', pixel_grid, persistent=False)

    def _encode_directions(self, x_scaled, y_scaled):
        freq = self.frequency.to(x_scaled.dtype)

        sin_x = torch.sin(x_scaled.unsqueeze(1) * freq)
        cos_x = torch.cos(x_scaled.unsqueeze(1) * freq)
        sin_y = torch.sin(y_scaled.unsqueeze(1) * freq)
        cos_y = torch.cos(y_scaled.unsqueeze(1) * freq)

        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)


class IntrinsicFusion(nn.Module):
    """Fuse backbone features with intrinsic ray encodings using different strategies."""

    AVAILABLE_MODES = ("add", "concat_conv", "gated_add", "film")

    def __init__(self, feature_dim, mode="add"):
        super().__init__()

        self.feature_dim = feature_dim
        self.mode_modules = nn.ModuleDict()
        self.mode = None

        self.set_mode(mode)

    def set_mode(self, mode):
        if mode not in self.AVAILABLE_MODES:
            raise ValueError(f"Unsupported fusion mode '{mode}'. Supported modes: {self.AVAILABLE_MODES}")

        self.mode = mode

        if mode == "concat_conv":
            self._ensure_concat_conv()
        elif mode == "gated_add":
            self._ensure_gated_add()
        elif mode == "film":
            self._ensure_film()

    def _ensure_concat_conv(self):
        if "concat_conv" not in self.mode_modules:
            self.mode_modules["concat_conv"] = nn.Sequential(
                nn.Conv2d(self.feature_dim * 2, self.feature_dim, 1, bias=False),
                nn.BatchNorm2d(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.feature_dim, self.feature_dim, 1),
            )
        return self.mode_modules["concat_conv"]

    def _ensure_gated_add(self):
        if "gated_add" not in self.mode_modules:
            self.mode_modules["gated_add"] = nn.Conv2d(self.feature_dim * 2, self.feature_dim, 1)
        return self.mode_modules["gated_add"]

    def _ensure_film(self):
        if "film" not in self.mode_modules:
            self.mode_modules["film"] = nn.ModuleDict({
                "scale": nn.Conv2d(self.feature_dim, self.feature_dim, 1),
                "shift": nn.Conv2d(self.feature_dim, self.feature_dim, 1),
            })
        return self.mode_modules["film"]

    def forward(self, features, encoding):
        if self.mode == "add":
            return features + encoding

        if self.mode == "concat_conv":
            module = self._ensure_concat_conv()
            return module(torch.cat([features, encoding], dim=1))

        if self.mode == "gated_add":
            gate_conv = self._ensure_gated_add()
            gate = torch.sigmoid(gate_conv(torch.cat([features, encoding], dim=1)))
            return features + gate * encoding

        if self.mode == "film":
            film = self._ensure_film()
            scale = torch.sigmoid(film["scale"](encoding))
            shift = film["shift"](encoding)
            return features * scale + shift

        raise RuntimeError(f"Unhandled fusion mode '{self.mode}'")

    def encode_points(self, pixel_coords_b2, intrinsics_b33):
        if pixel_coords_b2.shape[0] != intrinsics_b33.shape[0]:
            raise ValueError("Pixel coordinate and intrinsics batches must have the same length")

        x = pixel_coords_b2[:, 0]
        y = pixel_coords_b2[:, 1]

        fx = intrinsics_b33[:, 0, 0]
        fy = intrinsics_b33[:, 1, 1]
        cx = intrinsics_b33[:, 0, 2]
        cy = intrinsics_b33[:, 1, 2]

        x_scaled = (x - cx) / fx * self.scale
        y_scaled = (y - cy) / fy * self.scale

        return self._encode_directions(x_scaled, y_scaled)

    def encode_grid(self, intrinsics_b33, spatial_size, dtype, device):
        B = intrinsics_b33.shape[0]
        H, W = spatial_size

        grid = self.pixel_grid_2HW[:, :H, :W].to(device=device, dtype=dtype)

        x = grid[0].unsqueeze(0).expand(B, -1, -1)
        y = grid[1].unsqueeze(0).expand(B, -1, -1)

        fx = intrinsics_b33[:, 0, 0].view(B, 1, 1)
        fy = intrinsics_b33[:, 1, 1].view(B, 1, 1)
        cx = intrinsics_b33[:, 0, 2].view(B, 1, 1)
        cy = intrinsics_b33[:, 1, 2].view(B, 1, 1)

        x_scaled = (x - cx) / fx * self.scale
        y_scaled = (y - cy) / fy * self.scale

        freq = self.frequency.view(1, -1, 1, 1).to(dtype=dtype, device=device)

        sin_x = torch.sin(x_scaled.unsqueeze(1) * freq)
        cos_x = torch.cos(x_scaled.unsqueeze(1) * freq)
        sin_y = torch.sin(y_scaled.unsqueeze(1) * freq)
        cos_y = torch.cos(y_scaled.unsqueeze(1) * freq)

        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)


class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=512):
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

    def forward(self, res):

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc


class Regressor(nn.Module):
    """
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(
        self,
        mean,
        num_head_blocks,
        use_homogeneous,
        num_encoder_features=512,
        intrinsics_fusion_mode="add",
    ):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim)
        self.ray_encoder = IntrinsicRayEncoding(self.feature_dim, self.OUTPUT_SUBSAMPLE)
        self.intrinsics_fusion = IntrinsicFusion(self.feature_dim, intrinsics_fusion_mode)
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @property
    def intrinsics_fusion_mode(self):
        return self.intrinsics_fusion.mode

    def _fuse_intrinsics(self, features, intrinsics):
        if intrinsics is None:
            return features

        ray_encoding = self.ray_encoder.encode_grid(
            intrinsics,
            spatial_size=features.shape[-2:],
            dtype=features.dtype,
            device=features.device,
        )
        return self.intrinsics_fusion(features, ray_encoding)

    def set_intrinsics_fusion_mode(self, mode):
        self.intrinsics_fusion.set_mode(mode)

    @classmethod
    def create_from_encoder(
        cls,
        encoder_state_dict,
        mean,
        num_head_blocks,
        use_homogeneous,
        intrinsics_fusion_mode="add",
    ):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(
            "Creating Regressor using pretrained encoder with %s feature size and '%s' fusion.",
            num_encoder_features,
            intrinsics_fusion_mode,
        )

        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, intrinsics_fusion_mode)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def _detect_intrinsics_fusion_mode(state_dict):
        if any(k.startswith("intrinsics_fusion.mode_modules.film") for k in state_dict):
            return "film"
        if any(k.startswith("intrinsics_fusion.mode_modules.gated_add") for k in state_dict):
            return "gated_add"
        if any(k.startswith("intrinsics_fusion.mode_modules.concat_conv") for k in state_dict):
            return "concat_conv"
        return "add"

    @classmethod
    def create_from_state_dict(cls, state_dict, intrinsics_fusion_mode=None):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        detected_mode = cls._detect_intrinsics_fusion_mode(state_dict)
        fusion_mode = intrinsics_fusion_mode or detected_mode

        _logger.info(
            "Creating regressor from pretrained state_dict:\n\tNum head blocks: %s\n\tHomogeneous coordinates: %s\n\tEncoder feature size: %s\n\tIntrinsics fusion: %s",
            num_head_blocks,
            use_homogeneous,
            num_encoder_features,
            fusion_mode,
        )

        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, fusion_mode)

        # Load all weights.
        strict_load = not (intrinsics_fusion_mode and fusion_mode != detected_mode)
        load_result = regressor.load_state_dict(state_dict, strict=strict_load)
        if load_result.missing_keys:
            _logger.debug("Missing keys while loading regressor: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            _logger.debug("Unexpected keys while loading regressor: %s", load_result.unexpected_keys)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(
        cls,
        encoder_state_dict,
        head_state_dict,
        intrinsics_fusion_mode=None,
    ):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        for k, v in head_state_dict.items():
            merged_state_dict[f"heads.{k}"] = v

        if intrinsics_fusion_mode == "auto":
            intrinsics_fusion_mode = None

        return cls.create_from_state_dict(merged_state_dict, intrinsics_fusion_mode)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs, intrinsics=None):
        features = self.encoder(inputs)
        return self._fuse_intrinsics(features, intrinsics)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs, intrinsics=None):
        """
        Forward pass.
        """
        features = self.get_features(inputs, intrinsics)
        return self.get_scene_coordinates(features)
