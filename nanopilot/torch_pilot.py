"""
TorchPilot — drop-in replacement for KerasLinear in the DonkeyCar vehicle loop.

Implements the same run(img_arr) → (yaw, pitch, roll, altitude) interface.
"""

import math

import numpy as np
import torch

try:
    from .torch_model import LinearModel
except ImportError:
    from torch_model import LinearModel
from tub_schema import heading_frame_velocity


class TorchPilot:
    def __init__(self, input_shape=(3, 120, 160), seq_len=3,
                 altitude_pwm_scale=None):
        """
        Args:
            input_shape: (C, H, W) channels-first, matching LinearModel convention.
            altitude_pwm_scale: must match what the model was TRAINED with (see
                TubDataset). When set, the model's altitude output is in
                throttle-PWM space and is raised to the power 1/scale here to
                recover the raw altitude command DroneGymEnv expects. Leaving
                this unset for a model trained with it (or vice versa) produces
                a silently wrong altitude axis, not an error.
        """
        self.input_shape = input_shape
        self.seq_len = seq_len
        self.altitude_pwm_scale = altitude_pwm_scale
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = LinearModel(input_shape=input_shape, imu_shape=(seq_len, 6)).to(self.device)
        self.model.eval()
        self._imu_history = np.zeros((seq_len, 6), dtype=np.float32)

    def load(self, model_path):
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"TorchPilot: loaded {model_path} on {self.device}")

    def run(self, img_arr, acl_x=0.0, acl_y=0.0, acl_z=0.0, gyr_x=0.0, gyr_y=0.0,
            gyr_z=0.0, vel_x=0.0, vel_y=0.0, vel_z=0.0, yaw_deg=0.0):
        """Argument order matches drone_manage's part inputs:
        cam/image_array, IMU_KEYS, VELOCITY_KEYS, imu/yaw.

        Velocity is WORLD frame here (as recorded/telemetered); it is rotated
        into the heading frame internally by the same helper the dataset uses.
        """
        if img_arr is None:
            return 0.0, 0.0, 0.0, 0.0

        # img_arr: (H, W, C) uint8 numpy array from camera
        arr = np.asarray(img_arr, dtype=np.float32) / 255.0
        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        # Downscale to the training resolution so runtime features match training.
        # (AdaptiveAvgPool keeps the model runnable at any size, but the conv
        # receptive fields only match what it was trained on at that resolution.)
        target_h, target_w = self.input_shape[1], self.input_shape[2]
        if tensor.shape[-2:] != (target_h, target_w):
            # antialias=True is REQUIRED, not cosmetic: TubDataset resizes with
            # PIL (which antialiases), and at the 4x downscale from the 640x480
            # camera to 120x160 plain bilinear aliases hard -- measured max pixel
            # difference 0.355 (35% of range) vs 0.003 with antialiasing. Without
            # it every deployed model flies on different images than it trained
            # on, with no error to notice.
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_h, target_w),
                mode='bilinear', align_corners=False, antialias=True,
            )
        # Swapping acl_x/y and gyr_x/y to match the dataset's coordinate correction
        imu_sample = np.array(
            [acl_y / 10.0, acl_x / 10.0, acl_z / 10.0, gyr_y / 5.0, gyr_x / 5.0, gyr_z / 5.0],
            dtype=np.float32,
        )
        self._imu_history = np.roll(self._imu_history, shift=-1, axis=0)
        self._imu_history[-1] = imu_sample
        imu_tensor = torch.from_numpy(self._imu_history).unsqueeze(0).to(self.device)

        vel_tensor = torch.tensor(
            [heading_frame_velocity(vel_x, vel_y, vel_z, yaw_deg)],
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(tensor, imu_tensor, vel_tensor)

        yaw = float(output[0, 0])
        throttle = float(output[0, 1])
        roll = float(output[0, 2])
        altitude = float(output[0, 3])
        if self.altitude_pwm_scale:
            # Model predicted in PWM space; invert to the raw command that
            # DroneGymEnv will re-expand with sign(a)*|a|**scale.
            altitude = math.copysign(
                abs(altitude) ** (1.0 / self.altitude_pwm_scale), altitude
            )
        return yaw, throttle, roll, altitude
