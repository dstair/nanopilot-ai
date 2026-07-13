import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import json

class TubDataset(Dataset):
    # Shared normalization constants
    IMU_SCALES = {
        'accel': 10.0, # m/s^2
        'gyro': 5.0    # rad/s
    }

    def __init__(self, tub_paths, seq_len=3, transform=None):
        """
        Args:
            tub_paths: List of paths to donkeycar tubs.
            seq_len: Number of historical IMU steps to include.
            transform: PyTorch transforms for the image.
        """
        self.tub_paths = tub_paths
        self.seq_len = seq_len
        self.transform = transform
        self.records = []
        self.imu_keys = [
            'imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'
        ]
        self.missing_imu_records = 0
        
        # Load all record metadata
        for tub_path in tub_paths:
            catalog_files = [
                f for f in os.listdir(tub_path)
                if f.startswith('catalog_') and f.endswith('.catalog')
            ]
            catalog_files.sort()
            
            for catalog_file in catalog_files:
                with open(os.path.join(tub_path, catalog_file), 'r') as f:
                    for line in f:
                        record = json.loads(line)
                        if 'cam/image_array' not in record:
                            continue
                        if any(k not in record for k in self.imu_keys):
                            self.missing_imu_records += 1
                        # Store absolute path to image and the raw IMU/Control data
                        record['_tub_path'] = tub_path
                        self.records.append(record)

    def __len__(self):
        return len(self.records)

    def normalize_imu(self, imu_data):
        """
        Normalizes IMU data. 
        Raw accel is ~±10-20 m/s^2, Gyro is ~±5 rad/s.
        We scale them down so they are in a similar magnitude to the [0, 1] image pixels.
        """
        # COORDINATE SYSTEM ALIGNMENT:
        # If the drone rolls when it should pitch, your IMU axes are likely swapped (NED vs ENU).
        # Standard NED (Expected): X=Forward, Y=Right, Z=Down
        # Standard ENU: X=Right, Y=Forward, Z=Up
        
        # To fix a 90-degree swap (ENU -> NED):
        val_x, val_y, val_z = imu_data[1], imu_data[0], imu_data[2]
        gyro_x, gyro_y, gyro_z = imu_data[4], imu_data[3], imu_data[5]

        normalized = np.array([
            val_x / self.IMU_SCALES['accel'], # acl_x
            val_y / self.IMU_SCALES['accel'], # acl_y
            val_z / self.IMU_SCALES['accel'], # acl_z
            gyro_x / self.IMU_SCALES['gyro'],  # gyr_x
            gyro_y / self.IMU_SCALES['gyro'],  # gyr_y
            gyro_z / self.IMU_SCALES['gyro']   # gyr_z
        ], dtype=np.float32)
        return normalized

    def __getitem__(self, idx):
        record = self.records[idx]
        tub_path = record['_tub_path']

        # 1. Load and Normalize Image
        img_path = os.path.join(tub_path, record['cam/image_array'])
        if not os.path.exists(img_path):
            img_path = os.path.join(tub_path, 'images', record['cam/image_array'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        else:
            # Default normalization [0, 1] and ToTensor (C, H, W)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # 2. Load and Normalize IMU Sequence
        # We need (seq_len, 6)
        imu_seq = []
        for i in range(idx - self.seq_len + 1, idx + 1):
            # Handle start-of-tub boundary by repeating the first available record
            safe_idx = max(0, i)
            # Ensure we don't accidentally pull a record from a different tub 
            # if tubs are concatenated
            if self.records[safe_idx]['_tub_path'] != tub_path:
                safe_idx = idx # Fallback to current if tub boundary crossed
            
            r = self.records[safe_idx]
            raw_imu = [r.get(k, 0) for k in self.imu_keys]
            imu_seq.append(self.normalize_imu(raw_imu))
        
        imu_tensor = torch.tensor(np.array(imu_seq), dtype=torch.float32)

        # 3. Load Targets [yaw, pitch, roll, altitude]
        def get_ctrl(r):
            return [
                r.get('user/angle', 0.0),
                r.get('user/throttle', 0.0),
                r.get('user/roll', 0.0),
                r.get('user/altitude', 0.0),
            ]

        targets = torch.tensor(get_ctrl(record), dtype=torch.float32)

        # NOTE: previous controls are deliberately NOT returned as a model input.
        # Consecutive tub controls are identical 90-99% of the time, so feeding the
        # previous control lets the model "cheat" by copying it and ignoring vision.
        return image, imu_tensor, targets

'''
### How to use this in your training script:
You can now replace your existing data loading logic with this `TubDataset`. It will automatically produce the `(3, 240, 320)` image tensors and the `(3, 6)` IMU tensors that your new `LinearModel` expects.

For the normalization, notice that I used `accel_scale = 10.0`. If you find the model is ignoring IMU data during training, you might want to try **Standardization** instead (subtracting the mean and dividing by the standard deviation), which ensures each feature has a mean of 0 and a variance of 1.

<!--
[PROMPT_SUGGESTION]How can I implement a script to calculate the mean and standard deviation of my IMU data for better normalization?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]Can you show me how to update the training loop in torch_train.py to use this new TubDataset?[/PROMPT_SUGGESTION]
'''
