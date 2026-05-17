import torch
import torch.nn as nn
import time
from torch_model import LinearModel

def run_smoke_test():
    """
    A quick training run using synthetic data to verify the model 
    architecture and training pipeline.
    """
    # Detect device (M1/M2/M3 Mac support)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"--- Starting Smoke Test on {device} ---")

    # Parameters matching drone_config_65mm.py
    image_h, image_w = 240, 320
    imu_seq_len = 3
    batch_size = 16
    
    # 1. Instantiate Model
    model = LinearModel(input_shape=(3, image_h, image_w), imu_shape=(imu_seq_len, 6)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()

    # 2. Generate Synthetic Data (Random noise)
    # Vision Input: (B, 3, H, W) | IMU Input: (B, Seq, 6)
    dummy_input = torch.randn(batch_size, 3, image_h, image_w).to(device)
    dummy_imu = torch.randn(batch_size, imu_seq_len, 6).to(device)
    dummy_prev_ctrl = torch.randn(batch_size, 4).to(device)
    # Labels: (B, 4) -> [yaw, pitch, roll, altitude]
    dummy_labels = torch.randn(batch_size, 4).to(device)

    # 3. Training Loop Smoke Test
    print(f"Performing 10 training iterations...")
    model.train()
    start_time = time.time()
    
    for i in range(10):
        optimizer.zero_grad()
        outputs = model(dummy_input, dummy_imu, dummy_prev_ctrl)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()
        print(f"  Iteration {i+1}/10 | Loss: {loss.item():.6f}")

    end_time = time.time()
    print(f"Done! 10 iterations took {end_time - start_time:.2f}s")
    print("Model architecture is valid and trainable.")

if __name__ == "__main__":
    run_smoke_test()
