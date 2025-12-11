import net_v1
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = net_v1.AgentNet(device = device)

model.to(device)

# --- Example data ---
Img_example = np.zeros((255, 255, 3), dtype = np.uint8) # np array 
Lidar_example = np.zeros((280), dtype = np.float32) # np array 

# to tensor
img_tensor = torch.from_numpy(Img_example).float() / 255.0 # toTensor + normalization to (0, 1)
img_tensor = img_tensor.permute(2, 0, 1) # correct channels 
lidar_tensor = torch.from_numpy(Lidar_example).float() # toTensor + normalization (?)

# add batch size and transport to device
img_tensor = img_tensor.unsqueeze(0)
lidar_tensor = lidar_tensor.unsqueeze(0)

img_tensor = img_tensor.to(device)
lidar_tensor = lidar_tensor.to(device)

print(f'Image shape: {img_tensor.shape}')
print(f'Image shape: {lidar_tensor.shape}')


# --- How to use during evaluation ---
print('--- Evaluation ---')
model.eval()
with torch.no_grad():
    logits_x, logits_v = model(img_tensor, lidar_tensor)

print(f'\nOutput shape: x: {logits_x.shape}, {logits_v.shape}')
print(f'\nOutput values: x: {logits_x}, {logits_v}')


# --- How to use durgin training
model.train()
model.encoder_RGB.unfreeze_encode(first_layers = 3) # freez encoder, only 3 last layers leave unfreezed 


