import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models import resnet18
from torchvision import transforms

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor 

class imageProcessingModule(nn.Module):
  '''
  Image processing module is based on pretrained unet, finetuned to segmentation task.
  It distinguishes n = 8 classes, eg. background, walls, road, road lines, ect.

  Input: RGB Image (B, C, H, W): C = 3, H = 256, W =256
  Output: Features map: (B, C, H, W): C = 512, H = 8, W = 8
  '''
  def __init__(self, encoder_check_point_pth , n_classes = 8, device = 'cuda'):
    super().__init__()
    
    #  init full endoer model 
    model_full = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )
    
    #   load pre-trained weights
    checkpoint = torch.load(encoder_check_point_pth, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
      if k.startswith('model.'):
        new_state_dict[k[len('model.'):]] = v
      else:
        new_state_dict[k] = v

    model_full.load_state_dict(new_state_dict)

    # Extract encoder only
    self.feature_extractor = model_full.encoder

  def forward(self, x):
    x = self.feature_extractor(x)
    return x[-1]
  
  # def freeze_encoder(self):
  #   for param in self.feature_extractor.parameters():
  #     param.requires_grad = False

  # def freeze_encoder(self, unfreezed_layers = 3):
  #   # first_layer - number of first layer that shall be unfreeze
  #   # first_layer shall be less than total number of encoder layeres, that is 5 (in ResNet18)
  #   for idx, block in enumerate(self.feature_extractor.blocks):
  #     if idx > unfreezed_layers:
  #       for param in block.parameters():
  #         param.requires_grad = True
  #     else:
  #       for param in block.parameters():
  #         param.requires_grad = False

  def freeze_encoder(self, unfreezed_layers=3):
      # get child items
      layers = list(self.feature_extractor.children())
      total_layers = len(layers)
      
      # Freeze all
      for param in self.feature_extractor.parameters():
          param.requires_grad = False
      # Unfreeze a few last layers
      for i in range(1, unfreezed_layers + 1):
          if total_layers - i >= 0:
              for param in layers[total_layers - i].parameters():
                  param.requires_grad = True




class lidarProcessingModule(nn.Module):
  def __init__(self):
    super().__init__()

    self.ConvBlock = nn.Sequential(
      nn.Conv1d(1, 4, kernel_size=(3), padding = 1), # (B, C, L): (B, 1, 280) -> (B, 4, 280)
      nn.BatchNorm1d(4),
      nn.ReLU(),
      nn.MaxPool1d(2), # (B, C, L): (B, 4, 280) -> (B, 4, 140)
      nn.Conv1d(4, 16, kernel_size=(3), padding = 1), # (B, C, L): (B, 4, 140) -> (B, 16, 140) 
      nn.BatchNorm1d(16),
      nn.ReLU(),
      nn.MaxPool1d(2), # (B, C, L): (B, 16, 140) -> (B, 16, 70) 
      nn.Conv1d(16, 32, kernel_size=(3), padding = 1), # (B, C, L): (B, 16, 70) -> (B, 32, 70) 
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.MaxPool1d(2), # (B, C, L): (B, 32, 70) -> (B, 32, 35) 
    )

    self.Linear = nn.Linear(32*35, 512)


  def forward(self, x):
    '''
    Input: lidar vector, shape: (B, L), L = 280
    Output: features vector, shape: (B, L), L = 512 
    '''
    x = x.unsqueeze(1) # (B, L) -> (B, C, L), C = 1
    x = self.ConvBlock(x) # (B, C, L): (B, 1, 280) -> (B, 32, 35) 
    x = torch.flatten(x, start_dim=1) # (B, C, L):  (B, 32, 35) -> (B, 1120)
    x = self.Linear(x) # (B, 1120) -> (B, 512)
    return x 


class AgentNet(BaseFeaturesExtractor):
  def __init__(self, observation_space, encoder_check_point_pth, device = 'cuda', features_dim = 1024):
    super().__init__(observation_space, features_dim)

    self.encoder_RGB = imageProcessingModule(encoder_check_point_pth, device = device) # Vector 
    self.encoder_RGB.freeze_encoder(unfreezed_layers = 3)

    self.encoder_Lidar = lidarProcessingModule() # Vector: 512

    self.AvgPoolRGB = nn.AdaptiveAvgPool2d((1, 1)) # 8x8x512 -> 1x1x512

    # self.agent_head = nn.Sequential(nn.Linear(512*2, 128),
    #                                 nn.ReLU(),
    #                                 nn.Dropout(p=0.5),
    #                                 nn.Linear(128, 64),
    #                                 nn.ReLU(),
    #                                 nn.Dropout(p=0.5),
    #                                 nn.Linear(64, action))
    
    # self.critic_head = nn.Sequential(nn.Linear(512*2, 128),
    #                                  nn.ReLU(),
    #                                  nn.Dropout(p=0.5),
    #                                  nn.Linear(128, 64),
    #                                  nn.ReLU(),
    #                                  nn.Dropout(p=0.5),
    #                                  nn.Linear(64, 1))


  def forward(self, observations):

    # get data
    img = observations['image'] # (B, H, W, C)
    lidar = observations['lidar'] 

    if img.shape[-1] == 3:
      img = img.permute(0, 3, 1, 2)


    img = img.float() / 255.0

    # Image branch
    img_features = self.encoder_RGB(img)
    img_features = self.AvgPoolRGB(img_features) # (B, C=512, H=8, W=8) -> (B, C=512, H=1, W=1)
    img_features = torch.flatten(img_features, start_dim=1) # (B, C, 1, 1) -> (B, C=512)

    # Lidar branch 
    lidar_features = self.encoder_Lidar(lidar) # (B, C=512)

    # Concatenate
    features = torch.cat((img_features, lidar_features), dim = 1) # (B, C1=512) + (B, C2=512) -> (B, C1+C2=1024) 

    # Decision head
    # x = self.agent_head(features) 
    # returns: x = (x1, x2), where:   
    # -> x1 -> linear velocity in x direction
    # -> x2 -> angular velocity around the z axis 

    # Critic head
    # v = self.critic_head(features) 
    # returns: v - predicted prize value

    # return x, v
    return features # shape: (B, 1024)
    


# class AgentDeployModel:
#   def __init__(self, device):

#     model = AgentNet(device)



#   def LidarBEV(self, vct_in):

#     # args: 
#     # return: map (C, H, W)

#     # m_tensor = vct_in.to(device)
#     # Tranform to cartesian and scale coords
#     x_coords = (self.scale_factor*vct_in*torch.sin(self.angle_tensor)).int() + self.grid_shape
#     y_coords = (self.scale_factor*vct_in*torch.cos(self.angle_tensor)).int()
#     # init BEV map   
#     map = torch.zeros((self.grid_shape, 2*self.grid_shape, 1), dtype = torch.float32, device = device)
#     # fill points from lidar
#     map[y_coords, x_coords, 0] = 1.0
#     map.unsqueeze(0)

#     return map