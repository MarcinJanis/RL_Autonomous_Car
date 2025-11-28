import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision import transforms


class imageProcessingModule(nn.Module):
  '''
  Image processing module is based on pretrained unet, finetuned to segmentation task.
  It distinguishes n = 8 classes, eg. background, walls, road, road lines, ect.
  '''
  def __init__(self, check_point_pth, n_classes, device = 'cuda'):
    super().__init__()
    print("\n[Init] Initialization of image processing module:")

    #  init full endoer model 
    model_full = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_classes
        )
    #   load pre-trained weights
    checkpoint = torch.load(check_point_pth, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
      if k.startswith('model.'):
        new_state_dict[k[len('model.'):]] = v
      else:
        new_state_dict[k] = v
    model_full.load_state_dict(new_state_dict)
    print("\nUnet weights imported.")
    # Extract encoder only
    self.feature_extractor = model_full.encoder
    self.feature_extractor.to(device)
    self.feature_extractor.eval()

    # Transform to provide correct data type for Unet
    self.transform = transforms.Compose([
                     transforms.Resize((256, 256)),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ])
    print("\nUnet encoder created.")

  def forward(self, x):
    x = self.feature_extractor(x)
    return x
  
  def freeze_encoder(self):
    for param in self.feature_extractor.parameters():
      param.requires_grad = False

  def unfreeze_encoder(self, first_layer = 3):
    # first_layer - number of first layer that shall be unfreeze
    # first_layer shall be less than total number of encoder layeres, that is 5 (in ResNet18)
    for idx, block in enumerate(self.feature_extractor.blocks):
      if idx > first_layer:
        for param in block.parameters():
          param.requires_grad = True
      else:
        for param in block.parameters():
          param.requires_grad = False


class lidarProcessingModule(nn.Module):
  def __init__(self, angle_min, angle_max, r_max, n_beans, grid_shape):
      super().__init__()

      # -- lidar parameters --
      self.angle_min = angle_min 
      self.angle_max = angle_max
      self.r_max = r_max
      self.grid_shape = grid_shape 
      n = n_beans
      self.angle_tensor = torch.linspace(angle_min, angle_max, n)
      self.scale_factor = grid_shape / r_max

  def forward(self, x):
    x = self.LidarBEV(x)

    pass 

  def LidarBEV(self, vct_in, device = 'cuda'):
    m_tensor = vct_in.to(device)
    # Tranform to cartesian and scale coords
    x_coords = (self.scale_factor*m_tensor*torch.sin(self.angle_tensor)).int() + self.grid_shape
    y_coords = (self.scale_factor*m_tensor*torch.cos(self.angle_tensor)).int()
    # init BEV map   
    map = torch.zeros((self.grid_shape, 2*self.grid_shape, 1), dtype = torch.float32, device = device)
    # fill points from lidar
    map[y_coords, x_coords, 0] = 1.0
    return map






class agent_model():
  def __init__(self, ...):
    super().__init__()

  def forward(self, ...):
    pass
    
 def act(self, x)
    logits = self.forward(x)

