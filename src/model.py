import pytorch_lightning as pl
import torch.nn as nn
from src.utils import load_pretrained_model

# class UpStep(nn.Module):
#     def __init__(self, input_shape):
#         self.input_shape = input_shape
#         self.stack = nn.Sequential([
#             # Halves the number of channels of the feature maps while
#             # keeping weight and width the same
#             nn.ConvTranspose2d(
#                 in_channels=self.input_shape, 
#                 out_channels=self.input_shape/2, 
#                 kernel_size=2, 
#                 stride=2, 
#                 padding=0
#             ),
#             #
#             nn.Conv2d(
#                 in_channels=output1.shape[0],
#                 out_channels=int(output1.shape[0]/2),
#                 kernel_size=3,
#                 stride=1,
#                 padding=1
#             )
            
#             # COME PASSARE A QUESTO ULTIMO LAYER IL NUOVO INPUT SHAPE, RISULTANTE DAL PASSAGGIO PRECEDENTE? FARLO NELLA FORWARD? 
#             # COME INSERIRE IL CONTROLLO DI FERMARSI QUANDO SI E' ARRIVATI A 256? NELLA CLASSE DECODER DIREI.
            
#         ])
        
#     def forward(self, x):
#         return self.upconv(x)

class UpBlock(nn.Module):
    '''
    Does upsampling to double size and convolutions to halve the number of channels.
    '''
    def __init__(self, in_channels):
        super().__init__()
        self.out_channels = in_channels//2
        self.block = nn.Sequential(
            # Up-conv 2x2: doubles the size and halves the channels.
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=2, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.out_channels),
            # Conv 3x3 n.1: keeps both size and channels.
            # (in the UNet paper this conv too halves the channels because the skip connections doubles them again.
            # However here I do not use skip connections so this conv keeps the same number of channels
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.out_channels),
            # Conv 3x3 n.2: same as before
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        
    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """
    Pretrained ResNet to use as encoder of the backbone.
    
    pruning: how many layers remove from the end of the model.
    """
    def __init__(self, ckpt_path, pruning, device):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device
        self.pruning = pruning
        self.model = self.prepare_model()
        
    def prepare_model(self):
        model = load_pretrained_model(path=self.ckpt_path, device=self.device)
        model = nn.Sequential(*list(model.children())[:-self.pruning])
        return model
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    

class BasicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_channels = [256, 128, 64, 32, 16]
        self.backbone = nn.Sequential(
            nn.ModuleList([UpBlock(c) for c in n_channels]),
            nn.Conv2d(in_channels=n_channels[-1], out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):     
        x = self.backbone(x)
        return x
    
        
       

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass



class SegResNet(pl.LightningModule):
    def __init__(self, ckpt_path, pruning=3, device='cuda:0'):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.pruning = pruning
        self.device = device
        self.encoder = Encoder(ckpt_path=self.ckpt_path, pruning=self.pruning, device=self.device)
        self.decoder = BasicDecoder()
        
    def forward(self, x):
        y = self.encoder(x)
        
        y = self.decoder()
        





        
# class Decoder(nn.Module):
#     def __init__(self, input_shape, hidden_dim):
#         super(Decoder, self).__init__()
        
#         # Define the input shape of the decoder
#         self.input_shape = input_shape
        
#         # Define the hidden dimension of the decoder
#         self.hidden_dim = hidden_dim
        
#         # Define the convolutional layers
#         self.conv1 = nn.ConvTranspose2d(input_shape[1], hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.conv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim//2, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.conv3 = nn.ConvTranspose2d(hidden_dim//2, hidden_dim//4, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.conv4 = nn.ConvTranspose2d(hidden_dim//4, 1, kernel_size=3, stride=1, padding=1)
        
#         # Define the activation functions
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         # Pass the input through the convolutional layers and activation functions
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.relu(self.conv3(out))
#         out = self.sigmoid(self.conv4(out))
        
#         # Resize the output to match the input size
#         out = nn.functional.interpolate(out, size=self.input_shape[2:], mode='bilinear', align_corners=False)
        
#         # Return the output
#         return out
