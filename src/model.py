import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from src.utils import load_pretrained_model

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
    def __init__(self, ckpt_path, pruning): # device
        super().__init__()
        self.ckpt_path = ckpt_path
        # self.device = device
        self.pruning = pruning
        self.model = self.prepare_model()
        
    def prepare_model(self):
        model = load_pretrained_model(path=self.ckpt_path) # device=self.device
        model = nn.Sequential(*list(model.children())[:-self.pruning])
        return model
    
    def forward(self, x):
        output = self.model(x)
        return output
    
    

class BasicDecoder(nn.Module):
    def __init__(self, channel_progression):
        super().__init__()
        self.channel_progression = channel_progression
        self.backbone = nn.ModuleList([UpBlock(c) for c in channel_progression])
        self.final_conv = nn.Conv2d(in_channels=channel_progression[-1]//2, out_channels=1, kernel_size=1, stride=1) # last UpBlock returns "channel_progression[-1]//2" channels
        self.sigmoid = nn.Sigmoid() # maybe softmax?
        
    def forward(self, x):     
        for block in self.backbone: # ModuleList is just a list so a forward for each block must be defined
            x = block(x)
        x = self.final_conv(x)
        x = self.sigmoid(x)
        return x

        
       
class ProDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        pass



class SegResNet(pl.LightningModule):
    def __init__(self, ckpt_path, pruning, lr):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.pruning = pruning
        self.loss_fn = nn.BCELoss()
        # self.loss_fn = monai.losses.DiceLoss(sigmoid=False)
        self.metric = torchmetrics.Dice(threshold=0.5, ignore_index=0)
        self.lr = lr
        # self.scheduler_params = scheduler_params   # DELENDUM
        # Instantiate encoder
        self.encoder = Encoder(ckpt_path=self.ckpt_path, pruning=self.pruning)
        # # Freeze encoder
        # self.freeze_encoder()
        # self.reinitialize_parameters()
        # Find encoder output shape, using a dummy input
        self.encoder_output_shape = list(self.encoder(torch.rand([1, 3, 256, 256])).shape)
        # Find channel progression
        self.channel_progression = self.find_channel_progression(
            initial_size=self.encoder_output_shape[2],
            initial_channels=self.encoder_output_shape[1]
        )
        # Instantiate decoder
        self.decoder = BasicDecoder(self.channel_progression)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        
    def find_channel_progression(self, initial_size, initial_channels):
        '''Find the number of channels in each decoder step. This 
        number of channels is computed based upon the number of channels
        of the output of the encoder and upon their dimensions (HxW). Since
        each decoder step doubles the dimensions, it must be found how many
        times (that is, how many steps) this has to be done to go from the encoder
        output to dimensions to the desidered output dimensions (256x256). The
        channels progression is computed on the basis of this number of steps.
        For example, if the encoder output has shape [128, 32, 32], n steps
        are required to reach shape [C, 256, 256] and the number of channels
        will be halved at each step, being C the final number. 
        
        '''
        # Find number of blocks needed to reach dimension 256x256.
        size = initial_size
        size_progression = []
        while size <= 256:
            size_progression.append(size)
            size = size*2
        n_blocks = len(size_progression)-1
        # "-1" because these are the input sizes of the convolutions
        # and the last conv doubles it again (without "-1" initial size of 256
        # would out a mask 512x512)
        
        # Find progression of number of channels for the BasicDecoder.    
        channels = initial_channels
        channel_progression = [] # it should contain also the number of channels of the initial feature maps, to be used for the first "in_channels" parameter.
        for i in range(n_blocks):
            channel_progression.append(channels)
            channels = int(channels / 2)
        
        return channel_progression
    
    def training_step(self, batch, batch_idx):
        images, gts = batch
        gts = gts[:, None, :, :] # add channel dimension
        preds = self(images)
        # loss = self.loss_fn(preds, gts) # for the DiceLoss ### !
        loss = self.loss_fn(preds.float(), gts.float()) # for CE loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, gts = batch
        gts = gts[:, None, :, :] # add channel dimension
        preds = self(images)
        # loss = self.loss_fn(preds, gts) # for the DiceLoss ### !
        loss = self.loss_fn(preds.float(), gts.float()) # for CE loss
        metric = self.metric(preds, gts) # does 0.5 thresholding
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_metric', metric, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        images, gts = batch
        gts = gts[:, None, :, :] # add channel dimension
        preds = self(images)
        metric = self.metric(preds, gts)
        self.log('test_metric', metric, prog_bar=True)
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.01, 
            threshold_mode='abs'
        )
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}
    
    # def freeze_encoder(self):
    #     for param in self.encoder.parameters():
    #         param.requires_grad = False
    #     print('\nFREEZED!\n')
    
    # def reinitialize_parameters(self):
    #     for layer in self.encoder.children():
    #         if hasattr(layer, 'reset_parameters'):
    #             layer.reset_parameters()
    #     print('\nREINITIALIZED!\n')
 
