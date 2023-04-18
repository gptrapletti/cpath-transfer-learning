import pytorch_lightning as pl
import torch.nn as nn
from src.utils import load_pretrained_model

class SegEncoder(nn.Module):
    """
    Pretrained ResNet to use as encoder of the backbone.
    
    pruning: how many layers remove from the end of the model.
    """
    def __init__(self, ckpt_path, pruning=3, device='cuda:0'):
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
    
    
class SegDecoder(nn.Module):
    pass
        
        




class SegResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        pass
