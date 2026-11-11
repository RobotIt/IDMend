import torch
from torch import nn
from torchvision.models import resnet18
import torch.nn.functional as F

class ResNet18Link(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.encoder=resnet18(weights=None)
        if pretrained_path:
            self.encoder.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True))
        self.encoder.fc=nn.Identity()
        self.classifier=nn.Linear(512*2+20,1)
        self.register_buffer('feature_mean',torch.zeros(20))
        self.register_buffer('feature_std',torch.ones(20))
        self.register_buffer('rgb_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([.229,.224,.225]).view(1,3,1,1))

    def train(self,mode=True):
        super().train(mode)
        for m in self.encoder.modules():
            if isinstance(m,nn.BatchNorm2d): m.eval()
        return self

    def encode(self,pixels):
        x=pixels.float().permute(0,3,1,2)/255.
        return F.normalize(self.encoder((x-self.rgb_mean)/self.rgb_std),dim=1)

    def pair_logits(self,a,b,raw):
        x=torch.cat([torch.abs(a-b),a*b,(raw-self.feature_mean)/self.feature_std],dim=1)
        return self.classifier(x).squeeze(1)

    def forward(self,left,right,raw):
        e=self.encode(torch.cat([left,right],dim=0))
        a,b=e.chunk(2)
        return self.pair_logits(a,b,raw)
