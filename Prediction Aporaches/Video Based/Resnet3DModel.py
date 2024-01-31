import torchvision
import torch.nn as nn
class my3DResNet(nn.Module):
    def __init__(self,numOut):
        super(my3DResNet, self).__init__()
        #self.model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
        #self.model = torchvision.models.video.mc3_18(pretrained=True, progress=True)
        self.model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, numOut)

    def forward(self, x):
        return self.model.forward(x)
