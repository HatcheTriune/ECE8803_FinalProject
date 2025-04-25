import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.img_out_dim = 2048
        self.extra_mlp = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(self.img_out_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 6)
        )

    def forward(self, image, extra_features):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        img_feat = self.cnn(image).view(image.size(0), -1)
        extra_feat = self.extra_mlp(extra_features)
        combined = torch.cat((img_feat, extra_feat), dim=1)
        return self.fusion(combined)