import torch
import torch.nn as nn
from torchvision import models


def _load_densenet121(pretrained=True):
    """Load DenseNet-121, compatible with both old and new torchvision APIs."""
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        return models.densenet121(weights=weights)
    except AttributeError:
        return models.densenet121(pretrained=pretrained)


class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()

        densenet = _load_densenet121(pretrained=pretrained)
        self.features = densenet.features          
        self.gap = nn.AdaptiveAvgPool2d((1, 1))    

        # clinical branch: age + gender -> 16-dim embedding
        self.clinical_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # fuse image (1024) + clinical (16) -> 14 logits
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, age, gender):
        x = self.features(image)
        x = torch.relu(x)                  
        x = self.gap(x)
        x = torch.flatten(x, 1)           

        clinical = torch.cat([age, gender], dim=1)  
        clinical = self.clinical_branch(clinical)   

        combined = torch.cat([x, clinical], dim=1)   
        return self.classifier(combined)             


if __name__ == '__main__':
    model = ChestXrayModel(num_classes=14, pretrained=False).eval()
    out = model(torch.randn(4, 3, 224, 224), torch.randn(4, 1), torch.randn(4, 1))
    print("Output shape:", out.shape)   
