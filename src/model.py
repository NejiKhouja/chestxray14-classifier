import torch
import torch.nn as nn
from torchvision import models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(ChestXrayModel, self).__init__()

        # load pretrained DenseNet-121
        densenet = models.densenet121(pretrained=pretrained)

        # extract the feature extractor (everything except the final classifier)
        self.features = densenet.features

        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # clinical branch (age + gender = 2 inputs)
        self.clinical_branch = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # final classifier (1024 image features + 16 clinical features)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, age, gender):
        # image branch
        x = self.features(image)
        x = self.gap(x)
        x = torch.flatten(x, 1)  # (batch, 1024)

        # clinical branch
        clinical = torch.cat([age, gender], dim=1)  # (batch, 2)
        clinical = self.clinical_branch(clinical)    # (batch, 16)

        # combine
        combined = torch.cat([x, clinical], dim=1)  # (batch, 1040)
        out = self.classifier(combined)              # (batch, 15)

        return out


if __name__ == '__main__':
    model = ChestXrayModel(num_classes=15, pretrained=True)
    model.eval()

    # dummy inputs to test shapes
    dummy_image = torch.randn(4, 3, 224, 224)   # batch of 4 images
    dummy_age = torch.randn(4, 1)
    dummy_gender = torch.randn(4, 1)

    output = model(dummy_image, dummy_age, dummy_gender)
    print(f"Output shape: {output.shape}")       # should be (4, 15)
    print("Model loaded successfully")