import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT

        model = resnet18(weights=weights)

        # Remove final classification layer
        self.features = nn.Sequential(
            *list(model.children())[:-1]
        )

        # Freeze ResNet
        for param in self.features.parameters():
            param.requires_grad = False

        self.eval()

    def forward(self, x):

        with torch.no_grad():

            x = self.features(x)

            x = torch.flatten(x, 1)

        return x


def get_feature_extractor():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNetFeatureExtractor()

    model = model.to(device)

    return model, device