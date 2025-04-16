import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """
    Image encoder based on ResNet50
    """

    def __init__(self, embed_size=512):
        super(ImageEncoder, self).__init__()

        # Load pre-trained ResNet and remove the final FC layer
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.cnn = nn.Sequential(*modules)

        # Linear layer to convert image features to embedding
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.relu = nn.ReLU()

        # Initialize weights
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """
        Extract image features and convert to embedding
        """
        with torch.no_grad():
            features = self.cnn(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        features = self.relu(features)

        return features