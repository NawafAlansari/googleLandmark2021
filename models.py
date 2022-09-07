import torch.nn as nn 
import timm 

class LandmarkRetrievalModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True) -> None:
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, num_classes)

    def forward(self, x):
        features = self.model(x)
        output = self.fc(features)
        return output 

    def extract_features(self, x):
        features = self.model(x)
        return features