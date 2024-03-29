"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):    
    def __init__(self, n_classes=101, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(2*hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features1 = self.resnet(x[:, 0, :,:, :])
        features2 = self.resnet(x[:, 1, :,:, :])
        features = torch.cat([features1, features2], dim=1)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs, features
    
class ResNet50_base(nn.Module):   
    """ResNet50 but without the final fc layer"""
    
    def __init__(self, n_classes=101, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))
        outputs = self.fc(features)
        return outputs, features

