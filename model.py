import torch.nn as nn
import timm

class MultiZooViT(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='vit_base_patch16_224'):
        """ Vision Transformer tabanlı model """
        super(MultiZooViT, self).__init__()
        
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)

class MultiZooSwinTransformer(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='swin_base_patch4_window7_224'):
        """ Swin Transformer tabanlı model """
        super(MultiZooSwinTransformer, self).__init__()
        
        self.swin = timm.create_model(model_name, pretrained=pretrained)
        
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.swin(x)

class MultiZooDeiT(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_name='deit_base_patch16_224'):
        """ DeiT (Data-efficient image Transformer) tabanlı model """
        super(MultiZooDeiT, self).__init__()
        
        self.deit = timm.create_model(model_name, pretrained=pretrained)
        
        in_features = self.deit.head.in_features
        self.deit.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.deit(x)

def get_model(model_type, num_classes, pretrained=True):
    if model_type == 'vit':
        return MultiZooViT(num_classes, pretrained)
    elif model_type == 'swin':
        return MultiZooSwinTransformer(num_classes, pretrained)
    elif model_type == 'deit':
        return MultiZooDeiT(num_classes, pretrained)
    else:
        raise ValueError(f"Bilinmeyen model türü: {model_type}. 'vit', 'swin' veya 'deit' kullanın.") 