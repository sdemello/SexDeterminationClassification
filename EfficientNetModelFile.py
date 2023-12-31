import efficientnet.keras as efn
from torch import nn
from torchvision import models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import WeightsEnum
from torch.hub import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet


def EfficientNetModelImport():
    '''
    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    WeightsEnum.get_state_dict = get_state_dict
    # model = efn.EfficientNetB0(weights='imagenet')
    #model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    '''
    model = EfficientNet.from_pretrained('efficientnet-b1')
    '''
    # Freeze Layers
    for param in model.parameters():
        param.requires_grad = False

    model._fc = nn.Linear(in_features=model._fc.in_features, out_features=2)
    '''

    return model


