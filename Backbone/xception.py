# _*_ coding: utf-8 _*_
"""
    Author: Kwong
    Create time: 2020/10/27 13:22 
"""

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pretrainedmodels.models.xception import pretrained_settings
from pretrainedmodels.models.xception import Xception


"""
    example:
    model = get_xception("xception")
    output:
    [[b, 128, 1/4, 1/4],
     [b, 256, 1/8, 1/8],
     [b, 728, 1/16, 1/16],
     [b, 2048, 1/32, 1/32]]
"""

class XceptionEncoder(Xception):

    def __init__(self, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = 4
        self._in_channels = 3

        # modify padding to maintain output shape
        self.conv1.padding = (1, 1)
        self.conv2.padding = (1, 1)

        del self.fc

    def get_stages(self):
        return [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu, self.block1),
            self.block2,
            nn.Sequential(self.block3, self.block4, self.block5, self.block6, self.block7,
                          self.block8, self.block9, self.block10, self.block11),
            nn.Sequential(self.block12, self.conv3, self.bn3, self.relu, self.conv4, self.bn4),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        # state_dict.pop('fc.bias')
        # state_dict.pop('fc.weight')

        super().load_state_dict(state_dict)


xception_encoders = {
    'xception': {
        'encoder': XceptionEncoder,
        'pretrained_settings': pretrained_settings['xception'],
        'params': {
            'out_channels': (3, 64, 128, 256, 728, 2048),
        }
    },
}


def get_xception(name, weights="imagenet"):
    Encoder = xception_encoders[name]["encoder"]
    params = xception_encoders[name]["params"]
    encoder = Encoder(out_channels=params)
    if weights is not None:
        settings = xception_encoders[name]["pretrained_settings"][weights]
        pretrained_dict = model_zoo.load_url(settings["url"])
        model_dict = encoder.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)
    return encoder


if __name__ == "__main__":
    import torch
    model = get_xception("xception")
    # print(model)
    img = torch.rand((5, 3, 256, 256))
    for i in model(img):
        print(i.shape)