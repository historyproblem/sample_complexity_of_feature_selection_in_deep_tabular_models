"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name : str, input_shape : int, num_classes : int, dropout_rate : float, FC_size : int):
        super(VGG, self).__init__()
        self.input_shape = input_shape
        self.features = self._make_layers(cfg[vgg_name])
        
        output_size = input_shape[-1]
        last_num_channels = 0
        for e in cfg[vgg_name]:
            if e == 'M':
                output_size = output_size // 2
            else:
                last_num_channels = e
        output_size = (output_size**2)*last_num_channels
        
        self.classifier = nn.Sequential(
            nn.Linear(output_size, FC_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(FC_size, FC_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(FC_size, num_classes)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_shape[0]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG16', 10)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()