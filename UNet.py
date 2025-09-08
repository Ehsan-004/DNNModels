import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=True)):
        super(DoubleConv, self).__init__()

        layers = []
        # Conv → Norm → Act (first layer)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation is not None:
            layers.append(activation)

        # Conv → Norm → Act (second layer)
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation is not None:
            # create a new object of activation class (not to share the layer)
            layers.append(type(activation)())

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=False)):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, norm_layer, activation)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 up_mode='upconv', 
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.ReLU(inplace=False)):
        super(Up, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            # devided by 2 because of concatenate
        elif up_mode == 'interp':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1)
            )
        else:
            raise ValueError("up_mode should be 'upconv' or 'interp'")

        self.conv = DoubleConv(in_channels, out_channels, norm_layer, activation)

    def forward(self, x1, x2):
        # x1 = input from previous layer (low-res)
        # x2 = skip connection from encoder (high-res)
        x1 = self.up(x1)

        # if spatial sizes are not the same, add padding:
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # [pad_left, pad_right, pad_top, pad_bottom]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)  # concat channels
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

