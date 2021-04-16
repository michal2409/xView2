import torch
import torch.nn.functional as F
from torch import nn


class PPM(nn.Module):
    def __init__(self, in_channels, bins, out_channels):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True))
        return torch.cat(out, 1)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels):
        super(ASPP, self).__init__()
        modules = []
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        self.convs = nn.ModuleList(modules)

    def forward(self, x):
        res = [x]
        for conv in self.convs:
            res.append(conv(x))
        return torch.cat(res, dim=1)


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        return out


class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)

    def forward(self, inputs):
        return self.conv(inputs)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.batch_norm(out)
        out = self.lrelu(out)
        return out


class FusionBlock(nn.Module):
    def __init__(self, pre_conv, post_conv, channels):
        super(FusionBlock, self).__init__()
        self.pre_conv = pre_conv
        self.post_conv = post_conv
        self.conv_pre = ConvLayer(2 * channels, channels)
        self.conv_post = ConvLayer(2 * channels, channels)

    def forward(self, pre, post, dec_pre=None, dec_post=None, last_dec=False):
        pre = self.pre_conv(pre, dec_pre) if dec_pre is not None or last_dec else self.pre_conv(pre)
        post = self.post_conv(post, dec_post) if dec_post is not None or last_dec else self.post_conv(post)
        fmap = torch.cat([pre, post], 1)
        pre, post = self.conv_pre(fmap), self.conv_post(fmap)
        return pre, post


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, attention, dec_interp):
        super(UpsampleBlock, self).__init__()
        self.attention = attention
        self.dec_interp = dec_interp
        self.skip_channels = skip_channels
        if self.dec_interp:
            inc = skip_channels + in_channels
        else:
            self.conv_tranpose = ConvTranspose(in_channels, out_channels)
            inc = skip_channels + out_channels

        self.conv_block = ConvBlock(inc, out_channels)
        if skip_channels > 0 and self.attention:
            att_out = out_channels // 2
            self.conv_o = AttentionLayer(in_channels if self.dec_interp else out_channels, att_out)
            self.conv_s = AttentionLayer(skip_channels, att_out)
            self.psi = AttentionLayer(att_out, 1)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, skip):
        if self.dec_interp:
            out = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            out = self.conv_tranpose(inputs)

        if self.skip_channels == 0:
            return self.conv_block(out)

        if self.attention:
            out_a = self.conv_o(out)
            skip_a = self.conv_s(skip)
            psi_a = self.psi(self.relu(out_a + skip_a))
            attention = self.sigmoid(psi_a)
            skip = skip * attention
        out = self.conv_block(torch.cat((out, skip), dim=1))
        return out


class OutputBlock(nn.Module):
    def __init__(self, in_channels, nclass, interpolate):
        super(OutputBlock, self).__init__()
        self.interpolate = interpolate
        self.coral_loss = nclass == 3
        if self.coral_loss:
            self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
            self.bias = nn.Parameter(torch.tensor([[[1.0]], [[0.0]], [[-1.0]]]))
        else:
            self.conv = nn.Conv2d(in_channels, nclass, kernel_size=1)

    def forward(self, inputs):
        out = self.conv(inputs)
        if self.coral_loss:
            out = out + self.bias
        if self.interpolate:
            size = (512, 512) if self.training else (1024, 1024)
            out = F.interpolate(out, size, mode="bilinear", align_corners=True)
        return out
