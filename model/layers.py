import torch
import torch.nn.functional as F
from torch import nn


class PPM(nn.Module):
    def __init__(self, in_channels):
        super(PPM, self).__init__()
        self.features = []
        out_channels = in_channels // 4
        for bin in (1, 2, 3, 6):
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels, affine=True),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True),
                )
            )
        self.features = nn.ModuleList(self.features)
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True))
        out = self.conv(torch.cat(out, 1))
        return out


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(ASPPModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self._init_weight()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class ASPP(nn.Module):
    def __init__(self, in_channels, dilation):
        super(ASPP, self).__init__()
        out_channels = in_channels // 4
        dilations = [1, 3 * dilation, 6 * dilation, 9 * dilation]
        self.aspp1 = ASPPModule(in_channels, out_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = ASPPModule(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = ASPPModule(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = ASPPModule(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3])

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        return out


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
        inc = skip_channels + out_channels
        if self.dec_interp:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        else:
            self.conv_tranpose = ConvTranspose(in_channels, out_channels)

        self.conv_block = ConvBlock(inc, out_channels)
        if skip_channels > 0 and self.attention:
            att_out = out_channels // 2
            self.conv_o = AttentionLayer(out_channels, att_out)
            self.conv_s = AttentionLayer(skip_channels, att_out)
            self.psi = AttentionLayer(att_out, 1)
            self.sigmoid = nn.Sigmoid()
            self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, skip):
        if self.dec_interp:
            out = F.interpolate(self.conv(inputs), scale_factor=2, mode="bilinear", align_corners=True)
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
