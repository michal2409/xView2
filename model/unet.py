import torch
import torch.nn.functional as F
import torchvision.models as models
from resnest.torch import resnest50, resnest101, resnest200, resnest269
from torch import nn

from model.layers import ASPP, PPM, FusionBlock, OutputBlock, UpsampleBlock

resnest = {
    "resnest50": resnest50,
    "resnest101": resnest101,
    "resnest200": resnest200,
    "resnest269": resnest269,
}


def concat(x, y):
    return None if x is None or y is None else torch.cat([x, y], 1)


def get_nclass(args):
    if args.loss_str == "mse":
        return 1
    elif args.loss_str == "coral":
        return 3
    return 4


def get_dmg_unet(args):
    dmg_unets = {
        "siamese": SiameseUNet,
        "siameseEnc": SiameseEncUNet,
        "fused": FusedUNet,
        "fusedEnc": FusedEncUNet,
        "parallel": ParallelUNet,
        "parallelEnc": ParallelEncUNet,
        "diff": DiffUNet,
        "cat": CatUNet,
    }
    nclass = get_nclass(args)
    model = dmg_unets[args.dmg_model](args, nclass)
    return model


def get_encoder(encoder_str, dilation, pretrained=True, in_channels=3):
    assert "resnet" in encoder_str or "resnest" in encoder_str

    if "resnest" in encoder_str:
        encoder_channels = [128, 256, 512, 1024, 2048]
        if "50" in encoder_str:
            encoder_channels[0] = 64
        encoder = resnest[encoder_str](pretrained=pretrained, dilation=dilation)
    else:
        encoder_channels = [64, 256, 512, 1024, 2048]
        replace_stride_with_dilation = [False, dilation == 4, dilation in [2, 4]]
        if encoder_str == "resnet50":
            encoder = models.resnet50(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
        elif encoder_str == "resnet101":
            encoder = models.resnet101(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
        elif encoder_str == "resnet152":
            encoder = models.resnet152(pretrained=pretrained, replace_stride_with_dilation=replace_stride_with_dilation)
        else:
            raise f"Not implemented encoder {encoder_str}"

    if in_channels != 3:
        conv1 = encoder.conv1[0] if "st" in encoder else encoder.conv1
        conv1 = torch.nn.Conv2d(
            in_channels,
            conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias,
        )
        if "st" in encoder:
            encoder.conv1[0] = conv1
        else:
            encoder.conv1 = conv1

    encoder_layer1 = nn.Sequential(encoder.conv1, encoder.bn1, nn.ReLU(inplace=True))
    encoder_layer2 = nn.Sequential(encoder.maxpool, encoder.layer1)
    encoder_layer3 = encoder.layer2
    encoder_layer4 = encoder.layer3
    encoder_layer5 = encoder.layer4

    return encoder_channels, encoder_layer1, encoder_layer2, encoder_layer3, encoder_layer4, encoder_layer5


def get_decoder(encf, dilation, attn, no_skip=False, dec_interp=False):
    decf = [512, 256, 128, 64, 32]
    if dilation == 1:
        decoder_layer1 = UpsampleBlock(encf[-1], decf[0], 0 if no_skip else encf[-2], attn, dec_interp)
        decoder_layer2 = UpsampleBlock(decf[0], decf[1], 0 if no_skip else encf[-3], attn, dec_interp)
        decoder_layer3 = UpsampleBlock(decf[1], decf[2], 0 if no_skip else encf[-4], attn, dec_interp)
        decoder_layer4 = UpsampleBlock(decf[2], decf[3], 0 if no_skip else encf[-5], attn, dec_interp)
        decoder_layer5 = UpsampleBlock(decf[3], decf[4], 0, attn, dec_interp)
    elif dilation == 2:
        decoder_layer1 = None
        decoder_layer2 = UpsampleBlock(encf[-1], decf[1], 0 if no_skip else encf[-3], attn, dec_interp)
        decoder_layer3 = UpsampleBlock(decf[1], decf[2], 0 if no_skip else encf[-4], attn, dec_interp)
        decoder_layer4 = UpsampleBlock(decf[2], decf[3], 0 if no_skip else encf[-5], attn, dec_interp)
        decoder_layer5 = UpsampleBlock(decf[3], decf[4], 0, attn, dec_interp)
    elif dilation == 4:
        decoder_layer1, decoder_layer2 = None, None
        decoder_layer3 = UpsampleBlock(encf[-1], decf[2], 0 if no_skip else encf[-4], attn, dec_interp)
        decoder_layer4 = UpsampleBlock(decf[2], decf[3], 0 if no_skip else encf[-5], attn, dec_interp)
        decoder_layer5 = UpsampleBlock(decf[3], decf[4], 0, attn, dec_interp)
    else:
        raise ValueError("Dilation can be set to 1, 2 or 4")
    return decf, decoder_layer1, decoder_layer2, decoder_layer3, decoder_layer4, decoder_layer5


class UNetTemplate(nn.Module):
    def __init__(self, args, in_channels=3):
        super(UNetTemplate, self).__init__()
        self.use_ppm = args.ppm
        self.use_aspp = args.aspp
        self.dilation = args.dilation
        self.no_skip = args.no_skip
        self.interpolate = args.interpolate
        self.enc_chn, self.enc_l1, self.enc_l2, self.enc_l3, self.enc_l4, self.enc_l5 = get_encoder(
            args.encoder, self.dilation, in_channels=in_channels
        )

        if self.use_ppm:
            self.ppm = PPM(self.enc_chn[-1])
        elif self.use_aspp:
            self.aspp = ASPP(self.enc_chn[-1], self.dilation)

        self.dec_chn = None
        if not self.interpolate:
            self.dec_chn, self.dec_l1, self.dec_l2, self.dec_l3, self.dec_l4, self.dec_l5 = get_decoder(
                self.enc_chn, self.dilation, args.attention, self.no_skip, args.dec_interp
            )

    def forward(self, data):
        enc1 = self.enc_l1(data)
        enc2 = self.enc_l2(enc1)
        enc3 = self.enc_l3(enc2)
        enc4 = self.enc_l4(enc3)
        enc5 = self.enc_l5(enc4)

        if self.use_ppm:
            enc5 = self.ppm(enc5)
        elif self.use_aspp:
            enc5 = self.aspp(enc5)
        if self.interpolate:
            return enc5, None, None

        if self.dilation == 1:
            if self.no_skip:
                enc1, enc2, enc3, enc4 = None, None, None, None
            dec1 = self.dec_l1(enc5, enc4)
            dec2 = self.dec_l2(dec1, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 2:
            if self.no_skip:
                enc1, enc2, enc3 = None, None, None
            dec2 = self.dec_l2(enc5, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 4:
            if self.no_skip:
                enc1, enc2 = None, None
            dec3 = self.dec_l3(enc5, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)

        return dec5, dec4, dec3


class OutputTemplate(nn.Module):
    def __init__(self, n_class, deep_supervision, dec_chn, scale=1, interp=False, enc_last=0):
        super(OutputTemplate, self).__init__()
        self.deep_supervision = deep_supervision
        self.interp = interp
        if self.interp:
            d5 = enc_last * scale
            self.deep_supervision = False
        else:
            d3, d4, d5 = scale * dec_chn[-3], scale * dec_chn[-2], scale * dec_chn[-1]

        if self.deep_supervision:
            self.output_block_ds3 = OutputBlock(d3, n_class, interp)
            self.output_block_ds4 = OutputBlock(d4, n_class, interp)
        self.output_block = OutputBlock(d5, n_class, interp)

    def forward(self, dec5, dec4, dec3):
        out = self.output_block(dec5)
        if self.training and self.deep_supervision:
            out_dec3 = self.output_block_ds3(dec3)
            out_dec4 = self.output_block_ds4(dec4)
            return [out, out_dec4, out_dec3]
        return out


class UNetLoc(nn.Module):
    def __init__(self, args, in_channels=3, n_class=2):
        super(UNetLoc, self).__init__()
        self.unet = UNetTemplate(args, in_channels)
        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            self.unet.dec_chn,
            interp=args.interpolate,
            enc_last=self.unet.enc_chn[-1],
        )

    def forward(self, data):
        dec5, dec4, dec3 = self.unet(data)
        out = self.output_block(dec5, dec4, dec3)
        return out


class SiameseUNet(nn.Module):
    def __init__(self, args, n_class):
        super(SiameseUNet, self).__init__()
        self.unet = UNetTemplate(args)
        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            self.unet.dec_chn,
            2,
            args.interpolate,
            self.unet.enc_chn[-1],
        )

    def forward(self, data):
        pre_dec5, pre_dec4, pre_dec3 = self.unet(data[:, :3])
        post_dec5, post_dec4, post_dec3 = self.unet(data[:, 3:])
        dec5, dec4, dec3 = concat(pre_dec5, post_dec5), concat(pre_dec4, post_dec4), concat(pre_dec3, post_dec3)
        out = self.output_block(dec5, dec4, dec3)
        return out


class SiameseEncUNet(nn.Module):
    def __init__(self, args, n_class):
        super(SiameseEncUNet, self).__init__()
        self.use_ppm = args.ppm
        self.use_aspp = args.aspp
        self.dilation = args.dilation
        self.no_skip = args.no_skip
        if args.loss_str == "mse":
            n_class = 1
        elif args.loss_str == "level":
            n_class = 4

        self.enc_chn, self.enc_l1, self.enc_l2, self.enc_l3, self.enc_l4, self.enc_l5 = get_encoder(
            args.encoder, self.dilation
        )

        if self.use_ppm:
            self.ppm = PPM(self.enc_chn[-1])
        elif self.use_aspp:
            self.aspp = ASPP(self.enc_chn[-1], self.dilation)

        self.dec_chn = None
        self.enc_chn = [2 * enc for enc in self.enc_chn]
        self.dec_chn, self.dec_l1, self.dec_l2, self.dec_l3, self.dec_l4, self.dec_l5 = get_decoder(
            self.enc_chn, self.dilation, args.attention, self.no_skip, args.dec_interp
        )

        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            self.dec_chn,
            1,
        )

    def forward_enc(self, data):
        enc1 = self.enc_l1(data)
        enc2 = self.enc_l2(enc1)
        enc3 = self.enc_l3(enc2)
        enc4 = self.enc_l4(enc3)
        enc5 = self.enc_l5(enc4)
        if self.use_ppm:
            enc5 = self.ppm(enc5)
        elif self.use_aspp:
            enc5 = self.aspp(enc5)
        return enc1, enc2, enc3, enc4, enc5

    def forward(self, data):
        enc1_pre, enc2_pre, enc3_pre, enc4_pre, enc5_pre = self.forward_enc(data[:, :3])
        enc1_post, enc2_post, enc3_post, enc4_post, enc5_post = self.forward_enc(data[:, 3:])
        enc1 = concat(enc1_pre, enc1_post)
        enc2 = concat(enc2_pre, enc2_post)
        enc3 = concat(enc3_pre, enc3_post)
        enc4 = concat(enc4_pre, enc4_post)
        enc5 = concat(enc5_pre, enc5_post)

        if self.dilation == 1:
            if self.no_skip:
                enc1, enc2, enc3, enc4 = None, None, None, None
            dec1 = self.dec_l1(enc5, enc4)
            dec2 = self.dec_l2(dec1, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 2:
            if self.no_skip:
                enc1, enc2, enc3 = None, None, None
            dec2 = self.dec_l2(enc5, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 4:
            if self.no_skip:
                enc1, enc2 = None, None
            dec3 = self.dec_l3(enc5, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)

        out = self.output_block(dec5, dec4, dec3)
        return out


class FusedUNet(nn.Module):
    def __init__(self, args, n_class):
        super(FusedUNet, self).__init__()
        self.use_ppm = args.ppm
        self.use_aspp = args.aspp
        self.dilation = 1
        _, self.enc_l1_pre, self.enc_l2_pre, self.enc_l3_pre, self.enc_l4_pre, self.enc_l5_pre = get_encoder(
            args.encoder, self.dilation, in_channels=3
        )
        enc_chn, self.enc_l1_post, self.enc_l2_post, self.enc_l3_post, self.enc_l4_post, self.enc_l5_post = get_encoder(
            args.encoder, self.dilation, in_channels=3
        )

        self.fusion_block1 = FusionBlock(self.enc_l1_pre, self.enc_l1_post, enc_chn[0])
        self.fusion_block2 = FusionBlock(self.enc_l2_pre, self.enc_l2_post, enc_chn[1])
        self.fusion_block3 = FusionBlock(self.enc_l3_pre, self.enc_l3_post, enc_chn[2])
        self.fusion_block4 = FusionBlock(self.enc_l4_pre, self.enc_l4_post, enc_chn[3])
        self.fusion_block5 = FusionBlock(self.enc_l5_pre, self.enc_l5_post, enc_chn[4])

        _, self.dec_l1_pre, self.dec_l2_pre, self.dec_l3_pre, self.dec_l4_pre, self.dec_l5_pre = get_decoder(
            enc_chn, self.dilation, args.attention, args.dec_interp
        )

        dec_chn, self.dec_l1_post, self.dec_l2_post, self.dec_l3_post, self.dec_l4_post, self.dec_l5_post = get_decoder(
            enc_chn, self.dilation, args.attention, args.dec_interp
        )

        self.fusion_block_dec1 = FusionBlock(self.dec_l1_pre, self.dec_l1_post, dec_chn[0])
        self.fusion_block_dec2 = FusionBlock(self.dec_l2_pre, self.dec_l2_post, dec_chn[1])
        self.fusion_block_dec3 = FusionBlock(self.dec_l3_pre, self.dec_l3_post, dec_chn[2])
        self.fusion_block_dec4 = FusionBlock(self.dec_l4_pre, self.dec_l4_post, dec_chn[3])
        self.fusion_block_dec5 = FusionBlock(self.dec_l5_pre, self.dec_l5_post, dec_chn[4])

        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            dec_chn,
            2,
        )

    def forward(self, data):
        data_pre, data_post = data[:, :3], data[:, 3:]
        enc1_pre, enc1_post = self.fusion_block1(data_pre, data_post)
        enc2_pre, enc2_post = self.fusion_block2(enc1_pre, enc1_post)
        enc3_pre, enc3_post = self.fusion_block3(enc2_pre, enc2_post)
        enc4_pre, enc4_post = self.fusion_block4(enc3_pre, enc3_post)
        enc5_pre, enc5_post = self.fusion_block5(enc4_pre, enc4_post)

        dec1_pre, dec1_post = self.fusion_block_dec1(enc5_pre, enc5_post, enc4_pre, enc4_post)
        dec2_pre, dec2_post = self.fusion_block_dec2(dec1_pre, dec1_post, enc3_pre, enc3_post)
        dec3_pre, dec3_post = self.fusion_block_dec3(dec2_pre, dec2_post, enc2_pre, enc2_post)
        dec4_pre, dec4_post = self.fusion_block_dec4(dec3_pre, dec3_post, enc1_pre, enc1_post)
        dec5_pre, dec5_post = self.fusion_block_dec5(dec4_pre, dec4_post, last_dec=True)

        dec5, dec4, dec3 = concat(dec5_pre, dec5_post), concat(dec4_pre, dec4_post), concat(dec3_pre, dec3_post)
        out = self.output_block(dec5, dec4, dec3)
        return out


class FusedEncUNet(nn.Module):
    def __init__(self, args, n_class):
        super(FusedEncUNet, self).__init__()
        self.use_ppm = args.ppm
        self.use_aspp = args.aspp
        self.dilation = 1
        _, self.enc_l1_pre, self.enc_l2_pre, self.enc_l3_pre, self.enc_l4_pre, self.enc_l5_pre = get_encoder(
            args.encoder, self.dilation, in_channels=3
        )
        enc_chn, self.enc_l1_post, self.enc_l2_post, self.enc_l3_post, self.enc_l4_post, self.enc_l5_post = get_encoder(
            args.encoder, self.dilation, in_channels=3
        )

        self.fusion_block1 = FusionBlock(self.enc_l1_pre, self.enc_l1_post, enc_chn[0])
        self.fusion_block2 = FusionBlock(self.enc_l2_pre, self.enc_l2_post, enc_chn[1])
        self.fusion_block3 = FusionBlock(self.enc_l3_pre, self.enc_l3_post, enc_chn[2])
        self.fusion_block4 = FusionBlock(self.enc_l4_pre, self.enc_l4_post, enc_chn[3])
        self.fusion_block5 = FusionBlock(self.enc_l5_pre, self.enc_l5_post, enc_chn[4])

        dec_chn, self.dec_l1, self.dec_l2, self.dec_l3, self.dec_l4, self.dec_l5 = get_decoder(
            enc_chn, self.dilation, args.attention, args.dec_interp
        )

        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            dec_chn,
            1,
        )

    def forward(self, data):
        data_pre, data_post = data[:, :3], data[:, 3:]
        enc1_pre, enc1_post = self.fusion_block1(data_pre, data_post)
        enc2_pre, enc2_post = self.fusion_block2(enc1_pre, enc1_post)
        enc3_pre, enc3_post = self.fusion_block3(enc2_pre, enc2_post)
        enc4_pre, enc4_post = self.fusion_block4(enc3_pre, enc3_post)
        enc5_pre, enc5_post = self.fusion_block5(enc4_pre, enc4_post)

        dec1 = self.dec_l1(enc5_post, enc4_post)
        dec2 = self.dec_l2(dec1, enc3_post)
        dec3 = self.dec_l3(dec2, enc2_post)
        dec4 = self.dec_l4(dec3, enc1_post)
        dec5 = self.dec_l5(dec4, None)

        out = self.output_block(dec5, dec4, dec3)
        return out


class ParallelUNet(nn.Module):
    def __init__(self, args, n_class):
        super(ParallelUNet, self).__init__()
        self.unet_pre = UNetTemplate(args)
        self.unet_post = UNetTemplate(args)
        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            self.unet_pre.dec_chn,
            2,
            args.interpolate,
            self.unet_pre.enc_chn[-1],
        )

    def forward(self, data):
        dec5_pre, dec4_pre, dec3_pre = self.unet_pre(data[:, :3])
        dec5_post, dec4_post, dec3_post = self.unet_pre(data[:, :3])
        dec5, dec4, dec3 = concat(dec5_pre, dec5_post), concat(dec4_pre, dec4_post), concat(dec3_pre, dec3_post)
        out = self.output_block(dec5, dec4, dec3)
        return out


class ParallelEncUNet(nn.Module):
    def __init__(self, args, n_class):
        super(ParallelEncUNet, self).__init__()
        self.use_ppm = args.ppm
        self.use_aspp = args.aspp
        self.dilation = args.dilation
        self.no_skip = args.no_skip
        self.interpolate = args.interpolate

        self.enc_chn, self.enc_l1_pre, self.enc_l2_pre, self.enc_l3_pre, self.enc_l4_pre, self.enc_l5_pre = get_encoder(
            args.encoder, self.dilation
        )

        _, self.enc_l1_post, self.enc_l2_post, self.enc_l3_post, self.enc_l4_post, self.enc_l5_post = get_encoder(
            args.encoder, self.dilation
        )

        if self.use_ppm:
            self.ppm_pre = PPM(self.enc_chn[-1])
            self.ppm_post = PPM(self.enc_chn[-1])
        elif self.use_aspp:
            self.aspp_pre = ASPP(self.enc_chn[-1], self.dilation)
            self.aspp_post = ASPP(self.enc_chn[-1], self.dilation)

        self.dec_chn = None
        self.enc_chn = [2 * enc for enc in self.enc_chn]
        if not self.interpolate:
            self.dec_chn, self.dec_l1, self.dec_l2, self.dec_l3, self.dec_l4, self.dec_l5 = get_decoder(
                self.enc_chn, self.dilation, args.attention, self.no_skip, args.dec_interp
            )

        self.output_block = OutputTemplate(
            n_class,
            args.deep_supervision,
            self.dec_chn,
            1,
            args.interpolate,
            self.enc_chn[-1],
        )

    def forward_enc(self, data, pre):
        enc1 = self.enc_l1_pre(data) if pre else self.enc_l1_post(data)
        enc2 = self.enc_l2_pre(enc1) if pre else self.enc_l2_post(enc1)
        enc3 = self.enc_l3_pre(enc2) if pre else self.enc_l3_post(enc2)
        enc4 = self.enc_l4_pre(enc3) if pre else self.enc_l4_post(enc3)
        enc5 = self.enc_l5_pre(enc4) if pre else self.enc_l5_post(enc4)
        return enc1, enc2, enc3, enc4, enc5

    def forward(self, data):
        enc1_pre, enc2_pre, enc3_pre, enc4_pre, enc5_pre = self.forward_enc(data[:, :3], True)
        enc1_post, enc2_post, enc3_post, enc4_post, enc5_post = self.forward_enc(data[:, 3:], False)
        if self.use_ppm:
            enc5_pre = self.ppm_pre(enc5_pre)
            enc5_post = self.ppm_post(enc5_post)
        elif self.use_aspp:
            enc5_pre = self.aspp_pre(enc5_pre)
            enc5_post = self.aspp_post(enc5_post)

        if self.interpolate:
            return self.output_block(concat(enc5_pre, enc5_post), None, None)

        enc1 = concat(enc1_pre, enc1_post)
        enc2 = concat(enc2_pre, enc2_post)
        enc3 = concat(enc3_pre, enc3_post)
        enc4 = concat(enc4_pre, enc4_post)
        enc5 = concat(enc5_pre, enc5_post)

        if self.dilation == 1:
            if self.no_skip:
                enc1, enc2, enc3, enc4 = None, None, None, None
            dec1 = self.dec_l1(enc5, enc4)
            dec2 = self.dec_l2(dec1, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 2:
            if self.no_skip:
                enc1, enc2, enc3 = None, None, None
            dec2 = self.dec_l2(enc5, enc3)
            dec3 = self.dec_l3(dec2, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)
        elif self.dilation == 4:
            if self.no_skip:
                enc1, enc2 = None, None
            dec3 = self.dec_l3(enc5, enc2)
            dec4 = self.dec_l4(dec3, enc1)
            dec5 = self.dec_l5(dec4, None)

        out = self.output_block(dec5, dec4, dec3)
        return out


class DiffUNet(nn.Module):
    def __init__(self, args, n_class):
        super(DiffUNet, self).__init__()
        self.unet = UNetLoc(args, in_channels=3, n_class=n_class)

    def forward(self, data):
        data = data[:, :3] - data[:, 3:]
        out = self.unet(data)
        return out


class CatUNet(nn.Module):
    def __init__(self, args, n_class):
        super(CatUNet, self).__init__()
        self.unet = UNetLoc(args, in_channels=6, n_class=n_class)

    def forward(self, data):
        out = self.unet(data)
        return out
