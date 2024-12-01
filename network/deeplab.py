import torch
import torch.nn as nn
import torch.nn.functional as F
from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from network.aspp import build_aspp
from network.decoder import build_decoder
from network.backbone import build_backbone
from util.attention import MLCA, CDCA

Attention_blocks = [MLCA, CDCA]

class DeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=3,
                 sync_bn=True, freeze_bn=False, phi1=1, phi2=2):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        # 注意力机制的初始化改为根据传入的phi参数值来初始化对应的注意力模块
        self.phi1 = phi1
        self.attention_block1 = None
        if phi1 is not None and 1 <= self.phi1 <= len(Attention_blocks):
            self.attention_block1 = Attention_blocks[self.phi1 - 1](24)

        self.phi2 = phi2
        self.attention_block2 = None
        if phi2 is not None and 1 <= self.phi2 <= len(Attention_blocks):
            self.attention_block2 = Attention_blocks[self.phi2 - 1](256)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)

        if self.attention_block1 is not None:
            low_level_feat = self.attention_block1(low_level_feat)

        x = self.aspp(x)

        # 应用注意力机制（如果已初始化）
        if self.attention_block2 is not None:
            x = self.attention_block2(x)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
