

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import chkpts as cp

import sys
sys.path.append("..") # adds higher directory to python modules path.
try: # HACK: import beyond current level, may need to restructure
    from config import Config
except ImportError:
    assert False, 'Fail to import config.py'

class Net(nn.Module):
    """ 
    A base class provides a common weight initialization scheme.
    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if 'conv' in classname.lower():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if 'norm' in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    @staticmethod
    def save_mem(prev_feat, blk_func):
        if any(feat.requires_grad for feat in prev_feat):
            args = (prev_feat,) + tuple(blk_func.parameters()) 
            feat = cp.CheckpointFunction.apply(blk_func, 1, *args)
        else:
            feat = blk_func(prev_feat)
        return feat
        
    def forward(self, x):
        return x

class SquuezeExciteUnit(Net, Config):   
    def __init__(self, in_ch, se_ch):
        super(SquuezeExciteUnit, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.unit = nn.Sequential(
            nn.Linear(in_ch, se_ch),
            nn.ReLU(inplace=True),
            nn.Linear(se_ch, in_ch),
            nn.Sigmoid(),
        )

    def forward(self, prev_feat):
        def GlobalAvgPooling(l):
            return F.adaptive_avg_pool2d(l, (1, 1)).view(l.size(0), -1)

        feat = GlobalAvgPooling(prev_feat)
        feat = self.unit(feat) 
        # NOTE: hard coded the number of channels
        feat = feat.view(-1, 8, 1, 1)
        feat = prev_feat * feat
        return feat

class DenseUnit(Net, Config):   
    def __init__(self, in_ch, ksize, ch, padding='same', efficient=True):
        super(DenseUnit, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.efficient = efficient
        self.padding = padding
        # NOTE: may be more complicated if dilate etc. involve
        if padding == 'same':
            pad = [kernel // 2 for kernel in ksize]
        else:
            pad = [0 for kernel in ksize] 

        self.conv1 = nn.Sequential(
            nn.GroupNorm(in_ch // 4, in_ch, eps=1e-5),    
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, ch[0], ksize[0], stride=1, padding=pad[0], bias=False)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(ch[0] // 4, ch[0], eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[1], ksize[1], stride=1, padding=pad[1], bias=False)
        )
        self.squeeze_excite = SquuezeExciteUnit(ch[1], ch[1] // 2)

    def forward(self, prev_feat, attent=None):
        feat = self.save_mem(prev_feat, self.conv1)
        feat = self.conv2(feat)
        feat = self.squeeze_excite(feat)
        feat = feat * attent if attent is not None else feat
        feat = torch.cat([prev_feat, feat], dim=1)        
        return feat
        
class DenseBlock(Net, Config):
    def __init__(self, in_ch, out_ch, ksize, ch, nr_unit, padding='same'):
        super(DenseBlock, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.nr_unit = nr_unit
        # define 1 dense unit
        unit_input_ch = in_ch

        # wrapper so that params in list are visible
        self.dense_units = nn.ModuleList() 
        for i in range (0, nr_unit):
            self.dense_units.append(DenseUnit(unit_input_ch, ksize, ch, padding))
            unit_input_ch += ch[1]

        self.blk_final = nn.Sequential(
            nn.GroupNorm(unit_input_ch // 4, unit_input_ch, eps=1e-5), 
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_input_ch, out_ch, 1, stride=1, padding=1)
        )

    def forward(self, prev_feat, attent=None):
        for idx in range (0, self.nr_unit):
            dense_unit = self.dense_units[idx]
            prev_feat = dense_unit(prev_feat, attent)       
        feat = self.blk_final(prev_feat)
        return feat

class DenseNet(Net, Config):
    def __init__(self, input_ch, nr_classes, feat_mode=False):
        super(DenseNet, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.feat_mode = feat_mode

        self.d0 = nn.Conv2d(input_ch, 64, 7, stride=2, padding=3)

        self.d1_pool  = nn.MaxPool2d((3, 3), stride=2, padding=1)
        self.d1_dense = DenseBlock( 64,  64, [1, 3], [32, 8], 6)

        self.d2_pool  = nn.AvgPool2d((2, 2), stride=2, padding=-1)
        self.d2_dense = DenseBlock( 64,  80, [1, 3], [32, 8], 12)

        self.d3_pool  = nn.AvgPool2d((2, 2), stride=2, padding=-1)
        self.d3_dense = DenseBlock( 80, 168, [1, 3], [32, 8], 32)

        self.d4_pool  = nn.AvgPool2d((2, 2), stride=2, padding=-1)
        self.d4_dense = DenseBlock(168, 212, [1, 3], [32, 8], 32)

        self.preact_out = nn.Sequential(
            nn.GroupNorm(212 // 4, 212, eps=1e-5), 
            nn.ReLU(inplace=True), 
        )
        self.classifier = nn.Sequential(
            nn.Linear(212, nr_classes)
        )

        self.weights_init()

    def forward(self, imgs, attents=None):
        def GlobalAvgPooling(l):
            return F.adaptive_avg_pool2d(l, (1, 1)).view(l.size(0), -1)

        attent = [None] * 4
        # nucs = imgs[:,3:,:] # split channel
        # imgs = imgs[:,:3,:] # split channel 

        # nucs /= 255.0 # 2048x2048
        # attent[0] = F.interpolate(nucs, scale_factor=float(1/4) , mode='bilinear', align_corners=True)
        # attent[1] = F.interpolate(nucs, scale_factor=float(1/8) , mode='bilinear', align_corners=True)
        # attent[2] = F.interpolate(nucs, scale_factor=float(1/16), mode='bilinear', align_corners=True)
        # attent[3] = F.interpolate(nucs, scale_factor=float(1/32), mode='bilinear', align_corners=True)
        
        d0 = self.d0(imgs)
        d1 = self.d1_pool(d0)
        d1 = self.d1_dense(d1, attent[0])

        d2 = self.d2_pool(d1)
        d2 = self.d2_dense(d2, attent[1])

        d3 = self.d3_pool(d2)
        d3 = self.d3_dense(d3, attent[2])

        d4 = self.d4_pool(d3)
        d4 = self.d4_dense(d4, attent[3])
        
        out = self.preact_out(d4)
        # global average pooling operation
        out = GlobalAvgPooling(out)
        out = self.classifier(out)
        return out
