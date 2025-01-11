from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from layers_copy import *
from timm.models.layers import trunc_normal_
from depth_encoder import Cat


class Upsampling(nn.Module):  
    def __init__(self):
        super(Upsampling, self).__init__()
    def forward(self, x):
        return upsample(x)

class ExtractInitial(nn.Module):
    def forward(self, x):
        return x[2]

class ExtractSecond(nn.Module):
    def forward(self, x):
        return x[1]
    
class ExtractThird(nn.Module):
    def forward(self, x):
        return x[0]
    
# class Identity(nn.Module):
#     def forward(self, x):
#         return x

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'bilinear'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = (self.num_ch_enc / 2).astype('int')

        # decoder
        self.convs = OrderedDict()
        for i in range(2, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 2 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            # print(i, num_ch_in, num_ch_out)
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # self.cat1 = Cat(dim=1)
        self.cat_append = Cat(dim=1)
        self.upsampler = Upsampling()
        self.upsampler2 = Upsampling()
        self.initialExtractor = ExtractInitial()
        self.secondExtractor = ExtractSecond()
        self.thirdExtractor = ExtractThird()
        # self.ident = Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_features):
        self.outputs = {}
        input = input_features

        x = self.initialExtractor(input)
        x1 = self.secondExtractor(input)
        x2 = self.thirdExtractor(input)

        x = self.convs[("upconv", 2, 0)](x)
        x = self.upsampler(x)
        # x = self.listgen(x)
        # x = [upsample(x)]

        if self.use_skips:
                x = self.cat_append(x, x1)
            # else:
            # y = self.initialExtractor(input_features, i - 1)
            # # x += [input_features[i - 1]] # appending input_features to the upsampele list
            # x = self.cat_append(x, y)
        # x = torch.cat(x, 1)
        x = self.convs[("upconv", 2, 1)](x)

        if 2 in self.scales:
            f = self.convs[("dispconv", 2)](x)
            f = self.upsampler2(f)
            # f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
            self.outputs[("disp", 2)] = self.sigmoid(f)

        #------- next loop
        x = self.convs[("upconv", 1, 0)](x)
        x = self.upsampler(x)
        # x = self.listgen(x)
        # x = [upsample(x)]

        if self.use_skips:
                x = self.cat_append(x, x2)
            # else:
            # y = self.initialExtractor(input_features, i - 1)
            # # x += [input_features[i - 1]] # appending input_features to the upsampele list
            # x = self.cat_append(x, y)
        # x = torch.cat(x, 1)
        x = self.convs[("upconv", 1, 1)](x)

        if 1 in self.scales:
            f = self.convs[("dispconv", 1)](x)
            f = self.upsampler2(f)
            # f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
            self.outputs[("disp", 1)] = self.sigmoid(f)

        #------- next loop

        x = self.convs[("upconv", 0, 0)](x)
        x = self.upsampler(x)
        # x = self.listgen(x)
        # x = [upsample(x)]

           # else:
            # y = self.initialExtractor(input_features, i - 1)
            # # x += [input_features[i - 1]] # appending input_features to the upsampele list
            # x = self.cat_append(x, y)
        # x = torch.cat(x, 1)
        x = self.convs[("upconv", 0, 1)](x)

        if 0 in self.scales:
            f = self.convs[("dispconv", 0)](x)
            f = self.upsampler2(f)
            # f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
            self.outputs[("disp", 0)] = self.sigmoid(f)

        # for i in range(2, -1, -1):
        #     x = self.convs[("upconv", i, 0)](x)
        #     x = self.upsampler(x)
        #     # x = self.listgen(x)
        #     # x = [upsample(x)]

        #     if self.use_skips and i > 0:
        #         # y = input_features[i - 1]
        #         if(i == 2):
        #             x = self.cat_append(x, x1)
        #         elif(i == 1):
        #             x = self.cat_append(x, x2)
        #         # else:
        #         # y = self.initialExtractor(input_features, i - 1)
        #         # # x += [input_features[i - 1]] # appending input_features to the upsampele list
        #         # x = self.cat_append(x, y)
        #     # x = torch.cat(x, 1)
        #     x = self.convs[("upconv", i, 1)](x)

        #     if i in self.scales:
        #         f = self.convs[("dispconv", i)](x)
        #         f = self.upsampler2(f)
        #         # f = upsample(self.convs[("dispconv", i)](x), mode='bilinear')
        #         self.outputs[("disp", i)] = self.sigmoid(f)

        return self.outputs

