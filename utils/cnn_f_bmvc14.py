import torch
import torch.nn as nn
from torch.nn import init
import scipy.io as sio
from collections import OrderedDict
import os
import wget

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.constant_(m.bias, 0.0)

##================================
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)

##================================
def get_weight_bias(layers, i):
    """
    get weight and bias from matConvNet pretrained models
    """
    weights=layers[i][0][0][2][0][0]
    tensor_weight = torch.from_numpy(weights)
    tensor_weight = tensor_weight.permute(3,2,0,1)  # [H, W, C, N] to [N, C, H, W]
    bias=layers[i][0][0][2][0][1]
    tensor_bias = torch.from_numpy(bias)
    tensor_bias = torch.squeeze(tensor_bias)

    return tensor_weight, tensor_bias


##========================================================================
##========================================================================
# CNN_F converted from matConvNet pretrained model

class CNN_F(nn.Module):
    """
    implement from matConvNet (1x1 Conv instead of Linear layer in fc)
    """

    def __init__(self, code_length=12):
        super(CNN_F, self).__init__()

        self.code_length = code_length

        if not os.path.exists('imagenet-vgg-f.mat'):
            url = "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat"
            wget.download(url, 'imagenet-vgg-f.mat')
        cnn_f_matConvNet = sio.loadmat('imagenet-vgg-f.mat')
        cnn_f_layers = cnn_f_matConvNet['layers'][0]

        self.layer1 = nn.Sequential(OrderedDict([
            ('conv_1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0)),
            ('relu_1', nn.ReLU(inplace=True)),
            ('lrn_1', nn.LocalResponseNorm(size=5, k=2.0)),
            ('maxPool_1', nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        ]))

        self.layer2 = nn.Sequential(OrderedDict([
            ('conv_2', nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2)),
            ('relu_2',nn.ReLU(inplace=True)),
            ('lrn_2',nn.LocalResponseNorm(size=5, k=2.0)),
            ('maxPool_2',nn.MaxPool2d(kernel_size=3,stride=2,padding=1)),
        ]))

        self.layer3 = nn.Sequential(OrderedDict([
            ('conv_3', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu_3', nn.ReLU(inplace=True)),
        ]))

        self.layer4 = nn.Sequential(OrderedDict([
            ('conv_4', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu_4', nn.ReLU(inplace=True)),
        ]))

        self.layer5 = nn.Sequential(OrderedDict([
            ('conv_5', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('relu_5', nn.ReLU(inplace=True)),
            ('maxPool_5', nn.MaxPool2d(kernel_size=3,stride=2,padding=0)),
        ]))

        self.layer6 = nn.Sequential(OrderedDict([
            ('conv_6', nn.Conv2d(256, 4096, kernel_size=6, stride=1, padding=0)),
            # ('conv_6', nn.Linear(256*6*6, 4096)),
            ('relu_6', nn.ReLU(inplace=True)),
            ('drop_6', nn.Dropout()),
        ]))

        self.layer7 = nn.Sequential(OrderedDict([
            ('conv_7', nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0)),
            # ('conv_7', nn.Linear(4096, 4096)),
            ('relu_7', nn.ReLU(inplace=True)),
            ('drop_7', nn.Dropout()),
        ]))

        self.layer8 = nn.Sequential(OrderedDict([
            ('fc_8', nn.Linear(4096, self.code_length)),
            # ('conv_8', nn.Conv2d(4096, code_length, kernel_size=1, stride=1, padding=0)),
            ('tanh_8', nn.Tanh()),
        ]))

        self.features = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4, self.layer5)
        self.classifier = nn.Sequential(self.layer6, self.layer7)

        ##=======================================================
        ## weight initalizaiton
        layer_list = [0, 4, 8, 10, 12, 15, 17] # conv layers in the matConvNet cnn-f model
        for (i, k) in zip(range(1,8), layer_list):
            pre_weight, pre_bias = get_weight_bias(cnn_f_layers, k)
            layer = getattr(self, "layer{}".format(i))
            conv = getattr(layer, "conv_{}".format(i))
            conv.weight = torch.nn.Parameter(pre_weight)
            conv.bias = torch.nn.Parameter(pre_bias)
        self.layer8.apply(weights_init_classifier)
        ##=======================================================

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1) ### x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        x = self.layer8(x)
        return x


if __name__ == "__main__":

    ## testing
    cnn_f_matConvNet = sio.loadmat('imagenet-vgg-f.mat')
    cnn_f_layers = cnn_f_matConvNet['layers'][0]
    p_weight, p_bias = get_weight_bias(cnn_f_layers, 17) # [i][0][0][2][0][1]
    print(cnn_f_matConvNet['layers'][0][17])
    cnn_f_model = CNN_F()
