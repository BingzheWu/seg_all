from .ops import conv_block, upsample_layer
import torch
import torch.nn as nn
from torch.autograd import Variable
class unet(nn.Module):
    def __init__(self, in_c, out_c, num_downs, nf, norm_layer = nn.BatchNorm2d, use_dropout = False, gpu_ids =[] ):
        super(unet, self).__init__()
        self.gpu_ids = gpu_ids
        unet_block_ = unet_block(nf*8, nf*8, input_nc = None, innermost = True, submodule=None)
        for i in range(num_downs-5):
            unet_block_ = unet_block(nf*8, nf*8, input_nc = None, submodule = unet_block_)
        unet_block_ = unet_block(nf*4, nf*8, submodule = unet_block_ )
        unet_block_ = unet_block(nf*2, nf*4, submodule = unet_block_)
        unet_block_ = unet_block(nf, nf*2, submodule = unet_block_,)
        unet_blcok_ = unet_block(out_c, nf, input_nc = in_c, outermost = True, submodule = unet_block_)
        self.model = unet_blcok_
    def forward(self, input):
        return self.model(input)
def test_unet():
    unet_ = unet(in_c = 3, out_c = 1, num_downs = 7, nf = 64)
    input = Variable(torch.zeros((1,3, 256, 256)))
    out = unet_(input)
    print(out.size())
class unet_block(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc = None, 
                submodule = None, outermost = False, innermost = False, 
                norm_layer = nn.BatchNorm2d, use_dropout = False):
        super(unet_block, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = conv_block(input_nc, inner_nc, 4, 2, 1,alpha = 0.2)
        if innermost:
            upconv = upsample_layer(inner_nc, outer_nc)
            uprelu = nn.LeakyReLU(0.2,True)
            down = [downconv]
            up = [upconv, uprelu]
            model = down+up
        elif outermost:
            upconv = upsample_layer(inner_nc*2, outer_nc)
            uptanh= nn.Tanh()
            down = [downconv]
            up = [upconv]
            model = down + [submodule] + up 
        else:
            upconv = upsample_layer(inner_nc*2, outer_nc,)
            uprelu = nn.LeakyReLU(0.2, True)
            if submodule is not None:
                model = [downconv]+[submodule]+[upconv]+[uprelu]
            else:
                model = [downconv]+[upconv]+[uprelu]
        self.model = nn.Sequential(*model)
    def forward(self, x):
        if self.outermost:
            return(self.model(x))
        return torch.cat([x, self.model(x)], 1)
if __name__ == '__main__':
    test_unet()
