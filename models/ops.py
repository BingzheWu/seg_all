import torch
import torch.nn as nn

def upsample_layer(in_c, out_c, upsample_type = 'deconv', padding_type = 'zero'):
    if upsample_type == 'deconv':
        upconv = [nn.ConvTranspose2d(in_c, out_c, kernel_size = 4, stride = 2, padding= 1)]
    elif upsample_type == 'bilinear':
        upconv = [nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_c, out_c, kernel_size = 3, stride = 1, padding =0 )]
    else:
        raise NotImplementedError
    return nn.Sequential(*upconv)
def conv_block(in_c, out_c, k_size, strides, padding, 
	alpha = 0.2, bias = False, batch_norm = True):
        out = []
        out += [nn.Conv2d(in_c, out_c, k_size, strides, padding, bias = bias)]
        if batch_norm:
            out+=[nn.BatchNorm2d(out_c)]
        out+=[nn.LeakyReLU(alpha, inplace = True)]
        out = nn.Sequential(*out)
        return out