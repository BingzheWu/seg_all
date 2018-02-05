from .unet import unet

def factory(arch):
    if arch == 'unet':
        network = unet(in_c = 3, out_c = 2, num_downs = 7, nf = 64)
    return network
