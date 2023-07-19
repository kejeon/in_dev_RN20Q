import torch
import torch.nn as nn

def kl_div_loss(conv_filter):
    num_kernel = conv_filter.shape[0]
    mean_vec = torch.mean(conv_filter, dim=(1,2,3))
    std_vec = torch.std(conv_filter, dim=(1,2,3))

    mean_mat = mean_vec.repeat(num_kernel,1)
    std_mat = std_vec.repeat(num_kernel,1)

    mean_mat_T = mean_mat.T
    std_mat_T = std_mat.T

    kl_div_mat = torch.div(std_mat.T, std_mat).log() + (
        std_mat.square() + (mean_mat - mean_mat.T).square())/(2*std_mat.T.square()) - 1/2
    
    kl_div = torch.mean(kl_div_mat) * 2/(num_kernel*(num_kernel-1))

    return kl_div

def avg_kl_div_loss(model):
    kl_div_total = 0

    def nn_traversal(root):
        global kl_div_total
        for name, layer in root.named_children():
            if len(layer._modules) == 0:
                if not isinstance(layer, nn.Conv2d):
                    continue
                if layer.in_channels != layer.out_channels:
                    continue
                my_tensor = layer.weight
                kl_div_total += kl_div_loss(my_tensor)
            else:
                nn_traversal(layer)
        return
    nn_traversal(model)
    
    return kl_div_total