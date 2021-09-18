#
#
#
# NNs for AML.
#
import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hl_output_sizes, activation, output_layer, device, drop=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hl_output_sizes = hl_output_sizes
        self.activation = activation
        self.output_layer = output_layer
        self.device = device
        self.drop = drop
        self.nn = self.make_mlp()

    def make_mlp(self):
        """Make a multilayer perceptron."""
        net = OrderedDict()
        previous_size = self.in_size
        print(f'self.hl_output_sizes={self.hl_output_sizes}')
        for layer_number, hl_output_size in enumerate(self.hl_output_sizes):
            nm = f'dense{layer_number}'
            if layer_number+1 == len(self.hl_output_sizes):
                nm = f'pre_out{layer_number}'
            net[nm] = nn.Linear(previous_size, hl_output_size)
            net[f'activation{layer_number}'] = self.activation
            if self.drop is not None:
                net['drop{layer_number}'] = nn.Dropout(p=self.drop)
            previous_size = hl_output_size
        net['output'] = nn.Linear(previous_size, self.out_size)
        if self.output_layer is not None:
            net['output_layer'] = self.output_layer
        return nn.Sequential(net)


class GNN(MLP):
    """Neural net for learning a relation g."""
    #              self.g_inp_ndims, 1, hl_output_sizes=gh, activation=activation, out_activation=out_nl, device=device
    def __init__(self, in_size, out_size, hl_output_sizes, activation, out_activation, device):
        super().__init__(in_size, out_size, hl_output_sizes, activation, out_activation, device)
        self.register_buffer('on_mean', torch.zeros(1))
        self.register_buffer('off_mean', torch.zeros(1))
        # self.core_params = []; self.out_params = []
        # for name, child in self.nn.named_children():
        #    if name=='output':
        #        self.out_params.append(child.bias)
        #        self.core_params.append(child.weight)
        #    else:
        #        for param in child.parameters():
        #            self.core_params.append(param)
        # assert(len(self.core_params)>0)
        # assert(len(self.out_params)>0)
        self.to(device)
        self.train()

    def forward(self, inp):
        return self.nn(inp)

    # def toggle_out_grad(self, requires_grad, debug=False):
    #    for param in self.out_params:
    #        param.requires_grad_(requires_grad)
    #        if debug:
    #            print('param {:0.8f} grad {:0.8f}'.format(
    #                param.item(), param.grad.item()))


class RestrictedSyzygy(MLP):
    """Neural net for learning a syzygy."""
    def __init__(self, in_size, hl_output_sizes, nl, k, f_max, device):
        out_activation = torch.nn.Hardtanh(-f_max, f_max)
        super().__init__(in_size, k - 1, hl_output_sizes, nl, out_activation, device)
        # self.pairwise_nns = []
        # self.pairwise_nns = nn.ModuleList(
        #    [make_MLP(in_size, 1, hidden_layers, activation, output_layer=output_layer) for j in range(k-1)])
        self.to(device)
        self.train()
        print('Constructed RestrictedSyzygy for', k, 'relations')

    def forward(self, data, gj_outs, gjs_on_means, g_out, dbg_dict):
        keep_ids = keep_intersection(gj_outs, gjs_on_means)
        if keep_ids.shape[0] > 0:
            data = data[keep_ids]
            gj_outs = gj_outs[keep_ids]
            g_out = g_out[keep_ids]
        f_out = self.nn(data)
        syz_out = f_out[:, 0:gj_outs.size(-1)]*gj_outs - g_out
        # pairwise_syz_mean = []
        # assert(gj_outs.size(-1) <= len(self.pairwise_nns))
        # for j in range(gj_outs.size(-1)):
        #    keep_ids = keep_pairwise_intersection(gj_outs[:,j])
        #    if keep_ids.size(0)==0: continue
        #    pairwise_f_out = self.pairwise_nns[j](data[keep_ids])
        #    abs = torch.abs(
        #        pairwise_f_out*gj_outs[keep_ids,j] - g_out[keep_ids])
        #    pairwise_syz_mean.append(abs.mean())
        if dbg_dict is not None:
            dbg_dict['syz_fwd_f_out'] = torch.abs(f_out).mean()
            dbg_dict['syz_fwd_gj_outs'] = torch.abs(gj_outs).mean()
            dbg_dict['syz_fwd_g_out'] = torch.abs(g_out).mean()
        return syz_out, f_out, keep_ids.shape[0]

    def reset(self):
        self.nn.apply(weight_reset)


def keep_pairwise_intersection(gj_out):
    gj_out_nz = torch.where(torch.abs(gj_out) < gj_out.mean(),
                            gj_out, torch.zeros_like(gj_out))
    keep_ids = torch.nonzero(gj_out_nz, as_tuple=True)[0]
    return keep_ids


def keep_intersection(gj_outs, gjs_on_means):
    # Retain only interesting points in data: those where all g_{k-1}
    # hold (both on- and off-manifold).
    gj_outs = torch.where(torch.abs(gj_outs) < gjs_on_means.min(),
                          gj_outs, torch.zeros_like(gj_outs))
    keep_ids = torch.nonzero(gj_outs, as_tuple=True)[0]
    if keep_ids.shape[0] == 0:
        print('Warning: RestrictedSyzygy.keep_intersection would be empty')
    return keep_ids


# https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()
