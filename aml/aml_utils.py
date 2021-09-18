#
#
#
# Utils for AML.
#
import torch


def print_grads(nn, optim, debug_dict):
    # https://discuss.pytorch.org/t/
    # how-to-print-the-computed-gradient-values-for-a-network/34179/8
    # g_base_loss.register_hook(lambda grad: print(grad))
    # g_loss.register_hook(lambda grad: print(grad))
    print('============= GRADS ==============')
    print(nn)
    torch.set_printoptions(precision=4, linewidth=150, threshold=10)
    for loss_nm, loss in debug_dict.items():
        if '_loss' not in loss_nm:
            continue  # other debug data
        optim.zero_grad()
        loss.backward(retain_graph=True)
        for param_nm, param in nn.named_parameters():
            grad, *_ = param.grad.data
            print(f'{loss_nm}: NN grad of {param_nm} {param}:\n{grad}')
    torch.set_printoptions(precision=4, linewidth=150, threshold=500000)
