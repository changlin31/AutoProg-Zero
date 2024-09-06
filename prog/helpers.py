import torch

import diffusion

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import numpy as np
import torch.nn as nn
from collections import defaultdict

def set_model_config(model, current_l):


    train_dict = {1: [f'blocks.{num}' for num in range(0,28,4)],
                   2: [f'blocks.{num}' for num in range(0,28,4)] + [f'blocks.{num}' for num in range(1,28,4)],
                   3: [f'blocks.{num}' for num in range(0,28,4)] + [f'blocks.{num}' for num in range(1,28,4)] + [f'blocks.{num}' for num in range(2,28,4)],
                   4: [f'blocks.{num}' for num in range(0,28)],
                   }


    for name, param in model.named_parameters():
        param.requires_grad = any([kw in name for kw in train_dict[current_l]])

def no_repeats(a: list):
    b = []
    for e in a:
        if e not in b:
            b.append(e)
    return b



# Zico calculation
def getgrad(model:torch.nn.Module, grad_dict:defaultdict):


    for name,mod in model.named_modules():
        if (isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear)) and mod.weight.grad is not None:
            grad_dict[name].append(mod.weight.grad.cpu().reshape( -1).numpy())
    return grad_dict

def caculate_zico(grad_dict):
    allgrad_array=None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname]= np.array(grad_dict[modname])
    cnt = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx])
        if tmpsum==0:
            pass
        else:
            cnt += 1
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx]/nsr_std[nonzero_idx]))

    # return nsr_mean_avg_abs / cnt
    return nsr_mean_sum_abs


# NTK calculation
def get_ntk_n(networks, x_start, t, model_kwargs=None, diffusion=None, optimizer=None):

    ######
    grads = [[] for _ in range(len(networks))]

    # inputs = inputs.cuda(device=device, non_blocking=True)
    for net_idx, network in enumerate(networks):

        optimizer.zero_grad()

        logit = network(x_start, t, **model_kwargs)
        # print("logit.shape:",logit.shape)
        for _idx in range(len(x_start)):
            logit[_idx:_idx+1].backward(torch.ones_like(logit[_idx:_idx+1]), retain_graph=True)
            grad = []
            for name, W in network.named_parameters():
                if 'weight' in name and W.grad is not None:
                    grad.append(W.grad.view(-1).detach().to('cpu'))
            grads[net_idx].append(torch.cat(grad, -1))
            optimizer.zero_grad()

            torch.cuda.empty_cache()


    ######
    # 找到最小的梯度大小
    min_grad_size = min([min(g.shape[0] for g in _grads) for _grads in grads])
    # 对每个梯度进行截取以统一尺寸
    processed_grads = []
    for _grads in grads:
        processed = [grad[:min_grad_size] for grad in _grads]
        processed_grads.append(torch.stack(processed, 0))
    grads = processed_grads
    # grads = [torch.stack(_grads, 0) for _grads in grads]
    ntks = [torch.einsum('nc,mc->nm', [_grads, _grads]) for _grads in grads]
    conds = []
    for ntk in ntks:
        eigenvalues = torch.linalg.eigvalsh(ntk, UPLO='U')  # ascending

        conds.append(np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True))


    return conds



def compute_NTK_score(model, x_start, t, model_kwargs=None, diffusion=None, optimizer=None):
    ntk_score = get_ntk_n([model], x_start, t, model_kwargs=model_kwargs, diffusion=diffusion, optimizer=optimizer)[0]
    return -1 * ntk_score


if __name__ == '__main__':
    pass