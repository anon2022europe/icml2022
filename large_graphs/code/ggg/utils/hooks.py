from _warnings import warn
from logging import warning

import torch as pt


def forward_clip_hook(module, input, output):
    if pt.is_tensor(output):
        outs = [output]
    else:
        outs = output
    new_o = []
    for i, o in enumerate(outs):
        if pt.isinf(o).any():
            warn(f"Found {pt.isinf(o).sum().item()} infs in output of{module},clamping")
            new_o.append(pt.clamp(output, -1e20, 1e20))
        else:
            new_o.append(o)
    if pt.is_tensor(output):
        return new_o[0]
    else:
        return tuple(new_o)


def backward_trace_hook(module, grad_input, grad_output):
    _grad_input = grad_input
    if pt.is_tensor(grad_input):
        grad_input = [grad_input]
    if pt.is_tensor(grad_output):
        grad_input = [grad_output]
    for i in grad_input:
        if pt.is_tensor(i) and pt.isnan(i).any():
            # set_trace()
            print("Got a Nan value")
    for o in grad_output:
        if pt.is_tensor(i) and pt.isnan(o).any():
            # set_trace()
            print("Got a Nan value")
    return _grad_input


def backward_trace_hook_t(grad_input):
    _grad_input = grad_input
    if pt.is_tensor(grad_input):
        grad_input = [grad_input]
    for i in grad_input:
        if pt.is_tensor(i) and pt.isnan(i).any():
            print("Got a Nan value")
            # set_trace()
    return _grad_input

def tensor_backward_clean_hook(grad,name):
    nans=pt.isfinite(grad)==False
    if grad.any():
        grad[nans]=0.0
        warning(f"Replacing {len(nans.nonzero())} entries in {name}-grad with 0.0")



def backward_clean_hook(module, grad_input=None, grad_output=None):
    _grad_input = grad_input
    _grad_output = grad_output
    if pt.is_tensor(grad_input):
        grad_input = [grad_input]
    if pt.is_tensor(grad_output):
        grad_input = [grad_output]
    gi_out = []
    go_out = []
    for i in grad_input:
        if pt.is_tensor(i):
            in_nans = pt.isnan(i)
            if in_nans.any():
                i = i.clone()
                i[in_nans] = 0.0
        gi_out.append(i)
    if pt.is_tensor(_grad_input):
        gi_out = gi_out[0]
        return gi_out
    return tuple(gi_out)
