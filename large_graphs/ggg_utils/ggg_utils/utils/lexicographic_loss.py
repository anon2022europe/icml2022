from warnings import warn

import torch as pt
try:
    import torchsort
except:
    warn("Couldn't import torchsort, please install it with 'pip install torchsort' if you want to use the lexicographic regularizer")
def lexicographic_rank(X:pt.Tensor, d=0)->pt.Tensor:
    """
    Takes a 2D or 3D tensor, ranks the tensor in lexicographic order of the rows (last dim).
    :param X:
    :return:
    """
    if len(X.shape)==2:
        # do initial sort (according to github, argsort is stable by default)
        # extract slice on which we sort
        Xslice=X[:,0]
        s=pt.argsort(Xslice)
        si=pt.argsort(s)
        # if there are any consecutive zero diffs, we need to break the tie
        is_zero=pt.isclose(Xslice[s].diff(),pt.zeros((1),device=X.device).float()).float()
        # create the mask and inverse the permutation to select from the original array
        mask=pt.cat([pt.ones([1],device=X.device).bool(),is_zero],-1).bool()[si]
        if is_zero.any() and X.shape[-1]>1:
            # break the tie by descending one level as long as this makes sense (i.e. as long as the rere are dimensions to descend intoc
            ranks=lexicographic_rank(X[mask, 1:], d=d + 1) # descend onto reduced vec
            tie_breaker=pt.zeros_like(Xslice)
            eps=1/(10+ranks.shape[-1])
            tie_breaker[mask]=ranks.float().softmax(-1)*(eps) # tie breaker: downstream rank * small constant
            Xbroken=Xslice.clone()
            Xbroken[s]+=tie_breaker # add the tie braker to the original order
            s=pt.argsort(Xbroken) # calcualte tie broken lexicographic sort
        # after the ties are broken or if we hit recursion limit, either return the original sort or the tie broken one
        return s
    elif len(X.shape)==3:
        # TODO: vectorized impl
        return pt.stack([lexicographic_rank(x) for x in X], 0)
    else:
        raise ValueError("Can only do lexicographic ordering for 2 or 3 dim tensors atm")
def hacky_soft_lexicographic_rank(X:pt.Tensor, regularization="l2", regularization_strength=1.0)->pt.Tensor:
    # hack in tie break by simply adding an out-of gradient tracking constant (since the rounding is arbitrarily otherwise)
    # the broper way would be to fully batch the 3 dim case and then use a mix of the nondiff and softrank
    # losses to run
    X=X.clone()
    if X.dim()==2:
        slice=X[:,0]
        with pt.no_grad():
            eps = 1 / (slice.shape[-1] + 10)
            lr=lexicographic_rank(X)
            diff=slice[lr].diff()
            dmin=diff[diff!=0].abs().min(-1).values if (diff!=0).any() else slice.abs().min(-1).values
            tie_break=lr.float().softmax(-1)*dmin*eps
        slice=slice[lr]+tie_break
        slice=slice.permute(1,0) # torch sort works on the last dimension, we want to sort along the first
        ranks= torchsort.soft_rank(slice,regularization_strength=regularization_strength,regularization=regularization)
        ranks=ranks.permute(1,0)
    elif X.dim()==3:
        ranks=lexicographic_rank(X)
        slices=X[:,:,0]
        eps = 1 / (slices.shape[-1] + 10)
        broken_slices=[]
        for lr,slice in zip(ranks,slices):
            with pt.no_grad():
                diff = slice[lr].diff()
                dmin = diff[diff != 0].abs().min(-1).values if (diff!=0).any() else slice.abs().min(-1).values
                tie_break = lr.float().softmax(-1) * dmin * eps
            bslice=slice[lr] + tie_break
            broken_slices.append(bslice)
        slices=pt.stack(broken_slices,0).permute(1,0)
        ranks= torchsort.soft_rank(slices,regularization_strength=regularization_strength,regularization=regularization)
        ranks=ranks.permute(1,0)
    else:
        raise ValueError("Only supporting 2 and 3 dim tensors")
    return ranks




if __name__=="__main__":
    s1gt=pt.tensor([1,2,0])
    x1=pt.tensor([
        [0.5,0.8,0.3],  #1
        [0.5, 0.9, 0.3], # 2
        [0.5, 0.8, 0.2], # 0
    ])
    s1=lexicographic_rank(x1)
    assert (s1==s1gt).all(),f"{s1}!={s1gt}"
    s2gt = pt.tensor([2, 1, 0])
    x2 = pt.tensor([
        [0.3, 0.8, 0.3],  # 1
        [0.2, 0.9, 0.3],  # 2
        [0.1, 0.8, 0.2],  # 0
    ])
    s2 = lexicographic_rank(x2)
    assert (s2 == s2gt).all(), f"{s2}!={s2gt}"
    x3=pt.stack([x1,x2],0)
    s3gt=pt.stack([s1gt,s2gt],0)
    s3 = lexicographic_rank(x3)
    assert (s3 == s3gt).all(), f"{s3}!={s3gt}"
    x3.requires_grad=True
    sr3=hacky_soft_lexicographic_rank(x3)
    sr3.pow(2).sum().backward()
    print(x3.grad)


