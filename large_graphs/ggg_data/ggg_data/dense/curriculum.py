from collections import defaultdict
from typing import List, Tuple, Union, Optional, Dict


import numpy as np
import torch as pt

class DegreeCurriculumScheduler:
    """
    Simple heuristic to reduce graph size, probably should do some spectral method to make sure that the graph stays connected,
     but that can be done after discussion with lucas
    """
    def __init__(self,schedule:List[Tuple[float,int]]):
        if schedule is None:
            self.dummy=True
        else:
            self.dummy=False
            percentages = pt.tensor([x[0] for x in schedule])
            assert (percentages.diff() >= 0).all(), "Percentages schould be nondecreasing"
            visits = pt.tensor([x[1] for x in schedule])
            assert (visits.diff() >= 0).all(), "Steps schould be nondecreasing"
            assert visits[0] == 0, "Need to define a starting percentage"
            assert percentages[-1] == 1.0, "Need to finish on 100%"
            self.visits:Dict[int,int]=defaultdict(lambda : 0)
            self.last_key:Dict[int,int]=defaultdict(lambda :0)
            self.schedule=schedule
            self.percentages=pt.tensor([x[0] for x in schedule])
            self.curriculum:Dict[int,Tuple[pt.Tensor,pt.Tensor,pt.Tensor]]=dict()
    def get_key(self,idx):
        visits=self.visits[idx]
        last_key=self.last_key[idx]
        # starting from the last key, check which key we should use for this idx
        key=last_key
        # on first visit, we want to use the first key, otherwise we search based of the last key
        # for the first key that we can NOT use, then return the previous one (which is the last one that we *can* use)
        for _,vs in self.schedule[last_key:]:
            key += 1
            if vs>=visits:
                self.last_key[idx]=key-1
                return key-1
        # finally, if we don't find the early return, just return the last key
        return key-1

    def __call__(self, X:pt.Tensor,A:pt.Tensor,n:int,idx:int):
        if self.dummy:
            return X,A,n
        else:
            if idx not in self.curriculum:
                self.curriculum[idx]=self.make_curriculum(X,A,n)
            key=self.get_key(idx)
            X,A,N= self.curriculum[idx]
            self.visits[idx]=self.visits[idx]+1
            return X[key],A[key],N[key]

    def make_curriculum(self, X:pt.Tensor,A:pt.Tensor,n:int)->Tuple[pt.Tensor,pt.Tensor,pt.Tensor]:
        N=(self.percentages*n).ceil().long()
        Nmin=N.min()
        assert Nmin>=1
        assert N.max()==n,f"{N} should contain {n}"
        revN=list(reversed(N.tolist()))
        reducedX=[X.clone()]
        reducedA=[A.clone()]
        degrees=A.sum(-1)
        # nonzero
        nonzero=degrees!=0.0
        assert nonzero.any()
        odd=(degrees%2!=0)*nonzero
        dmin=degrees[nonzero].flatten().min()
        dminodd=degrees[odd].flatten().min()
        currn=n
        while True:
            # once we have reached the minimal node numbers, stop
            if (degrees!=0.0).sum()<=Nmin or len(reducedA)==len(revN):
                break

            if odd.any():
                node_to_remove=(degrees==dminodd).float().argmax() # get one node with min odd degree (1,3 etc, anything that can't be a bridge easily)
            else:
                node_to_remove = (degrees == dmin).float().argmax()  # get one node with min degree (either 1 or dmin)
            Aold=A.clone()
            Xold=X.clone()
            # remove the node
            A[node_to_remove,:]=0.0
            A[:,node_to_remove] = 0.0
            X[node_to_remove,:]=0.0
            currn-=1
            # update the degrees
            degrees=A.sum(-1)
            # nonzero
            nonzero=degrees!=0.0
            # and dmin
            if not nonzero.any():
                # undo the removal, faking out further removals
                A=Aold
                X=Xold
                degrees = A.sum(-1)
                # nonzero
                nonzero = degrees != 0.0
                dmin=degrees[nonzero].flatten().min()
            else:
                dmin=degrees[nonzero].flatten().min()
            if currn==revN[len(reducedA)]:
                reducedA.append(A.clone())
                reducedX.append(X.clone())

        reducedA=list(reversed(reducedA))
        reducedX=list(reversed(reducedX))
        assert len(reducedA)==len(N)
        assert len(reducedX)==len(N)
        return pt.stack(reducedX),pt.stack(reducedA),N







