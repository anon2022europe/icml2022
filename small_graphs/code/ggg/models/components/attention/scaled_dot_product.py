import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from sparsemax import Sparsemax
import numpy as np


class MemSparsemax(Sparsemax):
    def forward(self, x):
        x_orig = x
        B_orig = x.shape[0]
        B = x.shape[0]
        while B >= 1:
            outs = []
            num_batches = int(np.ceil(B_orig / B))
            batches = [x[i * B : (i + 1) * B] for i in range(num_batches)]
            try:
                if len(batches) == 1:
                    out = super(MemSparsemax, self).forward(x)
                else:
                    out = torch.cat(
                        [super(MemSparsemax, self).forward(x) for x in batches]
                    )
                return out
            except RuntimeError:
                print(f"Batch size {B} caused memory error in sparsemax, trying {B//2}")
                B = B // 2
        raise RuntimeError("Can't make Sparsemax work due to memory limitations")


class ScaledDotProductAttention(nn.Module):
    def forward(
        self,
        query,
        key,
        value,
        mask=None,
        return_attention_and_scores=False,
        att_sig="sigmoid",
    ):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if att_sig == "sigmoid":
            attention = torch.nn.Sigmoid()(scores)
        elif att_sig == "softmax":
            attention = F.softmax(scores, dim=-1)
        elif att_sig == "sparsemax":
            attention = MemSparsemax(dim=-1)(scores)
        else:
            raise ValueError(f"Unkown att_sig {att_sig}")
        if return_attention_and_scores:
            return attention.matmul(value), attention, scores
        else:
            return attention.matmul(value)


if __name__ == "__main__":
    x = MemSparsemax(-1)
    print(x(torch.randn(3, 4)))
