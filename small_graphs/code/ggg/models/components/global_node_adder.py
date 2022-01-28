import torch as pt


class GlobalNodeAdder(pt.nn.Module):
    def forward(self, X, A):
        B, N, F = X.shape
        all_edges = pt.ones(B, 1, N).type_as(A)
        Ag = pt.cat([A, all_edges], 1)
        all_edges = pt.ones(B, N + 1, 1).type_as(A)
        Ag = pt.cat([Ag, all_edges], -1)
        Ag[:, -1, -1] = 0.0
        assert Ag.shape[-1] == Ag.shape[-2] == (N + 1)
        not_global = pt.zeros(B, N, 1).type_as(X)
        Xg = pt.cat([X, not_global], -1)
        yes_global = pt.zeros(B, 1, F + 1).type_as(X)
        yes_global[:, :, -1] = 1.0
        Xg = pt.cat([Xg, yes_global], -2)
        assert Xg.shape[-1] == (F + 1)
        assert Xg.shape[-2] == (N + 1)
        assert Xg.shape[-2] == Ag.shape[-1] == Ag.shape[-2]
        return Xg, Ag


if __name__ == "__main__":
    ga = GlobalNodeAdder()
    A = pt.randn(1, 5, 5)
    x = pt.randn(1, 5, 3)
    xg, Ag = ga(x, A)
    print(xg, Ag)
    print(xg.shape, Ag.shape)
