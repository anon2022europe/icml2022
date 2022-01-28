"""
    disc_penalty_mode = attr.ib(
        default="interpolate_Adj",
        validator=attr.validators.in_(
            {
                "avg_grads",
                "interpolate_Adj",
                "interpolate_emebbeddings",
                "GNN_layerwise_penatlty",
            }
        ),
    )
"""
