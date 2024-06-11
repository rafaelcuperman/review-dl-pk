using Lux




model = Lux.Chain(
    Normalize([150.f0, 3.5f0]),
    Lux.BranchLayer(
        Lux.Chain(
            Lux.SelectDim(1, 1),
            Lux.ReshapeLayer((1,)),
            Lux.Dense(1, 12, Lux.swish), # was smooth_relu
            Lux.Parallel(vcat, 
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32),
                Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
            )
        ),
        Lux.Chain(
            Lux.SelectDim(1, 2),
            Lux.ReshapeLayer((1,)),
            Lux.Dense(1, 12, Lux.swish),
            Lux.Dense(12, 1, Lux.softplus, init_bias=Lux.ones32)
        )
    ),
    Combine(1 => [1, 2], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)

model2 = Lux.Chain(
    Normalize([150.f0, 3.5f0]),
    Lux.BranchLayer(
        MultiHeadedBranch(1, 12, 2),
        SingleHeadedBranch(2, 12),
    ),
    Combine(1 => [1, 2], 2 => [1]), # Join tuples
    AddGlobalParameters(4, [3, 4]; activation=Lux.softplus)
)