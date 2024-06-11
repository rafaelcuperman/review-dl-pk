################################################################################
##########                                                            ##########
##########                       Normalize layer                      ##########
##########                                                            ##########
################################################################################

struct Normalize <: Lux.AbstractExplicitLayer
    lb::Vector{Float32}
    ub::Vector{Float32}
    Normalize(lb::T, ub::T) where T<:Vector{<:Real} = new(Float32.(lb), Float32.(ub))
    Normalize(ub::T) where T<:Vector{<:Real} = new(zeros(Float32, length(ub)), Float32.(ub))
end
Normalize(lb::Real, ub::Real) = Normalize(Float32[lb], Float32[ub])
Normalize(ub::Real) = Normalize(Float32[ub])

Lux.initialparameters(::Random.AbstractRNG, ::Normalize) = NamedTuple()
Lux.initialstates(::Random.AbstractRNG, l::Normalize) = (lb=l.lb, ub=l.ub)

Lux.parameterlength(::Normalize) = 0
Lux.statelength(l::Normalize) = 2 * length(l.ub)  # is this correct?

function (l::Normalize)(x::AbstractArray, ps, st::NamedTuple)
    y = (x .- st.lb) ./ (st.ub - st.lb)
    return y, st
end

################################################################################
##########                                                            ##########
##########                 Add global parameter layer                 ##########
##########                                                            ##########
################################################################################


struct AddGlobalParameters{T, F1, F2} <: Lux.AbstractExplicitLayer
    theta_dim::Int
    out_dim::Int
    locations::AbstractVector{Int}
    init_theta::F1
    activation::F2
end

AddGlobalParameters(out_dim, loc, T=Float32; init_theta=Lux.glorot_uniform, activation=softplus) = AddGlobalParameters{T, typeof(init_theta), typeof(activation)}(length(loc), out_dim, loc, init_theta, activation)

Lux.initialparameters(rng::Random.AbstractRNG, l::AddGlobalParameters) = (theta = l.init_theta(rng, l.theta_dim, 1),)
Lux.initialstates(rng::Random.AbstractRNG, l::AddGlobalParameters{T,F1,F2}) where {T,F1,F2} = (indicator_theta = indicator(l.out_dim, l.locations, T), indicator_x = indicator(l.out_dim, (1:l.out_dim)[Not(l.locations)], T))
Lux.parameterlength(l::AddGlobalParameters) = l.theta_dim
Lux.statelength(::AddGlobalParameters) = 2

# the indicators should be in the state!
function (l::AddGlobalParameters)(x::AbstractMatrix, ps, st::NamedTuple)
    if size(st.indicator_x, 2) !== size(x, 1)
        indicator_x = st.indicator_x * st.indicator_x' # Or we simply do not do this, the one might already be in the correct place following the combine function.
    else
        indicator_x = st.indicator_x
    end
    y = indicator_x * x + st.indicator_theta * repeat(l.activation.(ps.theta), 1, size(x, 2))
    return y, st
end


################################################################################
##########                                                            ##########
##########                  Combine parameters layer                  ##########
##########                                                            ##########
################################################################################

struct Combine{T1, T2} <: Lux.AbstractExplicitLayer
    out_dim::Int
    pairs::T2
end

function Combine(pairs::Vararg{Pair}; T=Float32)
    out_dim = maximum([maximum(pairs[i].second) for i in eachindex(pairs)])
    return Combine{T, typeof(pairs)}(out_dim, pairs)
end

function get_state(l::Combine{T1, T2}) where {T1, T2}
    indicators = Vector{Matrix{T1}}(undef, length(l.pairs))
    negatives = Vector{Vector{T1}}(undef, length(l.pairs))
    for pair in l.pairs
        Iₛ = indicator(l.out_dim, pair.second, T1)
        indicators[pair.first] = Iₛ
        negatives[pair.first] = abs.(vec(sum(Iₛ, dims=2)) .- one(T1))
    end
    return (indicators = indicators, negatives = negatives)
end

Lux.initialparameters(rng::Random.AbstractRNG, ::Combine) = NamedTuple()
Lux.initialstates(rng::Random.AbstractRNG, l::Combine) = get_state(l)
Lux.parameterlength(::Combine) = 0
Lux.statelength(::Combine) = 2

function (l::Combine)(x::Tuple, ps, st::NamedTuple) 
    indicators = @ignore_derivatives st.indicators
    negatives = @ignore_derivatives st.negatives
    y = reduce(.*, _combine.(x, indicators, negatives))
    return y, st
end

_combine(x::AbstractMatrix, indicator::AbstractMatrix, negative::AbstractVector) = indicator * x .+ negative

################################################################################
##########                                                            ##########
##########                    Interpretable layers                    ##########
##########                                                            ##########
################################################################################

function SingleHeadedBranch(covariate_idx::Int, neurons::Int; activation = swish, init_bias = Lux.ones32) 
    return make_branch(
        covariate_idx, 
        Lux.Dense(1, neurons, activation), 
        Lux.Dense(neurons, 1, softplus, init_bias=init_bias)
    )
end

function MultiHeadedBranch(covariate_idx::Int, neurons::Int, heads::Int; activation = swish, init_bias = Lux.ones32)
    if heads == 1
        head = Lux.Dense(neurons, 1, softplus, init_bias=init_bias)
    else
        head = Lux.Parallel(vcat, 
            [Lux.Dense(neurons, 1, Lux.softplus, init_bias=init_bias) for _ in 1:heads]...
        )        
    end
    return make_branch(
        covariate_idx, 
        Lux.Dense(1, neurons, activation), 
        head
    )
end

function make_branch(covariate_idx::Int, layers...)
    return Lux.Chain(
        Lux.SelectDim(1, covariate_idx),
        Lux.ReshapeLayer((1,)),
        layers...
    )
end

function interpret_branch(model::LuxCore.AbstractExplicitContainerLayer, branch_idx::Int)
    model.layers
end

#ps,st = Lux.setup(Random.default_rng(), model)
