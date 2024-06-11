basic_tgrad(u,p,t) = zero(u)

struct LatentEncoderDecoder <: Lux.AbstractExplicitContainerLayer{(:encoder,:node,:decoder)} # Lu et al
    encoder
    node
    decoder
end

"""This version only takes a single sample from the VAE, there should be a specific version for the ELBO."""
# function forward(m::CustomLuxModel{O,M,P}, population::Population, p_::ComponentArray, st; kwargs...) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
function forward(m::LatentEncoderDecoder, population::Population, p::NamedTuple, st; kwargs...)
    p_node = ComponentVector((weights = p.weights.node, I = 0.f0,)) # Only using a ComponentVector for the NODE weights saves 0.1M allocations
    𝜙, _ = m.encoder(population.x, p.weights.encoder, st.encoder)
    k = Integer(size(𝜙, 1) / 2)
    z₀ = 𝜙[1:k, :] + randn(eltype(𝜙), k, size(𝜙, 2)) .* softplus.(𝜙[k+1:end, :]) # How are we going to do muliple samples?
    zₜ = forward_ode.((m,), population, eachcol(z₀), (p_node,), (st,); kwargs...)
    ŷ = first.(m.decoder.(zₜ, (p.weights.decoder,), (st.decoder,)))
    return getindex.(ŷ, 1, :) # decoder outputs matrices, so need to convert to vectors
end

forward_adjoint(m::LatentEncoderDecoder, args...) = forward(m, args...; sensealg = InterpolatingAdjoint(; autojacvec = ReverseDiffVJP())) 

"""Ideally this is re-usable for all CustomLuxModels based on NeuralODEs"""
# function forward_ode(m::CustomLuxModel{O,M,P}, individual::AbstractIndividual, z0, p, st; get_z = false, saveat = is_timevariable(individual) ? individual.t.y : individual.t, interpolate=false, sensealg=nothing) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
function forward_ode(m, individual::AbstractIndividual, z0, p, st; get_z = true, saveat = is_timevariable(individual) ? individual.t.y : individual.t, interpolate=false, sensealg=nothing)
    @ignore_derivatives p.I = zero(p.I)
    saveat_ = interpolate ? empty(saveat) : saveat
    node = StatefulLuxLayer(m.node, nothing, st.node)
    dzdt(z, p, t; model=node) = model(z, p.weights) .+ p.I
    
    ff = ODEFunction{false}(dzdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, Vector(z0), (-0.1f0, maximum(saveat)), p)
    
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        saveat = saveat_, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return get_z ? reduce(hcat, sol.u) : sol
end