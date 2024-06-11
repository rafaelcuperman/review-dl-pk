


basic_tgrad(u, p, t) = zero(u)

struct LatentEncoderDecoder <: AbstractExplicitContainerLayer{(:encoder,node,decoder)} # Lu et al
    encoder
    node
    decoder
    latent_dim
end

"""This version only takes a single sample from the VAE, there should be a specific version for the ELBO."""
function forward(m::CustomLuxModel{O,M,P}, population::Population, p_::ComponentArray) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
    p = ComponentVector(p_, I = 0.f0) # Add the dosing variable
    𝜙, _ = m.encoder(population.x, p.weights.encoder, p.st.encoder)
    k = Integer(size(𝜙, 1) / 2)
    z₀ = 𝜙[1:k, :] + randn(eltype(𝜙), k, size(𝜙, 2)) .* softplus.(𝜙[k+1:end, :]) # How are we going to do muliple samples?
    zₜ = forward_ode.((m,), population, eachcol(z₀), (p,))
    return m.decoder.(zₜ, (p.weights.decoder,), (p.st.decoder,))
end

"""Ideally this is re-usable for all CustomLuxModels based on NeuralODEs"""
function forward_ode(m::CustomLuxModel{O,M,P}, individual::AbstractIndividual, z0, p; get_z = false, saveat = is_timevariable(individual) ? individual.t.y : individual.t, interpolate=false, sensealg=nothing) where {O<:FixedObjective,M<:LatentEncoderDecoder,P}
    @ignore_derivatives p.I = zero(p.I)
    saveat_ = interpolate ? empty(saveat) : saveat
    
    dzdt(z, p, t; model=m.node, st=p.st.node) = model(z, p.node, st) .+ p.I
    ff = ODEFunction{false}(dzdt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, z0, (-0.1f0, maximum(saveat)), p)
    
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        saveat = saveat_, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return get_z ? sol.u : sol
end

struct LowDimensionalNODE # Bräm et al.
    node
end

struct HybridDCM # Somewhat inspired by Valderrama et al.
    # ann -> some of the declared parameters
    # node -> unknown model components
end


function (n::NeuralODE)(x, p, st)
    model = StatefulLuxLayer(n.model, nothing, st)

    dudt(u, p, t) = model(u, p)
    ff = ODEFunction{false}(dudt; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, x, n.tspan, p)

    return (
        solve(prob, n.args...;
            sensealg = InterpolatingAdjoint(; autojacvec = ZygoteVJP()), n.kwargs...),
        model.st)
end