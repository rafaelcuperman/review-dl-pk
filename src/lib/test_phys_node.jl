import Zygote: ignore_derivatives
import Optimisers
import Zygote
import Plots
import CSV

include("src/lib/constrain.jl");
include("src/lib/callbacks.jl");
include("src/lib/population.jl");

using Lux
using DataFrames
using ComponentArrays
using PartialFunctions
using SciMLSensitivity
using DifferentialEquations

basic_tgrad(u,p,t) = zero(u)

df = DataFrame(CSV.File("data/warfarin.csv"))
df_group = groupby(df, :ID)

indvs = Vector{AbstractIndividual}(undef, length(df_group))
for (i, group) in enumerate(df_group)
    x = Vector{Float32}(group[1, [:WEIGHT, :AGE, :SEX]])
    ty = group[(group.DVID .== 1) .& (group.MDV .== 0), [:TIME, :DV]]
    𝐈 = Matrix{Float32}(group[group.MDV .== 1, [:TIME, :DOSE, :RATE, :DURATION]])
    callback = generate_dosing_callback(𝐈)
    indvs[i] = Individual(x, Float32.(ty.TIME), Float32.(ty.DV), callback; id = group.ID[1])
end
population = Population(indvs)


##### Define model
latent_dim = 2

node = Chain(
    Normalize([ones(latent_dim) ; 150; 100; 1]),
    Dense(latent_dim + size(population.x, 1), 16, tanh),
    Dense(16, 16),
    Dense(16, 8),
    Dense(8, latent_dim, softplus)
)

dudt(u, p, t; x, node, st) = node([u; x], p.weights, st)[1] + [p.I, 0.f0]

function dudt(u, p, t; x, node, st)
    du, st = node([u; x], p.weights, st) 
    return [p.I - u[1] * du[1], u[1] * du[1] - du[2]]
end

(dudt)(; kwargs...) = dudt$(; kwargs...)

forward(model, population::Population, p, st; kwargs...) = forward.((model,), population, (p,), (st,); kwargs...)

forward_adjoint(model, container, p, st) = forward(model, container, p, st; get_dv = true, full=true, sensealg=InterpolatingAdjoint(; autojacvec = ReverseDiffVJP()))

function forward(model, individual::AbstractIndividual, p, st; dudt = dudt, get_dv::Bool=false, sensealg=nothing, full::Bool=false, interpolate::Bool=false, saveat_ = is_timevariable(individual) ? individual.t.y : individual.t)
    @ignore_derivatives p.I = zero(p.I)
    u0 = isempty(individual.initial) ? zeros(Float32, 2) : individual.initial
    saveat = interpolate ? empty(saveat_) : saveat_
    save_idxs = full ? (1:length(u0)) : 2
    
    f = dudt(; x = individual.x, node = model, st = st)
    ff = ODEFunction{false}(f; tgrad = basic_tgrad)
    prob = ODEProblem{false}(ff, u0, (-0.1f0, maximum(saveat_)), p)
    
    interpolate && (individual.callback.save_positions .= 1)
    sol = solve(prob, Tsit5(),
        save_idxs = save_idxs, saveat = saveat, callback=individual.callback, 
        tstops=individual.callback.condition.times, sensealg=sensealg
    )
    interpolate && (individual.callback.save_positions .= 0)
    return get_dv ? sol[2, :] : sol
end

ps, st = Lux.setup(Random.default_rng(), node)
p_ = ComponentVector((weights = ps, theta = [10.f0], I = 0.f0))

forward_adjoint(node, population, p_, st)

function obj(node, population, p, st)
    ŷ = forward_adjoint(node, population, p, st)
    return sum(abs2, reduce(vcat, population.y - ŷ))
end

p = (weights = ps,)
# p = (weights = ps, theta = [10.f0])
opt = Optimisers.Adam(1e-2)
opt_state = Optimisers.setup(opt, p)
for epoch in 1:500
    loss, back = Zygote.pullback(p_ -> obj(node, population, ComponentVector((weights = p_.weights, I = 0.f0)), st), p)
    # loss, back = Zygote.pullback(p_ -> obj(node, population, ComponentVector((weights = p_.weights, theta = p_.theta, I = 0.f0)), st), p)
    grad = first(back(1))
    println("Epoch $epoch, loss = $loss")
    opt_state, p = Optimisers.update(opt_state, p, grad)
end



sol = forward(node, population[2], ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = true)
Plots.plot(sol); Plots.scatter!(population[2].t, population[2].y)
