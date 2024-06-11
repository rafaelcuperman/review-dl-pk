import Optimisers
import Zygote
import CSV

include("src/lib/constrain.jl");
include("src/lib/callbacks.jl");
include("src/lib/population.jl");
include("src/lib/model.jl");
include("src/lib/error.jl");
include("src/lib/objectives.jl");
include("src/lib/latent_encoder_decoder.jl");

using Lux
using DataFrames
using ComponentArrays
using SciMLSensitivity
using DifferentialEquations

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
latent_dim = 3

encoder = Chain(
    Normalize([150, 100, 1]),
    Dense(size(population.x, 1), 16, swish),
    Dense(16, 8, swish),
    Dense(8, latent_dim * 2)
)

node = Chain(
    Dense(latent_dim, 12, tanh),
    Dense(12, 12),
    Dense(12, latent_dim)
)

decoder = Chain(
    Normalize(fill(100, latent_dim)),
    Dense(latent_dim, 16, swish),
    Dense(16, 1, softplus)
)

function obj(model, population, p, st)
    ŷ = forward_adjoint(model, population, p, st)
    return sum(abs2, reduce(vcat, population.y - ŷ))
end

model = LatentEncoderDecoder(encoder, node, decoder)
ps, st = Lux.setup(Random.default_rng(), model)
p = (weights = ps,)
opt = Optimisers.Adam(1e-2)
opt_state = Optimisers.setup(opt, p)

for epoch in 1:300
    loss, back = Zygote.pullback(p_ -> obj(model, population, p_, st), p)
    grad = first(back(1))
    println("Epoch $epoch, loss = $loss")
    opt_state, p = Optimisers.update(opt_state, p, grad)
end

p_node = ComponentVector((weights = p.weights.node, I = 0.f0,)) # Only using a ComponentVector for the NODE weights saves 0.1M allocations
𝜙, _ = model.encoder(population.x, p.weights.encoder, st.encoder)
k = Integer(size(𝜙, 1) / 2)
z₀ = 𝜙[1:k, :] + randn(eltype(𝜙), k, size(𝜙, 2)) .* softplus.(𝜙[k+1:end, :]) # How are we going to do muliple samples?

i = rand(1:length(population))
sol = forward_ode(model, population[i], z₀[:, i], p_node, st; get_z = false, interpolate = true)
Plots.plot(sol)

pred, _ = model.decoder(reduce(hcat, sol.u), p.weights.decoder, st.decoder)

Plots.plot(sol.t, pred[1, :])
Plots.scatter!(population[i].t, population[i].y)