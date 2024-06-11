import Zygote: @ignore_derivatives
import Optimisers
import Zygote
import Plots
import CSV
import Random
import GLM
using Lux
using DataFrames
using ComponentArrays
using PartialFunctions
using SciMLSensitivity
using DifferentialEquations
using JLD2
using Printf


include("../../../DCM/src/lib/population.jl");
include("../../../DCM/src/lib/model.jl");
include("../../../DCM/src/lib/error.jl");
include("../../../DCM/src/lib/objectives.jl");
include("../../../DCM/src/lib/callbacks.jl");
include("../../../DCM/src/lib/lux.helpers.jl");

basic_tgrad(u,p,t) = zero(u)

# Read Data
file = "neural_network_comparison/data/fviii_sim.csv"
df = CSV.read(file, DataFrame)

# One dataframe per patient
df_group = groupby(df, :id);

# Create population
function create_population(df_group_inds)
    indvs = Vector{AbstractIndividual}(undef, length(df_group_inds))
    for (i, group) in enumerate(df_group_inds)
        x = Vector{Float32}(group[1, [:weight, :age]])
        ty = group[(group.mdv .== 0), [:t, :dv]]
        𝐈 = Matrix{Float32}(group[group.mdv .== 1, [:t, :amt, :rate, :duration]])
        callback = generate_dosing_callback(𝐈)
        indvs[i] = Individual(x, Float32.(ty.t), Float32.(ty.dv), callback; id = group.id[1])
    end
    return Population(indvs);
end
population = create_population(df_group);

# Create neural network architecture.
function dnn(latent_dim)
    return Chain(
        Normalize([ones(latent_dim); 70; 40]),
        Dense(latent_dim + size(population.x, 1), 16, swish),
        Dense(16, 16, swish),
        Dense(16, 8, swish),
        Dense(8, latent_dim)
    )
end

# The function received by ODEFunction must have the form f(u, p, t; kwargs...).
# As the neural network receives [u; x] es inputs, we need to wrap the neural network on top of this function with the correct form.
# Additionally, we add the intervention as another parameter (will be used for the callbacks)
dudt(u, p, t; x, node, st) = node([u; x], p.weights, st)[1] + [p.I, 0.f0]

# Defines a method for the dudt function that accepts kwargs and calls the original dudt function with those arguments.
# The $ syntax (comes from PartialFunctions) is used to partially apply the function with the provided kwargs
dudt(; kwargs...) = dudt$(; kwargs...)

# Forward pass for one individual (solve the ODE)
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

# Forward pass for population. The Ref function "freezes" the parameters with respect to the dot operator
forward(model, population::Population, p, st; kwargs...) = forward.((model,), population, (p,), (st,); kwargs...)

# Function that will be called. Uses the default parameters
forward_adjoint(model, container, p, st) = forward(model, container, p, st; get_dv = true, full=true, sensealg=InterpolatingAdjoint(; autojacvec = ReverseDiffVJP()))

# Loss function (SSE)
function obj(node, population, p, st)
    ŷ = forward_adjoint(node, population, p, st)
    dif = reduce(vcat, population.y - ŷ)
    return sum(abs2, dif)/length(dif)
end

# Loss function (MAE)
function mae(node, population, p, st)
    ŷ = forward_adjoint(node, population, p, st)
    dif = reduce(vcat, population.y - ŷ)
    return sum(abs, dif)/length(dif)
end

# Train
function train_model(population_train, num_epochs, p; population_test=nothing, verbose=true, patience=100, tol=0.01)
    global opt_state
    losses = []
    losses_test = []
    times = []

    count = 0
    for epoch in 1:num_epochs

        elapsed_time = @elapsed begin
            loss, back = Zygote.pullback(p_ -> obj(node, population_train, ComponentVector((weights = p_.weights, I = 0.f0)), st), p)
            grad = first(back(1))
            opt_state, p = Optimisers.update(opt_state, p, grad)
        end
        push!(times, elapsed_time)

        push!(losses, loss)
        if population_test !== nothing
            push!(losses_test, obj(node, population_test, ComponentVector((weights = p.weights, I = 0.f0)), st))
        end

        if (epoch == 1 || epoch % 1 == 0)  && verbose
            println("Epoch $epoch, loss = $loss")
        end

        if epoch == 500      
            Optimisers.adjust!(opt_state, 0.001)
        end

        # Early stopping on percentual change of test loss. The test population is needed
        if population_test !== nothing
            if (epoch > 1) && ((losses_test[end-1] - losses_test[end])/losses_test[end-1] < tol)
                count += 1
                if count == patience
                    println("Early stopped. Percentual test loss did not improve at least $tol for $patience epochs. Trained for $epoch epochs\n")
                    verbose && println("\nMean ± std training time per epoch: $(@sprintf("%.2e", mean(times))) ± $(@sprintf("%.2e", std(times))) seconds")
                    return model, losses, losses_test
                end
            else 
                count = 0
            end
        end

    end
    verbose && println("\nMean ± std training time per epoch: $(@sprintf("%.2e", mean(times))) ± $(@sprintf("%.2e", std(times))) seconds")
    return p, losses, losses_test
end

# Plot predictions vs real values
function plot_predictions_real(node, p, st, population; threshold=0.2)
    real = reduce(vcat, population.y)
    preds = forward(node, population, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)
    predicted = reduce(vcat, map(x -> x.u, preds))

    plt = Plots.scatter(
        real,
        predicted,
        xlabel="True",
        ylabel="Predicted",
        markersize=2,
        markercolor=:blue,
        size=(600,400),
        legend=false
    )

    lr = GLM.lm(GLM.@formula(y ~ X), DataFrame(X=Float64.(real), y=Float64.(predicted)))

    accuracy = count(x -> x < threshold, abs.((real - predicted)) ./ (real .+ 1e-6))/length(real)
    max_value = ceil(maximum(vcat(real, predicted)))
    Plots.plot!(plt, 
        [0, max_value], 
        [0, max_value],
        #ribbon = [0, max_value] .* threshold,
        fillalpha = 0.2,
        #color = "red",
        linestyle = :dash,
        
        )

    regression = GLM.predict(lr, DataFrame(X=[0, max_value]))
    rsquared = GLM.r2(lr)

    Plots.plot!(plt, 
        [0, max_value], 
        regression,
        color = "blue",
        title = "R2 = $(@sprintf("%.2f", rsquared))"
        )
    display(plt)
end

function training_metrics(node, p, st, population)
    #MAE
    mean_absolute_error = mae(node, population, ComponentVector((weights = p.weights, I = 0.f0)), st)

    #R2
    real = reduce(vcat, population.y)
    preds = forward(node, population, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)
    predicted = reduce(vcat, map(x -> x.u, preds))
    lr = GLM.lm(GLM.@formula(y ~ X), DataFrame(X=Float64.(real), y=Float64.(predicted)))
    rsquared = GLM.r2(lr)

    return mean_absolute_error, rsquared
end


# Plot predictions vs real values
function plot_train_test_real(node, p, st, population_train, population_test)
    real = reduce(vcat, population_train.y)
    preds = forward(node, population_train, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)
    predicted = reduce(vcat, map(x -> x.u, preds))

    plt = Plots.scatter(
        real, 
        predicted,
        xlabel="True",
        ylabel="Predicted",
        markersize=2,
        markercolor=:blue,
        size=(600,400),
        label="Train"
    )

    real = reduce(vcat, population_test.y)
    preds = forward(node, population_test, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)
    predicted = reduce(vcat, map(x -> x.u, preds))

    Plots.scatter!(
        real, 
        predicted,
        xlabel="True",
        ylabel="Predicted",
        markersize=2,
        markercolor=:red,
        size=(600,400),
        label="Test"
    )

    max_value = ceil(maximum(reduce(vcat, population_train.y)))
    Plots.plot!(plt, 
        [0, max_value], 
        [0, max_value],
        linestyle = :dash,
        color=:black,
        label=nothing
        )
    display(plt)
end




latent_dim = 2
node = dnn(latent_dim)
ps, st = Lux.setup(Random.default_rng(), node)
p_ = ComponentVector((weights = ps, I = 0.f0));
p = (weights = ps,)
opt = Optimisers.Adam(0.01)
opt_state = Optimisers.setup(opt, p)
obj(node, population, p_, st)



num_epochs = 200
p_final, losses, _ = train_model(population, num_epochs, p; population_test = nothing)
println()
final_mse = losses[end]
println("Final MSE: $final_mse")
final_mae, final_r2 = training_metrics(node, p_final, st, population)
println("Final MAE: $final_mae")
println("Final R2: $final_r2")

plt = Plots.plot(losses, label = "Loss train")
display(plt)

plot_predictions_real(node, p_final, st, population)


# Predict a single individual. The predicted values are only saved for the individual.t times
function predict_individual(individual, p; plot_predictions=false)

    predicted = forward(node, individual, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)

    if plot_predictions
        plt = Plots.plot(predicted.t, predicted.u, 
                    label="Predicted", xlabel="Time", ylabel="Concentration",
                    color = :black,
                    size=(600,300))
        display(plt)
    end
    return predicted
end


# Predict a single real individual with labels. The predicted values are saved for all the times between 0 and max_time but the real labels are plotted for the real times
function predict_real_individual(individual, p; max_time = 120, plot_predictions = false, plot_true_labels = false)

    time_real = individual.t
    real_y = individual.y

    time = collect(0:0.1:(max_time-1))
    subject = Individual(individual.x, Float32.(time), Float32.(zeros(length(time))), individual.callback)

    predicted = forward(node, subject, ComponentVector(merge(p, (I = 0.f0, ))), st; interpolate = false)

    if plot_predictions
        plt = Plots.plot(predicted.t, predicted.u, 
                    label="Predicted", xlabel="Time", ylabel="Concentration",
                    color = :black,
                    size=(600,300))
        if plot_true_labels
            Plots.scatter!(plt, time_real, real_y, label="Real", color=:blue)
        end
        display(plt)
    end
    return predicted, time
end

# Create Intervention Matrix from [(time1, dose1), (time2, dose2), ...] list
function create_intervention_matrix(dose_list; duration = 120)
    if length(dose_list) == 0 return end
    M = []
    for (t, d) in dose_list
        row = [t, d, d * duration, 1/duration]
        push!(M, row)
    end

    # Convert the array of arrays to a matrix
    M = hcat(M...)
    M = M'
end


predict_real_individual(population[1], p_final; max_time = 72, plot_predictions = true, plot_true_labels = true);

jldsave("neural_network_comparison/models/NeuralODE/models/mymodel-fviii-6.jld2"; p_final, node, st)