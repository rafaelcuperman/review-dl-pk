import DifferentialEquations: DiscreteCallback
import Zygote.ChainRules: @ignore_derivatives


function dose_callback(I::AbstractMatrix; S1=1)
    #times_, doses_, rates_, durations_ = eachcol(I)
    
    times_, doses_ = eachcol(I)
    doses_ = doses_ .* S1
    durations_ = ones(length(doses_)) .* 1/120
    rates_ = doses_ ./ durations_

    times = Float32.([times_; times_ + durations_])
    rates = Float32.([rates_; 0])

    function condition(u, t, p) 
        return t ∈ times
    end

    function affect!(integrator)
        if !(integrator.t ∈ times) return end

        rate = rates[findfirst(isequal(integrator.t), times)]
        #println("Callback triggered at time $(integrator.t) with value $rate")

        @ignore_derivatives integrator.p[end, :] .= rate
    end
    return DiscreteCallback(condition, affect!; save_positions=(false, false))
end
