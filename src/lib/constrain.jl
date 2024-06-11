import Zygote.ChainRules: @ignore_derivatives
import Random

softplus_inv(x::T) where {T<:Real} = log(exp(x) - one(T))

# TODO: write model specific constraints (the if statement here result in an Union)
function constrain(p_::NamedTuple)
    st = @ignore_derivatives p_.st
    p = (; p_.weights, st)
    if :error in keys(p_) # Constrain ErrorModel parameters
        p = merge(p, (error = merge(p_.error, (sigma = softplus.(p_.error.sigma),)),))
    end

    if :omega in keys(p_)
        ω = softplus.(p_.omega.var) # TODO: rename this to sigma or similar, e.g. (prior = (omega = ..., corr = ...), )
        C = inverse(Bijectors.VecCorrBijector())(p_.omega.corr)
        p = merge(p, (omega = Symmetric(ω .* C .* ω'),))
    end
    return p
end

constrain_phi(::Type{MeanField}, 𝜙::NamedTuple) = (mean = 𝜙.mean, sigma = softplus.(𝜙.sigma))

sigma_corr_to_L(sigma, corr) = sigma .* inverse(Bijectors.VecCholeskyBijector(:L))(corr).L
function constrain_phi(::Type{FullRank}, 𝜙::NamedTuple)
    σ = softplus.(𝜙.sigma)
    L = sigma_corr_to_L.(eachcol(σ), eachcol(𝜙.corr))
    return (mean = 𝜙.mean, L = L)
end
