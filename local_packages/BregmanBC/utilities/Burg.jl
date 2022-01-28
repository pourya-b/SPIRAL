"""
**Burg kernel**
    Burg(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)
```
"""

export Burg


struct Burg{R<:Real} <: BregmanFunction
    λ::R
    function Burg{R}(λ::R) where {R<:Real}
        if any(λ .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(λ)
        end
    end
end

Burg(λ::R = 1.0) where {R<:Real} = Burg{R}(λ)

## gradient
function gradient!(
    y::AbstractArray{T},
    h::Burg{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    v = R(0)
    for k in eachindex(x)
        y[k] = -h.λ / x[k]
        v += log(x[k])
    end
    return -h.λ * v
end

### proximal map 


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Burg},Burg},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = sum(1 ./ γ)
    for k in eachindex(x)
        y[k] = γb / (g.lambda - x[k])
    end
    nothing
end


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Burg},Burg},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    γb = 1 / γ
    for k in eachindex(x)
        y[k] = γb / (g.lambda - x[k])
    end
    nothing
end


function prox_Breg!(    # I have this separate for now so that single prox can also be computed!
    y::AbstractArray{R},
    h::Burg,
    g::ProximalOperators.SqrNormL2,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    for k in eachindex(x)
        if g.lambda == R(0)
            y[k] = -1 / (x[k] * γb)
        else
            ss = x[k] / (2 * g.lambda)
            y[k] = ss + sqrt(ss^2 + h.λ / (g.lambda * γb))
        end
    end
    nothing
end


function prox_Breg!(
    y::AbstractArray{R},
    h::Array{Burg},
    g::ProximalOperators.SqrNormL2,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    for k in eachindex(x)
        if g.lambda == R(0)
            y[k] = -1 / (x[k] * γb)
        else
            ss = x[k] / (2 * g.lambda)
            y[k] = ss + sqrt(ss^2 + h[1].λ / (g.lambda * γb))
        end
    end
    nothing
end



function prox_Breg!(
    y::AbstractArray{R},
    h::Burg,
    g::ProximalOperators.SqrNormL2,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    for k in eachindex(x)
        if g.lambda == R(0)
            y[k] = -1 / (x[k] * γ)
        else
            ss = x[k] / (2 * g.lambda)
            y[k] = ss + sqrt(ss^2 + h.λ / (g.lambda * γ))
        end
    end
    nothing
end



function prox_Breg!(
    y::AbstractArray{R},
    h::Array{Burg},
    g::ProximalOperators.SqrNormL2,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    for k in eachindex(x)
        if g.lambda == R(0)
            y[k] = -1 / (x[k] * γ)
        else
            ss = x[k] / (2 * g.lambda)
            y[k] = ss + sqrt(ss^2 + h[1].λ / (g.lambda * γ))
        end
    end
    nothing
end


# fix the many gamma case better
# f.lambda is not incorporated in proxes! BUG PRONE! 
# decide how to deploy h and gamma, array or not  