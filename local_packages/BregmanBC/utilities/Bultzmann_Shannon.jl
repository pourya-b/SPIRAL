"""
**Bultzmann-Shannon kernel**
    BltzSnn(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i x_i log(x_i)
```
"""

export BltzSnn


struct BltzSnn{R<:Real} <: BregmanFunction
    λ::R
    function BltzSnn{R}(λ::R) where {R<:Real}
        if any(λ .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(λ)
        end
    end
end

BltzSnn(λ::R = 1.0) where {R<:Real} = BltzSnn{R}(λ)

## gradient
function gradient!(
    y::AbstractArray{T},
    h::BltzSnn{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    v = R(0)
    for k in eachindex(x)
        y[k] = h.λ * (1 + log(x[k]))
        x[k] == R(0) ? () : (v += log(x[k]) * x[k])
    end
    return h.λ * v
end

### proximal map 


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{BltzSnn},BltzSnn},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    # γb = sum(1 ./ γ)
    # for k in eachindex(x)
    #     y[k] = exp( (x[k] - g.lambda - γb ) / γb )
    # end
    γb = 1 / sum(1 ./ γ)
    for k in eachindex(x)
        y[k] = exp(γb * (x[k] - g.lambda) - 1)
    end
    nothing
end

function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{BltzSnn},BltzSnn},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    for k in eachindex(x)
        y[k] = exp(γ * (x[k] - g.lambda) - 1)
    end
    nothing
end
