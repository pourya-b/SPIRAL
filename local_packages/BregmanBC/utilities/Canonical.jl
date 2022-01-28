"""
**Canonical kernel**
    Canonical(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)
```
"""

export Canonical, gradient!


struct Canonical{R<:Real} <: BregmanFunction
    λ::R
    function Canonical{R}(λ::R) where {R<:Real}
        if any(λ .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(λ)
        end
    end
end

Canonical(λ::R = 1.0) where {R<:Real} = Canonical{R}(λ)

## gradient
function gradient!(
    y::AbstractArray{T},
    f::Canonical{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    v = R(0)
    for k in eachindex(x)
        y[k] = f.λ * x[k]
        v += x[k]^2
    end
    return f.λ / R(2) * v
end

### proximal map 


function prox_Breg!(
    y::AbstractArray{R},
    f::Array{Canonical},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    # for k in eachindex(x)
    #     y[k] = γb ./ (g.lambda - x[k])
    # end
    ProximalOperators.prox!(y, g, γb * x, γb)
    nothing
end


#### adding gradient from proximaloperators just for test! 


function gradient!(
    y::AbstractArray{T},
    f::ProximalOperators.LeastSquares,
    x::AbstractArray{T},
) where {R<:Real,T<:RealOrComplex{R}}
    fx = ProximalOperators.gradient!(y, f, x)
    return fx
end


function gradient(f::ProximalOperators.LeastSquares, x)
    y = similar(x)
    gradient!(y, f, x)
    fx = gradient!(y, f, x)
    return y, fx
end

function polycall(
    h::Canonical{R}, # h is considered here for other future methods
    x::AbstractArray{R},
) where {R<:Real}
    return 0.5 * norm(x)^2
end

function dist_Breg(
    h::Canonical{R},
    x::AbstractArray{R},
    y::AbstractArray{R},
) where {R<:Real}
    grad = similar(x)
    gradient!(grad, h, y)

    d = polycall(h,x) - polycall(h,y) - dot(grad, x-y)
    # bug prone! for dist_Breg, in-place function returns error, I opted to return the result value d
    return d
end

# TODO: check carefully if h_i are different! 
