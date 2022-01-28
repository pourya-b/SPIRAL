"""
**Burg kernel**
    Burg(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)
```
"""


export prox_Breg, prox_Breg!, gradient, gradient!

abstract type BregmanFunction end


function prox_Breg(
    f::Array{Th},
    g::ProximableFunction,
    x::AbstractArray{T},
    γ::Array{R},
) where {R<:Real,T<:RealOrComplex{R}, Th <: BregmanFunction}
    y = similar(x)
    fy = prox_Breg!(y, f, g, x, γ)   ## fix it if we want different H
    return y, fy
end


function prox_Breg(
    f::Array{Th},
    g::ProximableFunction,
    x::AbstractArray{T},
    γ::R,
) where {R<:Real,T<:RealOrComplex{R}, Th <: BregmanFunction}
    y = similar(x)
    fy = prox_Breg!(y, f, g, x, γ)   ## fix it if we want different H
    return y, fy
end


function prox_Breg(
    f::BregmanFunction,
    g::ProximableFunction,
    x::AbstractArray{T},
    γ::Array{R},
) where {R<:Real,T<:RealOrComplex{R}}
    y = similar(x)
    fy = prox_Breg!(y, f, g, x, γ)   ## fix it if we want different H
    return y, fy
end

function prox_Breg(
    f::BregmanFunction,
    g::ProximableFunction,
    x::AbstractArray{T},
    γ::R,
) where {R<:Real,T<:RealOrComplex{R}}
    y = similar(x)
    fy = prox_Breg!(y, f, g, x, γ)   ## fix it if we want different H
    return y, fy
end


function gradient(f::BregmanFunction, x)
    y = similar(x)
    gradient!(y, f, x)
    fx = gradient!(y, f, x)
    return y, fx
end

include("Burg_stcvx.jl")
include("Burg.jl")
include("Bultzmann_Shannon.jl")
include("Canonical.jl")
include("Poly42.jl")
include("KL.jl")
include("Quartic.jl")


# what about having different h_i? 

# prox is not returning function value for now!  
