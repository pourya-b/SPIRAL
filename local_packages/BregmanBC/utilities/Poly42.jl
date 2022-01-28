"""
**Poly42 kernel**
    Poly42(λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = - λ ∑_i log(x_i)???!!!
```
"""

export Poly42


struct Poly42{R<:Real} <: BregmanFunction
    λ::R
    function Poly42{R}(λ::R) where {R<:Real}
        if any(λ .< 0)
            error("coefficients in λ must be nonnegative")
        else
            new(λ)
        end
    end
end

Poly42(λ::R = 1.0) where {R<:Real} = Poly42{R}(λ)

function (Poly42{R})(
    x::AbstractArray{T},
    ) where {R<:Real,T<:RealOrComplex{R}}
        return 0.25 * norm(x)^4 + 0.5 * norm(x)^2
end 

## gradient
function gradient!(
    y::AbstractArray{T},
    h::Poly42{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    normx = norm(x)^2
    for k in eachindex(x)
        y[k] = h.λ * (normx + 1) * x[k]
    end
    return normx / 2 * (1 + normx / 2) * h.λ
end

### proximal map 


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Poly42},Poly42},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    prox!(y, g, x .* γb, γb)

    if norm(y) > 0 
        p = 1 / norm(y)^2
        q = sqrt(p^2 / 4 + p^3 / 27)
        tstar = cbrt(p / 2 + q) + cbrt(p / 2 - q) # according to Cardano's cubic solution
        y .*= tstar
    end
    nothing
end


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Poly42},Poly42},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    prox!(y, g, x .* γ, γ)

    if norm(y) > 0 
        p = 1 / norm(y)^2
        q = sqrt((p^2 / 4) + (p^3 / 27))
        tstar = cbrt(p / 2 + q) + cbrt(p / 2 - q) # according to Cardano's cubic solution
        y .*= tstar
    end
    nothing
end



function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Poly42},Poly42},
    g::ProximalOperators.IndBallL0,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    γb = 1 / sum(1 ./ γ)
    prox!(y, g, x .* γb, γb)
        
    q = norm(y)
    if q > 0 
        c =  sqrt( q^2/4 + 1/27 )    
        tstar = cbrt(-q/2 + c) + cbrt(-q/2 - c) # according to Cardano's cubic solution
        y .*= - tstar / q 
    end
    nothing
end


function prox_Breg!(
    y::AbstractArray{R},
    h::Union{Array{Poly42},Poly42},
    g::ProximalOperators.IndBallL0,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    prox!(y, g, x .* γ, γ)

    q = norm(y)
    if q > 0 
        c =  sqrt( q^2/4 + 1/27 )    
        tstar = cbrt(-q/2 + c) + cbrt(-q/2 - c) # according to Cardano's cubic solution
        y .*= - tstar / q 
    end
    nothing
end

function polycall(
    h::Poly42{R}, # h is considered here for other future methods
    x::AbstractArray{R},
) where {R<:Real}
    return 0.25 * norm(x)^4 + 0.5 * norm(x)^2
end

function dist_Breg(
    h::Poly42{R},
    x::AbstractArray{R},
    y::AbstractArray{R},
) where {R<:Real}
    grad = similar(x)
    gradient!(grad, h, y)

    d = polycall(h,x) - polycall(h,y) - dot(grad, x-y)
    # bug prone! for dist_Breg, in-place function returns error, I opted to return the result value d
    return d
end






# fix the many gamma case better
# f.lambda is not incorporated in proxes! BUG PRONE! 
