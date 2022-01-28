"""
**KullbackR kernel**
    KullbackR()
With a nonnegative scalar `λ`, returns the function
```math
f(x) = b * ( log(b) - log(<a,x>) ) - b + <a , x> 
```
"""

export KullbackR, KullbackL, Kullback_sqr


struct KullbackR{R<:Real} <: BregmanFunction
    a::AbstractVector{R}
    b::R
end

function (f::KullbackR{R})(x::AbstractArray) where {R<:Real}
    inp = dot(f.a, x)
    return f.b * (log(f.b / inp) - 1) + inp
end

## gradient
# naive!
function gradient!(
    y::AbstractArray{T},
    f::KullbackR{R},
    x::AbstractArray{T},
) where {R<:Real,T<:RealOrComplex{R}}
    inp = dot(f.a, x)
    @. y = (1 - f.b / inp) * f.a
    return f.b * (log(f.b / inp) - 1) + inp
end

###### 

struct KullbackL{R<:Real} <: BregmanFunction
    a::AbstractVector{R}
    b::R
end


function (f::KullbackL{R})(x::AbstractArray) where {R<:Real}
    inp = dot(f.a, x)
    return f.b + (log(inp / f.b) - 1) * inp
end

function gradient!(
    y::AbstractArray{T},
    f::KullbackL{R},
    x::AbstractArray{T},
) where {R<:Real,T<:RealOrComplex{R}}
    inp = dot(f.a, x)
    @. y = (log(inp) - log(f.b)) * f.a
    return f.b + (log(inp / f.b) - 1) * inp
end


######


struct Kullback_sqr{R<:Real} <: BregmanFunction
    a::AbstractVector{R}
    b::R
end




function (f::Kullback_sqr{R})(x::AbstractArray) where {R<:Real}
    inp = abs2(dot(f.a, x))
    return f.b * ( log(f.b) - log(inp) -1) + inp
end

function gradient!(
    y::AbstractArray{T},
    f::Kullback_sqr{R},
    x::AbstractArray{T},
) where {R<:Real,T<:RealOrComplex{R}}

    inp = dot(f.a, x)
    @. y = 2 * (inp - f.b / inp) * f.a
    return f.b * ( log(f.b) - log(abs2(inp)) -1) + abs2(inp)
end
