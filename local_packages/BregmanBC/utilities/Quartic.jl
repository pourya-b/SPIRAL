"""
**quartic kernel**
    quartic()
With a nonnegative scalar `λ`, returns the function
```math
f(x) = b * ( log(b) - log(<a,x>) ) - b + <a , x> 
```
"""

export Quartic


struct Quartic{R<:Real} <: BregmanFunction
    a::AbstractVector{R}
    b::R
end

function (f::Quartic{R})(x::AbstractArray) where {R<:Real}
    inp = dot(f.a, x)
    return (inp^2 - f.b)^2 / 4
end

## gradient
# naive!
function gradient!(
    y::AbstractArray{T},
    f::Quartic{R},
    x::AbstractArray{T},
) where {R<:Real,T<:RealOrComplex{R}}
    inp = dot(f.a, x)
    @. y = (inp^2 - f.b) * inp * f.a
    return (inp^2 - f.b)^2 / 4
end
