"""
**Burg_stcvx kernel**
    Burg_stcvx(c, b, λ=1.0)
With a nonnegative scalar `λ`, returns the function
```math
f(x) = λ |x|^2 * a  - 2λ b ∑_i log(x_i), where a = |a_i|^2 
```
"""

export Burg_stcvx


struct Burg_stcvx{R<:Real} <: BregmanFunction
    a::R    # \|a_i\|^2 
    b::R     # b_i
    λ::R
    function Burg_stcvx{R}(a::R, b::R, λ::R) where {R<:Real}
        if any(λ .< 0) || any(a .< 0)|| any(b .< 0)
            error("coefficients in λ and the values of a and b must be nonnegative")
        else
            new(a, b, λ)
        end
    end
end

Burg_stcvx(a::R, b::R, λ::R = 1.0) where {R<:Real} = Burg_stcvx{R}(a, b, λ)

## gradient
function gradient!(
    y::AbstractArray{T},
    h::Burg_stcvx{S},
    x::AbstractArray{T},
) where {S<:Real,R<:Real,T<:RealOrComplex{R}}
    v = R(0)
    for k in eachindex(x)
        y[k] = h.λ * 2 * (h.a * x[k] - h.b / x[k])
        v += h.λ * ( abs2(x[k]) * h.a   - 2 * h.b * log(x[k]) )
    end
    return  v
end

### proximal map 


function prox_Breg!(
    y::AbstractArray{R},
    h::Array{Burg_stcvx},
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::Array{R},
) where {R<:Real}
    

    # γb = 1 / sum(1 ./ γ)
    # cd = sum(h.a ./ γ)
    ca = 0.0 # compute sum |a_i|^2 / gamma_i
    cb = 0.0 # compute sum b_i / gamma_i
    for k in eachindex(h)
        ca += 4* h[k].a / γ[k]
        cb += 4* h[k].b / γ[k]
    end 
    # println(ca)
    # println(cb)
    for k in eachindex(x)
        temp =  x[k] - g.lambda
        y[k] = ( temp + sqrt( abs2(temp) + ca * cb) ) / ca
        # temp =  (x[k]-g.lambda) / (2* ca)
        # y[k] = temp + sqrt( abs2(temp) + 2 * cb / ca )
    end
    nothing
end




function prox_Breg!(
    y::AbstractArray{R},
    h::Burg_stcvx,
    g::ProximalOperators.NormL1,
    x::AbstractArray{R},
    γ::R,
) where {R<:Real}
    
    ca = 4 * h.a / γ       # h.a must be \sum |a_i|^2 /N
    cb = 4 * h.b / γ       # h.b must be  \sum b_i / N
    for k in eachindex(x)
        temp =  x[k]-g.lambda
        y[k] = ( temp + sqrt( abs2(temp) +  ca * cb) ) /  ca
        # temp =  (x[k]-g.lambda) / (2* ca)
        # y[k] = temp + sqrt( abs2(temp) + 2 * cb / ca )
    end
    nothing
end

# function prox_Breg!(
#     y::AbstractArray{R},
#     h::Array{Burg_stcvx},
#     g::ProximalOperators.NormL1,
#     x::AbstractArray{R},
#     γ::Array{R},
# ) where {R<:Real}
    

#     γb = 1 / sum(1 ./ γ)
#     # cd = sum(h.a ./ γ)
#     cd = 0.0 
#     for k in eachindex(h)
#         cd += h[k].c / γ[k]
#     end 

#     for k in eachindex(x)
#         temp =  (x[k]-g.lambda) / (2* cd)
#         y[k] = temp + sqrt( abs2(temp) + 1/ (cd * γb) )
#     end
#     nothing
# end




# function prox_Breg!(
#     y::AbstractArray{R},
#     h::Burg_stcvx,
#     g::ProximalOperators.NormL1,
#     x::AbstractArray{R},
#     γ::R,
# ) where {R<:Real}
    

#     cd = h.a / γ       # h.a must be \sum |a_i|^2 / \sum b_i
#     for k in eachindex(x)
#         temp =  (x[k]-g.lambda) / (2* cd)
#         y[k] = temp + sqrt( abs2(temp) + 1/ (cd * γ) )
#     end
#     nothing
# end




# fix the many gamma case better
# f.lambda is not incorporated in proxes! BUG PRONE! 
