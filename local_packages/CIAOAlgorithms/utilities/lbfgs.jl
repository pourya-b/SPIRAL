using LinearAlgebra

mutable struct LBFGS{R<:Real,C<:Union{R,Complex{R}},I<:Integer,T<:AbstractArray{C},M}
    currmem::I # how many inputs are in the memory
    curridx::I # the index of the newest input in the memory
    s::T
    y::T
    s_M::Vector{T}
    y_M::Vector{T}
    ys_M::Vector{R} # vector of <s,y> s
    alphas::Vector{R}
    H::R # H0 in the algorithm 8 of L04_Newton lecture notes
end

function LBFGS(
    x::T,
    M::I,
) where {R<:Real,C<:Union{R,Complex{R}},I<:Integer,T<:AbstractArray{C}}
    s_M = [zero(x) for i = 1:M]
    y_M = [zero(x) for i = 1:M]
    s = zero(x)
    y = zero(x)
    ys_M = zeros(R, M)
    alphas = zeros(R, M)
    LBFGS{R,C,I,T,M}(0, 0, s, y, s_M, y_M, ys_M, alphas, one(R))
end

function update!(L::LBFGS{R,C,I,T,M}, s, y) where {R,C,I,T,M}
    L.s .= s
    L.y .= y
    ys = real(dot(L.s, L.y))
    if ys > 0
        L.curridx += 1
        if L.curridx > M
            L.curridx = 1
        end
        L.currmem += 1
        if L.currmem > M
            L.currmem = M
        end
        L.ys_M[L.curridx] = ys
        copyto!(L.s_M[L.curridx], L.s)
        copyto!(L.y_M[L.curridx], L.y)
        yty = real(dot(L.y, L.y))
        L.H = ys / yty # initilization of H0 with matrix ys/yty I
    end
    return L
end

function reset!(L::LBFGS{R,C,I,T,M}) where {R,C,I,T,M}
    L.currmem, L.curridx = zero(I), zero(I)
    L.H = one(R)
end

import Base: *

function (*)(L::LBFGS, v)
    w = similar(v)
    mul!(w, L, v)
end

# Two-loop recursion

import LinearAlgebra: mul!

function mul!(d::T, L::LBFGS{R,C,I,T,M}, v::T) where {R,C,I,T,M}
    d .= v
    idx = loop1!(d, L)
    d .*= L.H
    d = loop2!(d, idx, L)
    return d
end

function loop1!(d::T, L::LBFGS{R,C,I,T,M}) where {R,C,I,T,M}
    idx = L.curridx
    for i = 1:L.currmem
        L.alphas[idx] = real(dot(L.s_M[idx], d)) / L.ys_M[idx]
        d .-= L.alphas[idx] .* L.y_M[idx]
        idx -= 1
        if idx == 0
            idx = M
        end
    end
    return idx
end

function loop2!(d::T, idx::Int, L::LBFGS{R,C,I,T,M}) where {R,C,I,T,M}
    for i = 1:L.currmem
        idx += 1
        if idx > M
            idx = 1
        end
        beta = real(dot(L.y_M[idx], d)) / L.ys_M[idx]
        d .+= (L.alphas[idx] - beta) .* L.s_M[idx]
    end
    return d
end