using LinearAlgebra
using ProximalOperators
using ProximalAlgorithms.IterationTools
# using ProximalAlgorithms: LBFGS, update!, mul!
using Printf
using Base.Iterators # to use take and enumerate functions
using Random
using StatsBase: sample
using BregmanBC
# using Flux
import BregmanBC: gradient, gradient!
# import ProximalOperators: prox!, prox, gradient, gradient!
export solution, epoch_count
import ProximalOperators: gradient, gradient!

abstract type CIAO_iterable end #? what is this? where is used?


include("Finito_basic.jl")
include("Finito_LFinito.jl")
include("Finito_adaptive.jl")
include("Finito_LFinito_lbfgs.jl")
include("Finito_DLFinito.jl")
include("Finito_DLFinito_lbfgs.jl")
include("Finito_LFinito_lbfgs_adaptive.jl")
include("Finito_LFinito_lbfgs_adaptive_DNN.jl")



struct Finito{R<:Real}
    γ::Maybe{Union{Array{R},R}} #? this is defined in CIAOAlgorithms, where CIAOAlgorithms is called/used? #other questions in that file ...
    sweeping::Int8
    LFinito::Bool
    lbfgs::Bool
    memory::Int
    η::R
    β::R
    adaptive::Bool
    DeepLFinito::Tuple{Bool,Int, Int}
    minibatch::Tuple{Bool,Int}
    maxit::Int
    verbose::Bool
    freq::Int
    α::R
    tol::R
    tol_b::R
    DNN_training::Bool
    function Finito{R}(;
        γ::Maybe{Union{Array{R},R}} = nothing,
        sweeping = 1,
        LFinito::Bool = false,
        lbfgs::Bool = false,
        memory::Int = 6, # lbfgs memory
        η::R = 0.7,
        β::R = 1/50,
        adaptive::Bool = false,
        # DeepLFinito::Tuple{Bool,Int} = (false, 3),
        DeepLFinito::Tuple{Bool,Int, Int} = (false, 3, 3),
        minibatch::Tuple{Bool,Int} = (false, 1),
        maxit::Int = 10000,
        verbose::Bool = false,
        freq::Int = 10000,
        α::R = R(0.999), # R is type
        tol::R = R(1e-8),
        tol_b::R = R(1e-9),
        DNN_training::Bool = false,
    ) where {R}
        @assert γ === nothing || minimum(γ) > 0
        @assert maxit > 0
        @assert memory >= 0
        @assert tol > 0
        @assert tol_b > 0
        @assert freq > 0
        new(γ, sweeping, LFinito, lbfgs, memory, η, β, adaptive, DeepLFinito, minibatch, maxit, verbose, freq, α, tol, tol_b, DNN_training)
    end
end

function (solver::Finito{R})( # this is a function definition. if solver = Finito(; kwargs...), and then solver(x0;kwargs...) is called, this function will be executed.
    x0::AbstractArray{C};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
) where {R,C<:RealOrComplex{R}}

    stop(state) = false # the stopping function for halt function

    disp(it, state) = @printf "%5d | %.3e  \n" it state.hat_γ

    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    if solver.LFinito
        if solver.DeepLFinito[1]
            iter = FINITO_DLFinito_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                solver.DeepLFinito[2],
                solver.DeepLFinito[3],
            )
        else
            iter = FINITO_LFinito_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
            )
        end
    elseif solver.lbfgs
        if solver.DeepLFinito[1]
            iter = FINITO_DFlbfgs_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
            LBFGS(x0, solver.memory),
            solver.DeepLFinito[2],
            solver.DeepLFinito[3]
        )
        elseif solver.adaptive
            iter = FINITO_lbfgs_adaptive_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.adaptive
            )
        else
            iter = FINITO_lbfgs_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
            )
        end
    elseif solver.adaptive
        iter = FINITO_adaptive_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.tol,
            solver.tol_b,
            solver.sweeping,
            solver.α,
        )
    else
        iter = FINITO_basic_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end

    iter = halt(iter, stop) #? where is halt defined?
    iter = take(iter, solver.maxit)
    iter = enumerate(iter)

    num_iters, state_final = nothing, nothing
    for (it_, state_) in iter  # unrolling the iterator (acts as tee and loop functions in the tutorial)
        # see https://docs.julialang.org/en/v1/manual/interfaces/index.html
        if solver.verbose && mod(it_, solver.freq) == 0
            disp(it_, state_)
        end
        num_iters, state_final = it_, state_
    end
    if solver.verbose && mod(num_iters, solver.freq) !== 0
        disp(num_iters, state_final)
    end # for the final iteration
    return solution(state_final), num_iters
end

"""
    Finito([γ, sweeping, LFinito, adaptive, minibatch, maxit, verbose, freq, tol, tol_b])

Instantiate the Finito algorithm for solving fully nonconvex optimization problems of the form

    minimize 1/N sum_{i=1}^N f_i(x) + g(x)

where `f_i` are smooth and `g` is possibly nonsmooth, all of which may be nonconvex.

If `solver = Finito(args...)`, then the above problem is solved with

	solver(x0, [F, g, N, L])

where F is an array containing f_i's, x0 is the initial point, and L is an array of
smoothness moduli of f_i's; it is optional in the adaptive mode or if γ is provided.

Optional keyword arguments are:
* `γ`: an array of N stepsizes for each coordinate
* `sweeping::Int` 1 for uniform randomized (default), 2 for cyclic, 3 for shuffled
* `LFinito::Bool` low memory variant of the Finito/MISO algorithm
* `adaptive::Bool` to activate adaptive smoothness parameter computation
* `minibatch::(Bool,Int)` to use batchs of a given size
* `maxit::Integer` (default: `10000`), maximum number of iterations to perform.
* `verbose::Bool` (default: `true`), whether or not to print information during the iterations.
* `freq::Integer` (default: `10000`), frequency of verbosity.
* `α::R` parameter where γ_i = αN/L_i
* `tol::Real` (default: `1e-8`), absolute tolerance for the adaptive case
* `tol_b::R` tolerance for the backtrack (default: `1e-9`)
"""

Finito(::Type{R}; kwargs...) where {R} = Finito{R}(; kwargs...) #? outer constructor? why is needed? where it is used? Type{R}?
Finito(; kwargs...) = Finito(Float64; kwargs...)


"""
If `solver = Finito(args...)`, then

    itr = iterator(solver, x0, [F, g, N, L])

is an iterable object. Note that [maxit, verbose, freq] fields of the solver are ignored here.

The solution at any given state can be obtained using solution(state), e.g.,
for state in Iterators.take(itr, maxit)
    # do something using solution(state)
end

See https://docs.julialang.org/en/v1/manual/interfaces/index.html
and https://docs.julialang.org/en/v1/base/iterators/ for a list of iteration utilities
"""


function iterator( #? how it is called?
    solver::Finito{R},
    x0::Union{AbstractArray{C},Tp};
    F = nothing,
    g = ProximalOperators.Zero(),
    L = nothing,
    N = N,
    S = nothing,
    F_full = nothing,
    data = nothing,
    DNN_config::Maybe{Tdnn} = nothing
) where {R,C<:RealOrComplex{R},Tp,Tdnn}
    F === nothing && (F = fill(ProximalOperators.Zero(), (N,)))
    # dispatching the iterator
    DNN_config == nothing ? w0 = copy(x0) : w0 = DNN_config()
    if solver.DNN_training
        iter = FINITO_lbfgs_adaptive_DNN_iterable(
            F,
            F_full,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.η,
            solver.β,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
            LBFGS(w0, solver.memory),
            solver.adaptive,
            solver.tol_b,
            data,
            DNN_config
        )
    elseif solver.LFinito
        if solver.DeepLFinito[1]
            iter = FINITO_DLFinito_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                solver.DeepLFinito[2],
                solver.DeepLFinito[3],
            )
        else
            iter = FINITO_LFinito_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
            )
        end
    elseif solver.lbfgs
        if solver.DeepLFinito[1]
            iter = FINITO_DFlbfgs_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
            LBFGS(x0, solver.memory),
            solver.DeepLFinito[2],
            solver.DeepLFinito[3]
        )
        elseif solver.adaptive
            iter = FINITO_lbfgs_adaptive_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.η,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
                solver.adaptive,
                solver.tol_b,
                S,
            )
        else
            iter = FINITO_lbfgs_iterable(
                F,
                g,
                x0,
                N,
                L,
                solver.γ,
                solver.β,
                solver.sweeping,
                solver.minibatch[2],
                solver.α,
                LBFGS(x0, solver.memory),
            )
        end
    elseif solver.adaptive
        iter = FINITO_adaptive_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.tol,
            solver.tol_b,
            solver.sweeping,
            solver.α,
        )
    else
        iter = FINITO_basic_iterable(
            F,
            g,
            x0,
            N,
            L,
            solver.γ,
            solver.sweeping,
            solver.minibatch[2],
            solver.α,
        )
    end
    return iter
end



#### TODO
    # remove finito_adaptive