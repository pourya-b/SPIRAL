using Random
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using SparseArrays
include("../utilities/utilities.jl")
cd(dirname(@__FILE__))
Random.seed!(10)
# Base.show(io::IO, f::Float64) = @printf(io, "%.16f", f)
T = Float64

## dimension
n, m = 600, 300 # n:cols m:rows
# n, m = 400, 10000

p = 100
rho = 1 # some positive number controlling how large solution is
λ = 1  

y_star = rand(m)
y_star ./= norm(y_star) #y^\star
C = rand(m, n) .* 2 .- 1
# # scaling the rows to make the problem harder
# C = diagm(0 =>  [12 * ones(Int(m/10)) ; ones(Int(9*m/10))]) *C
diag_n = dropdims(1 ./ sqrt.(sum(C .^ 2, dims = 1)), dims = 1)
C = C * spdiagm(0 => diag_n); # normalize columns

CTy = abs.(C' * y_star)
perm = sortperm(CTy, rev = true) # indices with decreasing order by abs

alpha = zeros(n)
for i = 1:n
    if i <= p
        alpha[perm[i]] = λ / CTy[perm[i]]
    else
        temp = CTy[perm[i]]
        if temp < 0.1 * λ
            alpha[perm[i]] = λ
        else
            alpha[perm[i]] = λ * rand() / temp
        end
    end
end
A = C * diagm(0 => alpha)   # scaling the columns of Cin
# generate the primal solution
x_star = zeros(n)
for i = 1:n
    if i <= p
        x_star[perm[i]] = rand() * rho / sqrt(p) * sign(dot(A[:, perm[i]], y_star))
    end
end
b = A * x_star + y_star
f_star = norm(y_star) / 2 + λ * norm(x_star, 1) # the solution

F = Vector{LeastSquares}(undef, 0) # array of f_i functions
L = Vector{T}(undef, 0) # array for Lipschitz constants
γ = Vector{T}(undef, 0) # array for gamma constants

F_sum = LeastSquares(A, b, Float64(1))

N = m
for i = 1:N
    tempA = A[i:i, :]
    f = LeastSquares(tempA, b[i:i], Float64(N))
    Lf = opnorm(tempA)^2 * N
    push!(F, f)
    push!(L, Lf)
    push!(γ, 0.999 * N / Lf)
end

g = NormL1(λ)
func = Cost_FS(F, g, N, n, γ, nothing)

stuff = [
    # Dict( # basic version (high memory)
    #     "LFinito" => false,
    #     "DeepLFinito" => (false, 1, 1),
    #     "single_stepsize" => false,
    #     "minibatch" => [(true, i |> Int) for i in [1]],
    #     "sweeping" => [2],
    #     "label" => "DS", # diff-stepsizes
    #     "lbfgs" => false,
    #     "adaptive" => false,
    # ),
    Dict( # no-ls
        "LFinito" => true,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "minibatch" => [(true, i |> Int) for i in [1]],
        "sweeping" => [2],
        "label" => "LM", # LFinito
        "lbfgs" => false,
        "adaptive" => false,
    ),
    Dict( # SPIRAL
        "LFinito" => false,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "sweeping" => [2],
        "minibatch" => [(true, i |> Int) for i in [1]],
        "label" => "lbfgs", # LFinito
        "lbfgs" => true,
        "adaptive" => false,
    ),
    Dict( # adaSPIRAL
        "LFinito" => false,
        "DeepLFinito" => (false, 1, 1),
        "single_stepsize" => false,
        "sweeping" => [2],
        "minibatch" => [(true, i |> Int) for i in [1]],
        "label" => "lbfgs_ada", # LFinito
        "lbfgs" => true,
        "adaptive" => true,
        "DNN" => false
    ),
]

Labelsweep = ["rnd", "clc", "sfld"] # randomized, cyclical, shuffled

# run comparisons and save data to path plot_data/str
str = "test1/"
β = 1/10 # ls division parameter
λ = 0
plot_extras = saveplot(func, β, nothing, str, false)
println("test 1 for L ratio: ", Int(floor(maximum(L) / minimum(L))))

############################## initial point ################################
x0 = ones(n)
R = eltype(x0)
γ = 1/(2 * maximum(L))
# println(γ)
solver_ = CIAOAlgorithms.SGD{R}(γ=γ)
iter_ = CIAOAlgorithms.iterator(solver_, x0, F = F, g = g, N = N, DNN_config = false)
~, ~, ~, x0 = loopnsave(iter_, N, 10, 1, plot_extras)
##############################################################################

maxit = 250 |> Int # maximum number of epochs (not exact for lbfgs)
convex_flag = true

# println(x0)
println("comparisons")
Comparisons!(stuff, Labelsweep, maxit, x0, L, plot_extras)
include("../utilities/comparisons_SVRG.jl")
include("../utilities/comparisons_SGD.jl")
include("../utilities/comparisons_SARAH.jl")
include("../utilities/comparisons_SAG_SAGA.jl")



