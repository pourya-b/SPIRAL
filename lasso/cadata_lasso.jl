#- implementing (45) in "ProxSARAH: An Efficient Algorithmic Framework for
#- Stochastic Composite Nonconvex Optimization"

using Test
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
# Base.show(io::IO, f::Float64) = @printf(io, "%.16f", f)
T = Float64
factor = 100
data = CSV.read("../../../../datasets/cadata.csv", DataFrame)
data = Matrix(data)/factor
println(typeof(data))
b = 1.0 * data[1:end,2] # removing labels (covetype)
A = 1.0 * hcat(data[1:end,1], data[1:end,4:end])
N, n = size(A, 1), size(A, 2)
λ = 0.5/N
for i in 1:size(A,2) # intact Lratio
	A[:,i] /= norm(A[:,i])
end

println(N)
println(n)

F = Vector{LeastSquares}(undef, 0) # array of f_i functions
L = Vector{T}(undef, 0) # array for Lipschitz constants
γ = Vector{T}(undef, 0) # array for gamma constants

F_sum = LeastSquares(A, b, Float64(1))
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
plot_extras = saveplot(func, β, nothing, str)
println("test 1 for L ratio: ", Int(floor(maximum(L) / minimum(L))))

############################## initial point ################################
x0 = ones(n)
R = eltype(x0)
γ = 1/(2 * maximum(L))
# println(γ)
solver_ = CIAOAlgorithms.SGD{R}(γ=γ)
iter_ = CIAOAlgorithms.iterator(solver_, x0, F = F, g = g, N = N)
~, ~, ~, x0 = loopnsave(iter_, N, 10, 1, plot_extras)
##############################################################################

maxit = 500 |> Int # maximum number of epochs (not exact for lbfgs)
convex_flag = true

# println(x0)
println("comparisons")
Comparisons!(stuff, Labelsweep, maxit, x0, L, plot_extras)
include("../utilities/comparisons_SVRG.jl")
include("../utilities/comparisons_SGD.jl")
include("../utilities/comparisons_SARAH.jl")
include("../utilities/comparisons_SAG_SAGA.jl")



