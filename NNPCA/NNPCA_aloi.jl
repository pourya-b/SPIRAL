using Test
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
include("../utilities/utilities.jl")
cd(dirname(@__FILE__))
# Base.show(io::IO, f::Float64) = @printf(io, "%.16f", f)
T = Float64

# ----------------creating the true data--------------------
data = CSV.read("../../../../datasets/aloi.scale.csv", DataFrame)

data = Matrix(data)
println(typeof(data))
data = 1.0* hcat(data[1:end,1], data[1:end,4:end]) # removing labels (MNIST)

############################# data normalization ############################
# normalize with one of these
# for i in 1:size(data,1) # normalization to make Lratio = 1
# 	data[i,:] /= norm(data[i,:])
# end

# for i in 1:size(data,1) # intact Lratio
# 	data[i,:] /= 255
# end
#############################################################################

xs = Matrix(data*1.0)
println(size(xs))

N, n = size(xs, 1), size(xs, 2)

F = Vector{LeastSquares}(undef, 0)
L = Vector{T}(undef, 0)
γ = Vector{T}(undef, 0)
q = zeros(1)
R = real
F_sum = LeastSquares(xs, zeros(N), Float64(-1))

for i = 1:N
    # QQ = - xs[i:i,:]' * xs[i:i,:]
    f = LeastSquares(xs[i:i,:], q, -1.0)
    push!(F, f)
    # compute L
    Lf = xs[i, :]' * xs[i, :]
    push!(L, Lf)
    push!(γ, 0.95 * N / Lf)
end

g = IndNonnegativeBallL2()
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
    ),    
]

Labelsweep = ["rnd", "clc", "sfld"] # randomized, cyclical, shuffled

# run comparisons and save data to path plot_data/str
str = "test1/"
β = 1/50 # ls division parameter
λ = 0
plot_extras = saveplot(func, β, nothing, str, false)
println("test 1 for L ratio: ", Int(floor(maximum(L) / minimum(L))))

############################## initial point ################################
x0 = ones(n)
R = eltype(x0)
γ = 1/(2 * maximum(L))
# println(γ)
solver = CIAOAlgorithms.SGD{R}(γ=γ)
iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, N = N, DNN_config = false)
~, ~, ~, x0 = loopnsave(iter, N, 10, 1, plot_extras)
##############################################################################

maxit = 50 |> Int # maximum number of epochs (not exact for lbfgs)
convex_flag = false

# println(x0)
println("comparisons")
Comparisons!(stuff, Labelsweep, maxit, x0, L, plot_extras)
include("../utilities/comparisons_Panoc.jl")
include("../utilities/comparisons_SVRG.jl")
include("../utilities/comparisons_SGD.jl")
include("../utilities/comparisons_SARAH.jl")
include("../utilities/comparisons_SAG_SAGA.jl")



