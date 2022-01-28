using LinearAlgebra
using BregmanBC
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using SparseArrays
using Images, TestImages
using Hadamard, Statistics
using Random

import BregmanBC: solution
import CIAOAlgorithms: solution

include("../utilities/utilities_phase.jl")
include("../utilities/utilities_breg.jl")
include("comparisons_SMD.jl")

cd(dirname(@__FILE__))
# Base.show(io::IO, f::Float64) = @printf(io, "%.16f", f)
rndseed = 50
Random.seed!(rndseed)
T = Float64
R = Float64
image_num = 100
digits = "digits_6"
res_xz = false 
p_fail = 0.02

#####----------------------image examples--------------------------------------
img = readdlm(string("../../datasets/", digits, ".csv"), ',', Float64, '\n')

# to view
p = image_num
x_star = -img[image_num, :] #first image vectorized (1 times 256), - to make background white
# Gray.(reshape(img[image_num,:], 16, 16)')

n = length(x_star)
k = 5 # from Duchi's Sec 6.3
N = k * n 
# generate A and b
A, b = Ab_image(x_star,k, p_fail) # random

# initial point according to Wang!
x0 =  1 * initializeX(A, b, N)
# x0 = ones(n)/n

#### ------------------------------------ init point visualization ---------------------------------
Gray.(reshape(x0, 16,16))
x_clamp = map(clamp01nan, x0)
save(string("solutions/datasets/",digits,"/initialization.png"), colorview(Gray, reshape(abs.(x_clamp), 16, 16)'))
Gray.(reshape(x_star, 16,16))
x_clamp = map(clamp01nan, x_star)
save(string("solutions/datasets/",digits,"/original.png"), colorview(Gray, reshape(abs.(x_clamp), 16, 16)'))
#### ---------------------------------------------------------------------------------------------


F = Vector{Quartic}(undef, 0)
H = Vector{Poly42}(undef, 0)
L = Vector{R}(undef, 0)
γ = Vector{R}(undef, 0)
for i = 1:N
    tempA = A[i, :]
    f = Quartic(tempA, b[i])
    normA = norm(tempA)
    Lf = 3 * normA^4 + normA^2 * abs(b[i])
    h = Poly42()
    push!(F, f)
    push!(H, h)
    push!(L, Lf)
    push!(γ, 0.999 * R(N) / Lf)
end

# str_res = res_xz ? "res/" : ""

# λ = 160 # for IndBallL0(λ)
λ = 0.5/N # for NormL1(λ)

g = NormL1(λ)
func = Cost_FS(F, g, N, n, γ, nothing)

stuff = [
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

stuff_breg = [
        Dict( #Finito/MISO
            "LBFinito" => false,
            "single_stepsize" => false,
            "minibatch" => [(true, i |> Int) for i in [1]],
            "sweeping" => [2],
            "label" => "DS", # diff-stepsizes
            "lbfgs" => false,
        ),
        Dict( # no-ls
              "LBFinito"          => true,
              "single_stepsize"   => false,
              "minibatch"         => [(true, i |> Int) for i in [1]],
              "sweeping"          => [2],
              "label"             =>  "LM", # LBFinito
              "lbfgs"             => false,
            ),
        Dict( # SPIRAL
            "LBFinito"          => false,
            "lbfgs"             => true,
            "single_stepsize"   => false,
            "minibatch"         => [(true, i |> Int) for i in [1]],
            "sweeping"          => [2],
            "label"             =>  "lbfgs", # LBFinito
          ),
    ]

Labelsweep = ["rnd", "clc", "sfld"] # randomized, cyclical, shuffled

# run comparisons and save data to path plot_data/str
β = 1/10 # ls division parameter
# plot_extras = saveplot(func, β, nothing, str)
println("test 1 for L ratio: ", Int(floor(maximum(L) / minimum(L))))

############################## initial point ################################
# x0 = ones(n)
# R = eltype(x0)
# γ = N/(2 * maximum(L))
# # println(γ)
# solver_ = CIAOAlgorithms.SGD{R}(γ=γ)
# iter_ = CIAOAlgorithms.iterator(solver_, x0, F = F, g = g, N = N)
# ~, ~, ~, x0 = loopnsave(iter_, N, 100, 1, plot_extras)
##############################################################################

maxit = 55 |> Int # maximum number of epochs (not exact for lbfgs)
convex_flag = true
str_res = ""
str_test = string("datasets/", digits, "/")
str_test_d = string("close/", str_res)
str = string(str_test, str_test_d)
tol = 1e-15
plot_extras = saveplot(func, β, nothing, str)

# println(x0)
println("comparisons")
Comparisons!(p, H, func, L, stuff, Labelsweep, maxit, tol, x0, str, γ, res_xz, p_fail, λ, x_star, plot_extras)
Comparisons_breg!(p, H, func, L, stuff_breg, Labelsweep, maxit, tol, x0, str, γ, res_xz, p_fail, λ, x_star)
Comparisons_SMD!(p, H, func, L, stuff, Labelsweep, maxit, tol, x0, str, γ, res_xz, p_fail, λ, x_star)




