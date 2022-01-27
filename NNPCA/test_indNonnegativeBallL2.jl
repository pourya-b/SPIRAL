using Test
using LinearAlgebra
using CIAOAlgorithms
using ProximalOperators
using ProximalAlgorithms: IterationTools
using Base.Iterators: take
using CSV, DataFrames # for loading the data
using DelimitedFiles
using ProximalAlgorithms
using Printf

Base.show(io::IO, f::Float64) = @printf(io, "%.4f", f)
include("../utilities/utilities.jl")
cd(dirname(@__FILE__))
T = Float64

g = IndNonnegativeBallL2()

N = 10
γ = 0.5
x = randn(N)
x /= 0.5*norm(x)
w = zeros(N)

CIAOAlgorithms.prox!(w, g, x, γ)

println("initial point: ", x)
println("prox point: ", w)
println("initial norm: ", x' * x)
println("prox norm: ", w' * w)
println("---------------------")

w, _ = CIAOAlgorithms.prox(g, x, γ)

println("initial point: ", x)
println("prox point: ", w)
println("initial norm: ", x' * x)
println("prox norm: ", w' * w)