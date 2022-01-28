module BregmanBC

using LinearAlgebra
using ProximalOperators

const RealOrComplex{R} = Union{R,Complex{R}}
const Maybe{T} = Union{T, Nothing}
const HermOrSym{T, S} = Union{Hermitian{T, S}, Symmetric{T, S}}
const RealBasedArray{R} = AbstractArray{C, N} where {C <: RealOrComplex{R}, N}
const TupleOfArrays{R} = Tuple{RealBasedArray{R}, Vararg{RealBasedArray{R}}}
const ArrayOrTuple{R} = Union{RealBasedArray{R}, TupleOfArrays{R}}
const TransposeOrAdjoint{M} = Union{Transpose{C,M} where C, Adjoint{C,M} where C}



# algorithms 
include("algorithms/BregmanFinito/Bregman_Finito.jl")
include("algorithms/StoMirrorDescent/SMD.jl")

# utilities
include("utilities/proxgrads.jl")

end # module





##### todo
## poisson tests with randn not rand!