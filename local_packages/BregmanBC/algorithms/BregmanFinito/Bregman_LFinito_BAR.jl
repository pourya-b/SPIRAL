struct LBreg_Finito_BAR_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                       # smooth term   
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8          # to only use one stepsize γ
    batch::Int                # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct LBreg_Finito_BAR_state{R<:Real,Tx}
    s::Array{Tx}            # for saving! 
    γ::Array{R}             # stepsize parameter
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int    # number of batches 
    epoch_cnt::R               
    # some extra placeholders 
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    z_full::Tx
    inds::Array{Int}        # needed for shuffled only! 
end

function LBreg_Finito_BAR_state(s::Array{Tx}, γ::Array{R}, av::Tx, ind, d) where {R,Tx}
    return LBreg_Finito_BAR_state{R,Tx}(
        s,
        γ,
        av,
        ind,
        d,
        0,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
    )
end

function Base.iterate(iter::LBreg_Finito_BAR_iterable{R, C, Tx}) where {R,C, Tx}
    N = iter.N
    r = iter.batch # batch size 
    # create index sets 
    ind = Vector{Vector{Int}}(undef, 0)
    d = Int(floor(N / r))
    for i = 1:d
        push!(ind, collect(r*(i-1)+1:i*r))
    end
    r * d < N && push!(ind, collect(r*d+1:N))
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            for i = 1:N
                isa(iter.L, R) ? (γ = fill(iter.α * R(iter.N) / iter.L, (N,))) :
                (γ[i] = iter.α * R(N) / (iter.L[i]))
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) # provided γ
    end
    #initializing the vectors 
                        s = Vector{Tx}(undef, 0)
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        av .-= ∇f ./ N
        ∇h, ~ = gradient(iter.H[i], iter.x0)
        av .+= ∇h ./ γ[i]
                        push!(s, ∇h / γ[i] - ∇f / N) # table of s_i
    end

    state = LBreg_Finito_BAR_state(s, γ, av, ind, cld(N, r))

    return state, state
end

function Base.iterate(
    iter::LBreg_Finito_BAR_iterable{R},
    state::LBreg_Finito_BAR_state{R},
) where {R}
    prox_Breg!(state.z_full, iter.H, iter.g, state.av, state.γ)

    # full update 
    state.av .= zero(state.z_full)
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.H[i], state.z_full)
        state.av .+= state.∇f_temp ./ state.γ[i]
        gradient!(state.∇f_temp, iter.F[i], state.z_full) # update the gradient
        state.av .-= state.∇f_temp ./ iter.N
    end

    # inner loop
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled
    for j in state.inds
        prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)
        for i in state.ind[j]
            gradient!(state.∇f_temp, iter.F[i], state.z_full) # update the gradient
            state.av .+= state.∇f_temp ./ iter.N
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.av .-= state.∇f_temp ./ iter.N
                          state.s[i] .= - state.∇f_temp ./ iter.N # just for plots 

            gradient!(state.∇f_temp, iter.H[i], state.z_full) # update the gradient
            state.av .-= state.∇f_temp ./ state.γ[i]
            gradient!(state.∇f_temp, iter.H[i], state.z) # update the gradient
            state.av .+= state.∇f_temp ./ state.γ[i]
                         state.s[i] .+= state.∇f_temp ./ state.γ[i] # just for plots 
        end
    end

    state.epoch_cnt += 2
    return state, state
end

solution(state::LBreg_Finito_BAR_state) = state.z_full
solution_epoch(state::LBreg_Finito_BAR_state) = state.epoch_cnt
solution_s(state::LBreg_Finito_BAR_state) = state.s
solution_γ(state::LBreg_Finito_BAR_state) = state.γ
    

## TODO
# prox_Breg currently accepts one H only! 
