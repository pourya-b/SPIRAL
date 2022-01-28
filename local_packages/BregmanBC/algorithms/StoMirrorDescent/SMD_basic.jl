struct SMD_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                       # smooth term   
    x0::Tx                  # initial point
    N::Int                    # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{R}             # stepsizes 
    α::R                    # in (0, 1), e.g.: 0.99
    diminishing::Bool       # diminishing stepsize
end

mutable struct SMD_basic_state{R<:Real,Tx}
    γ::R             # stepsize parameter
    z::Tx
    # some extra placeholders 
    s::Tx                  # temp variable
    ∇f_temp::Tx             # placeholder for gradients 
    idxr::Int                # running idx set
    epoch_cnt::R            # epoch counter 
end

function SMD_basic_state(γ::R, z::Tx, N) where {R,Tx}
    return SMD_basic_state{R,Tx}(γ, z, copy(z), copy(z), 1, 1/N)
end

function Base.iterate(iter::SMD_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            if isa(iter.L, R)
                L = iter.L
            else
                @warn "--> smoothness parameter most be a scalar..."
                L = maximum(iter.L)
            end

            if iter.diminishing
                # γ = R(10) / L
                γ = iter.α / L
                # println("diminishing-------------")
            else
                λ = iter.α / L
                γ = iter.α * (λ / (1 + λ * L))
            end
        end
    else
        isa(iter.γ, R) ? (γ = iter.γ) : (@warn "only single stepsize is supported in SMD") # provided γ
    end

    #initializing the vectors 
    idxr = rand(1:iter.N)
    ∇f, ~ = gradient(iter.H, iter.x0)
    s = ∇f ./ γ
    ∇f, ~ = gradient(iter.F[idxr], iter.x0)
    s .-= ∇f

    z, ~ = prox_Breg(iter.H, iter.g, s, γ)

    # println(z)
    state = SMD_basic_state(γ, z, N)

    return state, state
end

function Base.iterate(iter::SMD_basic_iterable{R}, state::SMD_basic_state{R}) where {R}

    # select an idxrex
    state.idxr = rand(1:iter.N)
    if iter.diminishing
        # state.γ = R(1) / (iter.L * sqrt(1 + (R(1) / (iter.L * state.γ))^2))
        # state.γ = R(1) / (iter.L +  (R(1) /  state.γ ) )
        state.γ = iter.α / (iter.N * (ceil(state.epoch_cnt) * iter.L))
    end
    # compute s
    gradient!(state.∇f_temp, iter.H, state.z) # update the gradient
    state.s .= state.∇f_temp ./ state.γ
    gradient!(state.∇f_temp, iter.F[state.idxr], state.z) # update the gradient
    state.s .-= state.∇f_temp

    prox_Breg!(state.z, iter.H, iter.g, state.s, state.γ)
    state.epoch_cnt += 1/iter.N # bug prone! it is for only batchsize = 1

    return state, state
end

solution(state::SMD_basic_state) = state.z
solution_γ(state::SMD_basic_state) = state.γ
solution_epoch(state::SMD_basic_state) = state.epoch_cnt


#TODO list
## only one H is supported in SMD...
# fix type of H
