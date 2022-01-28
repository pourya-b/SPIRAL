struct LBreg_Finito_lbfgs_iterable{R<:Real,C<:RealOrComplex{R},Tx<:AbstractArray{C},Tf,Tg,TH}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term
    H::Any                       # smooth term   
    x0::Tx                  # initial point
    N::Int                       # of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i    
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8          # to only use one stepsize γ
    batch::Int                   # batch size
    α::R                    # in (0, 1), e.g.: 0.99
    B::TH
end

mutable struct LBreg_Finito_lbfgs_state{R<:Real,Tx,TH}
    γ::Array{R}             # stepsize parameter
    av::Tx                  # the running average
    ind::Array{Array{Int}}  # running index set
    d::Int                  # number of batches
    B::TH                   # Hessian approx (LBFGS struct)
    # some extra placeholders 
    epoch_cnt::R       # number of lineseaches 
    z::Tx
    ∇f_temp::Tx             # placeholder for gradients 
    temp::R                # placeholder for bregman distances 
    zbar::Tx
    zbar_prev::Maybe{Tx}    # bar z previous
    res_zbar::Tx            # v
    res_zbar_prev::Maybe{Tx}# v
    dir::Tx                 # direction
    ∇f_sum::Tx              # for envelope value
    z_trial::Tx             # linesearch candidate
    inds::Array{Int}        # needed for shuffled only! 
    τ::Float64              # number of epochs
end

function LBreg_Finito_lbfgs_state(γ::Array{R}, av::Tx, ind, d, B::TH) where {R,Tx,TH}
    return LBreg_Finito_lbfgs_state{R,Tx,TH}(
        γ,
        av,
        ind,
        d,
        B,
        0.0,
        copy(av),
        copy(av),
        0.0,
        copy(av),
        nothing,
        copy(av),
        nothing,
        copy(av),
        copy(av),
        copy(av),
        collect(1:d),
        1.0
    )
end

function Base.iterate(iter::LBreg_Finito_lbfgs_iterable{R}) where {R}
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
    # γ /= N
    # println("step size devided by N")
    # println(γ)
    av = zero(iter.x0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        av .-= ∇f ./ N
        ∇f, ~ = gradient(iter.H[i], iter.x0)
        av .+= ∇f ./ γ[i]
    end
    state = LBreg_Finito_lbfgs_state(γ, av, ind, cld(N, r), iter.B)
    println("lbfgs version of Bregman_finito")

    return state, state
end

function Base.iterate(
    iter::LBreg_Finito_lbfgs_iterable{R},
    state::LBreg_Finito_lbfgs_state{R},
) where {R}

    if state.zbar_prev === nothing # for bfgs updates
        state.zbar_prev = zero(state.z)
        state.res_zbar_prev = zero(state.z)
    end

    prox_Breg!(state.zbar, iter.H, iter.g, state.av, state.γ)

    envVal = 0.0 # bug prone! in-place operations overlooked for envVal
    state.∇f_sum .= zero(state.av) 
    state.av .= zero(state.zbar)
    # full update
    for i = 1:iter.N
        gradient!(state.∇f_temp, iter.H[i], state.zbar)
        state.av .+= state.∇f_temp ./ state.γ[i]
        state.∇f_temp, fi_z = gradient(iter.F[i], state.zbar) # update the gradient
        state.av .-= state.∇f_temp ./ iter.N
        envVal += fi_z / iter.N
        state.∇f_sum .+= state.∇f_temp 
    end

    prox_Breg!(state.res_zbar, iter.H, iter.g, state.av, state.γ) # res_zbar = vbar - bug prone: γ_hat is omitted in all the formulations here
    gz = iter.g(state.res_zbar)
    envVal += gz 
    # envVal .+= gz
    state.res_zbar .-= state.zbar # v- z 

    if state.zbar_prev !== nothing
        update!(state.B, state.zbar - state.zbar_prev, -state.res_zbar +  state.res_zbar_prev) # to update H by lbfgs! (H, s-s_pre, y-y_pre)
        # store vectors for next update
        copyto!(state.zbar_prev, state.zbar)
        copyto!(state.res_zbar_prev, state.res_zbar)
    end
    mul!(state.dir, state.B, state.res_zbar) # update d
    envVal += real(dot(state.∇f_sum, state.res_zbar)) / iter.N
    # bug prone! for dist_Breg, in-place function returns error, I opted to return the result value
    state.temp = dist_Breg(iter.H[1], state.res_zbar + state.zbar, state.zbar) # bug prone! here I assume we have only one h similar for every f_i
    # println("dist is:", state.temp)
    state.temp *= sum(1 ./ state.γ)
    envVal += state.temp

    state.τ = 1.0
    # linesearch
    for i=1:5 
        state.z_trial .=  state.zbar .+ (1- state.τ) .* state.res_zbar + state.τ * state.dir

        state.av .= zero(state.z_trial)
        state.∇f_sum .= zero(state.av)
        envVal_trial = 0 # bug prone! in-place operations overlooked for envVal_trial
        for i = 1:iter.N
            gradient!(state.∇f_temp, iter.H[i], state.z_trial)
            state.av .+= state.∇f_temp ./ state.γ[i]
            state.∇f_temp, fi_z = gradient(iter.F[i], state.z_trial) # update the gradient
            state.av .-= state.∇f_temp ./ iter.N
            envVal_trial += fi_z / iter.N
            state.∇f_sum .+= state.∇f_temp 
        end
        prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)
        gz = iter.g(state.z)
        envVal_trial += gz
        state.z .-= state.z_trial # v-z 

        envVal_trial += real(dot(state.∇f_sum, state.z)) / iter.N
        state.temp = dist_Breg(iter.H[1], state.z + state.z_trial, state.z_trial) # bug prone! here I assume we have only one h similar for every f_i
        state.temp *= sum(1 ./ state.γ)
        envVal_trial += state.temp

        envVal_trial <= envVal + eps(R) && break # envVal: envelope value
        state.τ /= 50  
        state.epoch_cnt += 1     
        # println("line reduction occurred!")
    end
    # println(state.τ)
    state.zbar .= state.z_trial # of line search
    # state.z .= state.zbar

    # inner loop
    iter.sweeping == 3 && (state.inds = randperm(state.d)) # shuffled
    for j in state.inds
        prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)
        for i in state.ind[j]
            gradient!(state.∇f_temp, iter.F[i], state.zbar) # update the gradient
            state.av .+= (state.∇f_temp ./ iter.N)
            gradient!(state.∇f_temp, iter.F[i], state.z) # update the gradient
            state.av .-= (state.∇f_temp ./ iter.N)

            gradient!(state.∇f_temp, iter.H[i], state.zbar) # update the gradient
            state.av .-= (state.∇f_temp ./ state.γ[i])
            gradient!(state.∇f_temp, iter.H[i], state.z) # update the gradient
            state.av .+= (state.∇f_temp ./ state.γ[i])
        end
    end

    state.epoch_cnt += 3 # bug prone! batchsize is assumed to be equal to 1 here

    return state, state
end
solution(state::LBreg_Finito_lbfgs_state) = state.z
solution_epoch(state::LBreg_Finito_lbfgs_state) = state.epoch_cnt
