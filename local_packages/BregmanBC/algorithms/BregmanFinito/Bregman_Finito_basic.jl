struct Breg_FINITO_basic_iterable{R<:Real,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Tg}
    F::Array{Tf}            # smooth term  
    g::Tg                   # nonsmooth term 
    H::Any                       # smooth term  
    x0::Tx                  # initial point
    N::Int                  # number of data points in the finite sum problem 
    L::Maybe{Union{Array{R},R}}  # Lipschitz moduli of nabla f_i	
    γ::Maybe{Union{Array{R},R}}  # stepsizes 
    sweeping::Int8          # to only use one stepsize γ
    batch::Int              # batch size
    α::R                    # in (0, 1), e.g.: 0.99
end

mutable struct Breg_FINITO_basic_state{R<:Real,Tx}
    s::Array{Tx}            # table of x_j- γ_j/N nabla f_j(x_j) 
    γ::Array{R}             # stepsize parameters 
    av::Tx                  # the running average
    z::Tx
    ind::Array{Array{Int}} # running index set 
    d::Int                  # number of batches
    epoch_cnt::R            # epoch counter
    
    # some extra placeholders  
    ∇f_temp::Tx             # placeholder for gradients 
    ∇h_temp::Tx             # placeholder for gradients 
    idxr::Int               # running idx in the iterate 
    idx::Int                # location of idxr in 1:N 
    inds::Array{Int}        # needed for shuffled only  
end

function Breg_FINITO_basic_state(s, γ::Array{R}, av::Tx, z::Tx, ind, d) where {R,Tx}
    return Breg_FINITO_basic_state{R,Tx}(
        s,
        γ,
        av,
        z,
        ind,
        d,
        0.0,
        zero(av),
        zero(av),
        Int(1),
        Int(0),
        collect(1:d),
    )
end

function Base.iterate(iter::Breg_FINITO_basic_iterable{R,C,Tx}) where {R,C,Tx}
    N = iter.N
    # define the batches
    r = iter.batch # batch size 
    if iter.sweeping == 1
        ind = [collect(1:r)] # placeholder
    else
        ind = Vector{Vector{Int}}(undef, 0)
        d = Int(floor(N / r))
        for i = 1:d
            push!(ind, collect(r*(i-1)+1:i*r))
        end
        r * d < N && push!(ind, collect(r*d+1:N))
    end
    d = cld(N, r) # number of batches  

    # updating the stepsize 
    if iter.γ === nothing
        if iter.L === nothing
            @warn "--> smoothness parameter absent"
            return nothing
        else
            γ = zeros(R, N)
            if isa(iter.L, R)
                γ = fill(iter.α * R(iter.N) / iter.L, (N,))
                println(iter.L)
            else
                for i = 1:N
                    γ[i] = iter.α * R(N) / iter.L[i]
                end
            end
        end
    else
        isa(iter.γ, R) ? (γ = fill(iter.γ, (N,))) : (γ = iter.γ) #provided γ
    end
    # computing the gradients and updating the table s 
    s = Vector{Tx}(undef, 0)
    for i = 1:N
        ∇f, ~ = gradient(iter.F[i], iter.x0)
        ∇h, ~ = gradient(iter.H[i], iter.x0)
        push!(s, ∇h / γ[i] - ∇f / N) # table of s_i
    end
    #initializing the vectors 
    av = sum(s) # the running average  

    z, ~ = prox_Breg(iter.H, iter.g, av, γ)    # for now all h are the same!
    state = Breg_FINITO_basic_state(s, γ, av, z, ind, d)

    println("basic verison of bregman finito")

    return state, state
end

function Base.iterate(
    iter::Breg_FINITO_basic_iterable{R},
    state::Breg_FINITO_basic_state{R},
) where {R}
    # manipulating indices 
    if iter.sweeping == 1 # uniformly random    
        # state.ind = [rand(1:iter.N, iter.batch)]
        state.ind = [sample(1:iter.N, iter.batch, replace = false)]
    elseif iter.sweeping == 2  # cyclic
        state.idxr = mod(state.idxr, state.d) + 1
    elseif iter.sweeping == 3  # shuffled cyclic
        if state.idx == state.d
            state.inds = randperm(state.d)
            state.idx = 1
        else
            state.idx += 1
        end
        state.idxr = state.inds[state.idx]
    end
    # println("index set is $(state.ind) and batch size is $(iter.batch) and selected set is $(state.ind[state.idxr])")
    # the iterate
    for i in state.ind[state.idxr]
        # perform the main steps 
        gradient!(state.∇f_temp, iter.F[i], state.z)
        state.∇f_temp ./= iter.N
        gradient!(state.∇h_temp, iter.H[i], state.z)
        state.∇h_temp ./= state.γ[i]
        state.∇h_temp .-= state.∇f_temp
        @. state.av += state.∇h_temp - state.s[i]
        state.s[i] .= state.∇h_temp  #update s_i
    end
    # if ~isempty(findall(x->x>=0, state.av))
    #     println(state.av)
    # end

    prox_Breg!(state.z, iter.H, iter.g, state.av, state.γ)
    state.epoch_cnt += 1/iter.N # bug prone! it is for only batchsize = 1

    return state, state
end

solution(state::Breg_FINITO_basic_state) = state.z
solution_s(state::Breg_FINITO_basic_state) = state.s
solution_γ(state::Breg_FINITO_basic_state) = state.γ
solution_epoch(state::Breg_FINITO_basic_state) = state.epoch_cnt

#TODO list
## H[1] is used for now since they are the same
