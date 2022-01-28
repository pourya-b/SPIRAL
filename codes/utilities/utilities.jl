function dimension(config)
    L = length(config)
    dim = 0
    for i=1:1:L
        dim += (config[i][1] + 1) * config[i][2]
    end
    return dim
end

function my_model(x,w,config)
    L = length(config)
    out = x
    indx = 1
    for i=1:2:L
        dim = config[i]
        length = dim[1] * dim[2]
        
        out = reshape(w[indx:indx+length-1],config[i])' * out'
        indx += length
        length = dim[2]
        out = out .+ w[indx:indx+length-1]
        indx += length
        out = config[i+1].(out) 
        out = out'
    end
    return out
end

function myF(NewLoss,x_train,y_train,ind,w)
    return NewLoss(x_train[ind,:]', y_train[ind,:]',w)
end

id(x) = x

function DNN_config!(;gs=nothing)
    inx = 1
    if (typeof(gs) == Zygote.Grads)
        out = zeros(config_length)
        for par in ps
            xp = gs[par]
            xpr = reshape(xp,(length(xp),1))
            Len = length(xpr)
            out[inx:inx+Len-1] = xpr
            inx += Len
        end
        return out
    elseif gs==nothing
        out = zeros(config_length)
        for par in ps
            xpr = reshape(par,(length(par),1))
            Len = length(xpr)
            out[inx:inx+Len-1] = xpr
            inx += Len
        end
        return out
    else
        for i=1:2*size(config)[1]
            Len = length(ps[i])
            ps[i] .= reshape(gs[inx:inx+Len-1],size(ps[i]))
            inx += Len
        end
        return nothing
    end
end

struct hessian{R,Tx,T}
    x_s::Tx
    y_s::Array{R}
    L::Array{R}
    N::T
    α::R
end

function hessian(x_s::Tx, y_s::Array{R}, L::Array{R}, N::T, α::R) where {R,Tx,T}
    return hessian{R,Tx,T}(x_s, y_s, L, N, α)
end


struct Cost_FS{Tf,Tg,Ts}
    F::Array{Tf}          # smooth term
    g::Tg                   # nonsmooth term
    N::Int                  # number of data points in the finite sum problem
    n::Int                  # F[i] : R^n \to R
    γ::Union{Array{Float64}, Nothing}
    S::Ts
end

struct Cost_FSS{Tf,Tg,Ts,Tff}
    F::Union{Tf,Array{Tf}}         # smooth term
    g::Tg                   # nonsmooth term
    N::Int                  # number of data points in the finite sum problem
    n::Int                  # F[i] : R^n \to R
    γ::Union{Array{Float64}, Nothing}
    S::Ts
    F_full::Tff
end

function Cost_FS(F::Array{Tf}, g::Tg, N, n, γ, S::Ts) where {Tf, Tg, Ts}
    return Cost_FS{Tf,Tg,Ts}(
        F,
        g,
        N,
        n,
        nothing,
        S
    )
end

function Cost_FS(F::Union{Tf,Array{Tf}}, g::Tg, N, n, γ, S::Ts,F_full::Tff) where {Tf, Tff, Tg, Ts}
    return Cost_FSS{Tf,Tg,Ts,Tff}(
        F,
        g,
        N,
        n,
        nothing,
        S,
        F_full
    )
end

function (S::Cost_FS)(x)
    cost = S.g(x)
    # println("cost now: ", cost)
    # println("x now: ", x[1:2])
    for i = 1:N
        cost += S.F[i](x) / S.N
    end
    return cost
end

function (SS::Cost_FSS)(x)
    println("func test")
    cost = SS.g(x)
    cost += SS.F_full(x)
    return cost
end

struct saveplot{R}
    f                     # cost_FS
    β::R                # ls division parameter
    x_star #::Union{Tx, Nothing}
    str::String
    eval_gradient::Bool
end


function Comparisons!(
  stuff,
    Labelsweep,
    maxit::Int,
    x0::Array{R},
    L,
    plot_extras,
) where {R<:Real}

    n = func.n
    N = func.N

    cost_history = Vector{Vector{R}}(undef, 0)
    res_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cnt = 0 # counter
    t = length(stuff)
    sol = copy(x0)
    for i = 1:t # loop for all the matrials in stuff
        t2 = length(stuff[i]["sweeping"])
        for j = 1:t2 # loop for index update strategy

            LFinito = stuff[i]["LFinito"]
            DeepLFinito = stuff[i]["DeepLFinito"]

            lbfgs = stuff[i]["lbfgs"]
            adaptive = stuff[i]["adaptive"]
            single_stepsize = stuff[i]["single_stepsize"]
            label = stuff[i]["label"]
            sweeping = stuff[i]["sweeping"][j]
            # DNN_flag = stuff[i]["DNN"]
            t3 = length(stuff[i]["minibatch"])

            for l = 1:t3 # loop for minibatch numbers
                cnt += 1
                minibatch = stuff[i]["minibatch"][l]
                size_batch = minibatch[2]
                println("solving using $(label) and with $(Labelsweep[sweeping]) sweeping....."); flush(stdout)

                solver = CIAOAlgorithms.Finito{T}(
                    sweeping = sweeping,
                    LFinito = LFinito,
                    DeepLFinito = DeepLFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = plot_extras.β,
                    # DNN_training = DNN_flag,
                    # γ = plot_extras.f.γ,
                )

                # if single_stepsize
                
                L_s = nothing
                if L != nothing
                    if single_stepsize
                        L_s = maximum(L)   # has to be divided since 1/L_F = \bar γ = (\sum 1/γ_i)^-1 = γ/N
                    else
                        L_s = L
                    end
                end
                factor = N #  # number of grad evals per iteration
                # if low memory then every itr is 2N updates
                if LFinito
                    if DeepLFinito[1]
                        factor = N * (1 + DeepLFinito[3])
                    else
                        factor = 2 * N
                    end
                elseif lbfgs
                    if DeepLFinito[1]
                        factor = (2+  DeepLFinito[3]) * N
                    else
                        factor = 3 * N
                    end
                end

                Maxit = Int(ceil(maxit * N/ factor))
                freq = 1 #Int(ceil(Maxit / 100)) # keeping at most 100 data points in _hist arrays


                iter = CIAOAlgorithms.iterator(
                    solver,
                    x0,
                    F = func.F,
                    g = func.g,
                    L = L_s,
                    N = N,
                    S = func.S
                    # F_full = func.F_full
                )

                it_hist, cost_hist, res_hist, sol =
                    loopnsave(iter, factor, Maxit, freq, plot_extras) # here is the main iterations

                push!(cost_history, cost_hist)
                push!(res_history, res_hist)
                push!(it_history, it_hist)

                # saving
                output = [it_history[end] cost_history[end]]
                println("output size: $((cost_hist))")
                d = length(cost_hist)
                rr = 1 #Int(ceil(d / 50)) # keeping at most 50 data points
                red_output = output[1:rr:end, :] #reduced
                println("output size: $(size(red_output))")


                # if λ === 0
                #     par_λ = 0
                # else
                #     if plot_extras.x_star ===nothing
                #         par_λ = log10(round(λ * N, digits = 5)) |> Int
                #     else
                #         par_λ = log10(round(λ, digits = 5)) |> Int
                #     end
                # end
                par_λ = 0
                open(
                    string(
                        "plot_data/",
                        str,
                        "cost/",
                        "DNN_N_",
                        N,
                        "_n_",
                        n,
                        "_batch_",
                        size_batch, # for l
                        "_",
                        stuff[i]["label"], # for i
                        "_",
                        Labelsweep[sweeping], # for j
                        "_Lratio_",
                        Int(floor(maximum(L) / minimum(L))),
                        "_lambda_",
                        "1e",
                        par_λ,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, red_output)
                end

                # residual |z- prox(z)|
                output = [it_history[end] res_history[end]]
                d = length(res_hist)
                rr = Int(ceil(d / 50)) # keeping at most 50 data points
                red_output = output[1:rr:end, :]

                println("test for L ratio: ", Int(floor(maximum(L) / minimum(L))))
                
                open(
                    string(
                        "plot_data/",
                        str,
                        "res/",
                        "DNN_N_",
                        N,
                        "_n_",
                        n,
                        "_batch_",
                        size_batch,
                        "_",
                        stuff[i]["label"],
                        "_",
                        Labelsweep[sweeping],
                        "_Lratio_",
                        Int(floor(maximum(L) / minimum(L))),
                        "_lambda_",
                        "1e",
                        par_λ,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, red_output)
                end
            end
        end
    end
    return it_history, cost_history, sol
end

function eval_res(func, z)
    hat_γ = 1 / sum(1 ./ func.γ)
    N = func.N
    temp = copy(z)
    temp .*= N / hat_γ
    for i = 1:N
        ∇f, ~ = gradient(func.F[i], z)
        temp .-= ∇f
    end
    temp .*= hat_γ / N

    v, ~ = CIAOAlgorithms.prox(func.g, temp, hat_γ) # v= z^+

    return norm(v .- z) #/hat_γ is the criteria in proxSARAH, however, this criteria is very large when an unnormalized data is processed
end

function eval_gradient(func, z)
    println("grad test"); flush(stdout)
    ∇f_N = Flux.gradient(() -> func(z), params(z))[z] # already divided by N
    println("grad test ended"); flush(stdout)
    return norm(∇f_N) 
end



function loopnsave(iter, factor, Maxit, freq, plot_extras)

    R = eltype(x0) # element type  # is x0 kown in this scope? yes as the parent function has it
    res_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    # it_sum = Vector{R}(undef, 0)
    it_sum = 0
    sol = copy(x0)

    func = plot_extras.f
    x_star = plot_extras.x_star
    if x_star !== nothing
        f_star = func(x_star)         # sol
    end
    if plot_extras.eval_gradient
        resz = eval_gradient(func,x0)
    else
        resz = eval_res(func, x0,)
    end
    # initial point
    push!(cost_hist, func(x0))
    push!(res_hist, resz)
    push!(it_hist, 0)

    println("init point res: ", resz)
    flush(stdout)
    # println("x0:", x0[1:6])

    println("grads per iteration is $(factor) and maxit is $(Maxit)")
    flush(stdout)
    cnt = -1

    maxit = Maxit * factor / N # number of epochs

    for state in take(iter, Maxit |> Int)
        it_sum += it_counter(iter, state, plot_extras) # number of gradient evaluations
        
        if it_sum/N - factor/N >= maxit
            # println("gamma vec: $(state.γ/N)")
            break
        end

        if mod(cnt, freq) == 0 && cnt > 0 # cnt > 0 to skip the first iterates

            z = CIAOAlgorithms.solution(state)
            # z = state.z # uncomment for PANOC
            if plot_extras.eval_gradient
                resz = eval_gradient(func, z)
            else
                resz = eval_res(func, z)
            end
            cost = func(z)
            
            push!(res_hist, resz)
            if x_star !== nothing
                push!(cost_hist, cost - f_star )  # cost
            else
                push!(cost_hist, cost)  # cost
            end
            # println("current epoch is $(it_sum/ N) while using factor it is $(cnt * factor / N)")
            push!(it_hist, it_sum/N)
            gamma = isa(state.γ, R) ? state.γ : state.hat_γ
            println("epoch $(it_sum/N) cost is $(cost) - gamma: $(gamma) | gamma max: $(maximum(state.γ)/N) gamme min $(minimum(state.γ)/N) norm_0 $(norm(z,0))")
            flush(stdout)
            # println("epoch $(it_sum/N) cost is $(cost)")

            sol .= z
        end
        cnt += 1
    end
    return it_hist, cost_hist, res_hist, sol
end

####################################### iteration counters #######################################
it_counter(iter::CIAOAlgorithms.SGD_prox_iterable, state::CIAOAlgorithms.SGD_prox_state, plot_extras) = iter.N # it is =1 for each iter of SGD, but it is emplimented in this way for speed!
it_counter(iter::CIAOAlgorithms.SGD_prox_DNN_iterable, state::CIAOAlgorithms.SGD_prox_DNN_state, plot_extras) = iter.N # it is =1 for each iter of SGD, but it is emplimented in this way for speed!
it_counter(iter::CIAOAlgorithms.FINITO_basic_iterable, state::CIAOAlgorithms.FINITO_basic_state, plot_extras) = iter.N # it is =1 for each iter of basicFinito, but it is emplimented in this way for speed!
it_counter(iter::CIAOAlgorithms.FINITO_LFinito_iterable, state::CIAOAlgorithms.FINITO_LFinito_state, plot_extras) = iter.N + iter.N

function it_counter(iter::CIAOAlgorithms.FINITO_lbfgs_adaptive_DNN_iterable, state::CIAOAlgorithms.FINITO_lbfgs_adaptive_DNN_state, plot_extras)
        # return (3 + round(log(state.τ)/log(plot_extras.β))) * iter.N
        return (3 + round(log(1 / plot_extras.β, 1 / CIAOAlgorithms.epoch_count(state)))) * iter.N
        # return 2 * iter.N
end

function it_counter(iter::Union{CIAOAlgorithms.FINITO_lbfgs_iterable, CIAOAlgorithms.FINITO_lbfgs_adaptive_iterable}, state::Union{CIAOAlgorithms.FINITO_lbfgs_state, CIAOAlgorithms.FINITO_lbfgs_adaptive_state}, plot_extras)
    # return (3 + round(log(state.τ)/log(plot_extras.β))) * iter.N
    return (3 + round(log(1 / plot_extras.β, 1 / CIAOAlgorithms.epoch_count(state)))) * iter.N
    # return 2 * iter.N
end

function it_counter(iter::CIAOAlgorithms.FINITO_DFlbfgs_iterable, state::CIAOAlgorithms.FINITO_DFlbfgs_state, plot_extras)
        return (2 + iter.rep + round(log(1/plot_extras.β, 1 / CIAOAlgorithms.epoch_count(state))) ) * iter.N
end

it_counter(iter::CIAOAlgorithms.SAGA_basic_iterable, state::CIAOAlgorithms.SAGA_basic_state, plot_extras) = 1
it_counter(iter::CIAOAlgorithms.SAGA_prox_iterable, state::CIAOAlgorithms.SAGA_prox_state, plot_extras) = 2 * iter.N
it_counter(iter::CIAOAlgorithms.SVRG_basic_iterable, state::CIAOAlgorithms.SVRG_basic_state, plot_extras) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SVRG_basic_DNN_iterable, state::CIAOAlgorithms.SVRG_basic_DNN_state, plot_extras) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SARAH_basic_iterable, state::CIAOAlgorithms.SARAH_basic_state, plot_extras) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SARAH_prox_iterable, state::CIAOAlgorithms.SARAH_prox_state, plot_extras) = iter.N + iter.m
it_counter(iter::CIAOAlgorithms.SARAH_prox_DNN_iterable, state::CIAOAlgorithms.SARAH_prox_DNN_state, plot_extras) = iter.N + iter.m
function it_counter(iter::ProximalAlgorithms.PANOC_iterable, state::ProximalAlgorithms.PANOC_state, plot_extras)
   return  state.grad_eval * plot_extras.f.N
end
####################################### iteration counters #######################################

function loopnsave(iter::ProximalAlgorithms.ZeroFPR_iterable{R}, factor, Maxit, freq, plot_extras) where {R}

    res_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    it_lbfgs = Vector{R}(undef, 0)
    sol = copy(x0)

    func = plot_extras.f
    x_star = plot_extras.x_star
    if x_star !== nothing
        x_star = plot_extras.x_star
        f_star = func(x_star)         # sol
    end
    resz = eval_res(func, x0)
    # initial point
    push!(cost_hist, func(x0))
    push!(res_hist, resz)
    push!(it_hist, 0)

    println("-----------------ZeroFPR-------------")

    println("grads per iteration is $(factor) and maxit is $(Maxit)")
    # println(Maxit)
    cnt = -1
    for state in take(iter, Maxit + 2 |> Int)
        # println(ProximalAlgorithms.epoch_count(state))
        β = 2 # currently the linesearch backtrack par is 2 (division by 0.5)
        push!(it_lbfgs, 2 + round(log(β, 1 / ProximalAlgorithms.epoch_count(state))))
        if mod(cnt, freq) == 0 && cnt > 0

            z = ProximalAlgorithms.solution(state)

            cost = func(z)
            resz = eval_res(func, z)
            push!(res_hist, resz)

            if x_star !== nothing
                push!(cost_hist, cost - f_star )  # cost
            else
                push!(cost_hist, cost)  # cost
            end
            s = sum(it_lbfgs)
            push!(it_hist, s)  # devided by N to indicated # of passes
            println("epoch $(s) cost is $(cost)")

            sol .= z
        end
        cnt += 1
    end
    return it_hist, cost_hist, res_hist, sol
end
