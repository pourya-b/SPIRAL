# using RollingFunctions
# T = Float64
# RollingFunctions.float(::Type{Union{T,Missing}}) where {T} = Base.float(Float64)
batchmemaybe(x) = tuple(x)
batchmemaybe(x::Tuple) = x

function eval_DNN(str = nothing)
    error_cnt = 0
    for i in 1:N
        if (argmax(softmax(predict(x_train[i,:]))) != argmax(y_train[i,:]))
            error_cnt += 1
        end
    end
    println(string(str, ": Number of errors is $(error_cnt) out of $(N) samples\n")); flush(stdout)
end

function return_model_params(model)
    ps = params(model)
    ps_len = length(ps)
    struc = []
    for j in 1:2:ps_len-2
        push!(struc,Dense(size(ps[j],2),size(ps[j],1),tanh))
    end
    j = ps_len-1
    push!(struc,Dense(size(ps[j],2),size(ps[j],1)))

    dummy = Chain(struc) 
    ds = params(dummy)
    for i in 1:ps_len
        ds[i] .= ps[i]
    end
    return ds
end

function DNN_config!(;gs=nothing) # in: nothing out-> vectorized DNN params/ in: gs=grads out -> vectorized grad/ in:gs=params out -> nothing [sets DNN params by gs]
    inx = 1
    ps_len = length(ps)
    config_length = 0
    for j in 1:2:ps_len
        config_length += (size(ps[j],2)+1) * size(ps[j],1)
    end

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
        for i=1:ps_len
            Len = length(ps[i])
            ps[i] .= reshape(gs[inx:inx+Len-1],size(ps[i]))
            inx += Len
        end
        return nothing
    end
end

struct Cost_FS{Tf,Tg,Ts}
    F::Union{Tf,Array{Tf}}           # smooth term
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

function Cost_FS(F::Union{Tf,Array{Tf}}, g::Tg, N, n, γ, S::Ts) where {Tf, Tg, Ts}
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
    w = DNN_config!()
    cost = g(w)
    cost += NewLoss(batchmemaybe(x)...)
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
    x0::Union{Array{R},Tp},
    L,
    plot_extras,
    x_train,
    y_train,
    w0,
) where {R<:Real,Tp}
# R = eltype(w0)
n = func.n
    N = func.N

    cost_history = Vector{Vector{T}}(undef, 0)
    res_history = Vector{Vector{T}}(undef, 0)
    it_history = Vector{Vector{T}}(undef, 0)
    cnt = 0 # counter
    t = length(stuff)
    sol = copy(w0)
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
            DNN_flag = stuff[i]["DNN"]
            t3 = length(stuff[i]["minibatch"])

            for l = 1:t3 # loop for minibatch numbers
                cnt += 1
                minibatch = stuff[i]["minibatch"][l]
                size_batch = minibatch[2]
                println("solving using $(label) and with $(Labelsweep[sweeping]) sweeping....."); flush(stdout)

                L_finito = 1.0
                solver = CIAOAlgorithms.Finito{T}(
                    γ = 0.999/L_finito,
                    sweeping = sweeping,
                    LFinito = LFinito,
                    DeepLFinito = DeepLFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = plot_extras.β,
                    DNN_training = DNN_flag,
                    # γ = plot_extras.f.γ,
                )

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
                    F = NewLoss,
                    g = func.g,
                    N = N,
                    data = data,
                    DNN_config = DNN_config!
                )

                it_hist, cost_hist, res_hist, sol =
                    loopnsave(iter, factor, Maxit, freq, plot_extras) # here is the main iterations

                push!(cost_history, cost_hist)
                push!(res_history, res_hist)
                push!(it_history, it_hist)

                # saving
                output = [it_history[end] cost_history[end]]
                d = length(cost_hist)
                rr = 1 #Int(ceil(d / 50)) # keeping at most 50 data points
                red_output = output[1:rr:end, :] #reduced

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
                mkpath(string("plot_data/",str,"/cost"))
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
                        "_L_",
                        L_finito,
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
               
                mkpath(string("plot_data/",str,"/res"))
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
                        "_L_",
                        L_finito,
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

function eval_gradient()
    # println("grad test*"); flush(stdout)
    hat_γ = 1e-3
    z = DNN_config!()
    v = copy(z)
    gr = gradient(ps) do # sampled gradient
        ADAMLoss(batchmemaybe(data)...)
    end
    ∇f_N = DNN_config!(gs=gr) # already divided by N
    z = z - hat_γ * ∇f_N
    prox!(v, g, z, hat_γ)
    # println("grad test* ended"); flush(stdout)
    return norm(z - v) 
end

function loopnsave(iter, factor, Maxit, freq, plot_extras)

    R = eltype(w0) # element type  # is x0 kown in this scope? yes as the parent function has it
    res_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    # it_sum = Vector{R}(undef, 0)
    it_sum = 0
    sol = copy(w0)

    func = plot_extras.f
    x_star = plot_extras.x_star
    resz = eval_gradient()
    # initial point
    push!(cost_hist, func(data))
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
            resz = eval_gradient()
            cost = func(data)
            
            push!(res_hist, resz)
            push!(cost_hist, cost)  # cost
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
it_counter(iter::CIAOAlgorithms.GD_prox_DNN_iterable, state::CIAOAlgorithms.GD_prox_DNN_state, plot_extras) = iter.N # it is =1 for each iter of SGD, but it is emplimented in this way for speed!
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