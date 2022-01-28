
# function Residue_z!(H, F, g, z, γ::Array{Float64})

#     N = length(F)
#     NHZ = computeHatH(H, F, z, γ, N)  # nabla hat H(z) 
#     zplus, ~ = prox_Breg(H, g, NHZ, γ) # v= z^+

#     return norm( zplus .- z )^2
# end

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
    F::Array{Tf}            # smooth term
    g::Tg                   # nonsmooth term
    N::Int                  # number of data points in the finite sum problem
    n::Int                  # F[i] : R^n \to R
    γ::Union{Array{Float64}, Nothing}
    S::Ts
end

function Cost_FS(F::Array{Tf}, g::Tg, N, n, S::Ts) where {Tf, Tg, Ts}
    return Cost_FS{Tf, Tg,Ts}(
        F,
        g,
        N,
        n,
        nothing,
        S
    )
end


function (S::Cost_FS)(x)
    cost = S.g(x)
    # println("cost now: ", cost)
    # println("x now: ", x[1:2])
    cost = 0
    for i = 1:N
        cost += S.F[i](x) / S.N
    end
    return cost
end

struct saveplot{R}
    f                     # cost_FS
    β::R                # ls division parameter
    x_star #::Union{Tx, Nothing}
    str::String
end

# function computeHatH(H::Array{Th}, F, z, γ::Array{Float64}, N) where {Th}
#     temp = zero(z)
#     for i = 1:N
#         ∇h, ~ = BregmanBC.gradient(H[i], z)
#         temp .+= ∇h ./ γ[i]
#         ∇f, ~ = BregmanBC.gradient(F[i], z)
#         temp .-= ∇f ./ N
#     end
#     return temp
# end


# function computeOp(H::Array{Th}, F, z, v, γ::Array{Float64}, N) where {Th}
#     nmL = 0
#     for i = 1:N
#         # println(nmL)
#         ∇h, ~ = BregmanBC.gradient(H[i], z)
#         temp, ~ = BregmanBC.gradient(H[i], v)
#         ∇h .-= temp
#         ∇h ./=  γ[i]

#         temp, ~ = BregmanBC.gradient(F[i], z)
#         ∇h .-= temp ./ N
#         temp, ~ = BregmanBC.gradient(F[i], v)
#         ∇h .+= temp ./ N

#         if nmL <= 0.0
#             nmL += norm(∇h)^2
#         end
#         # println(nmL)
#     end
#     # println(nmL)
#     return nmL
# end






# function computeOp(H::Array{Th}, F, z, v, γ::Array{Float64}, N) where {Th}
#     nmL = 0
#     for i = 1:N
#         # println(nmL)
#         ∇h, ~ = BregmanBC.gradient(H[i], z)
#         temp, ~ = BregmanBC.gradient(H[i], v)
#         ∇h .-= temp
#         ∇h ./=  γ[i]

#         temp, ~ = BregmanBC.gradient(F[i], z)
#         ∇h .-= temp ./ N
#         temp, ~ = BregmanBC.gradient(F[i], v)
#         ∇h .+= temp ./ N

#         if nmL <= 0.0
#             nmL += norm(∇h)^2
#         end
#         # println(nmL)
#     end
#     # println(nmL)
#     return nmL
# end

function getresidue!(iter, tol, factor, Maxit, freq, γ::Array{Float64}, res_xz, H, plot_extras)

    dist_hist = Vector{R}(undef, 0)
    dist2_hist = Vector{R}(undef, 0)
    cost_hist = Vector{R}(undef, 0)
    it_hist = Vector{R}(undef, 0)
    sol_last = copy(x0)

    # initial residue
    if ~res_xz 
        cost = iter.g(x0)
        for i = 1:N
            cost += iter.F[i](x0) / N
        end
        D1, D2 = Residue_z!(H, iter.F, iter.g, x0, γ)
        push!(dist_hist, D1)
        push!(dist2_hist, D2)
        push!(cost_hist, cost)
        push!(it_hist, 0)
    end
    
    it_sum = 0
    cnt2 = -1
    maxit = Maxit * factor / N # number of epochs

    for state in take(iter, Maxit + 2 |> Int)
        it_sum += it_counter(iter, state, plot_extras) # number of gradient evaluations
        
        if it_sum/N - factor/N >= maxit
            # println("gamma vec: $(state.γ/N)")
            break
        end
        
        if mod(cnt2, freq) == 0 && cnt2 > 0
            # gamma = solution_γ(state)
            
            z = CIAOAlgorithms.solution(state)
            # epoch_cnt = solution_epoch(state)
            
            z_ = z #/ maximum(z)

            cost = iter.g(z_)
            for i = 1:N
                cost += iter.F[i](z_) / N
            end

            D1, D2 = Residue_z!(H, iter.F, iter.g, z, γ)

            push!(dist_hist, D1)    # z - v
            push!(dist2_hist, D2)   # nabla H(z) - nabla H(v) for v in prox(z)
            push!(cost_hist, cost)  # cost

            push!(it_hist, it_sum/N)  # indicating number of epochs

            println("iter $(it_sum) epoch $(it_sum/N) cost is $(cost) - residue1 is $(D1) residue2 is $(D2)") 
            nml = D1
            # if nml < tol
            #     if norm(z) < tol
            #         println("norm of solution is $(norm(z))...perhaps lambda is too large")
            #     end
            #     println("tolerance is met!")
            #     break
            # end
            sol_last .= z 
        end
        cnt2 += 1
    end
    return dist_hist, it_hist, sol_last, dist2_hist, cost_hist 
end


function Comparisons!(
    p::Real,
    H,
    func,
    L,
    stuff,
    Labelsweep,
    maxit::Int,
    tol::Float64,
    x0::Array{R},
    str,
    γ,
    res_xz,
    p_fail,
    λ,
    x_star,
    plot_extras
) where {R<:Real}

        # λ = func.g.lambda
    n = size(func.F[1].a, 1)
    N = size(func.F, 1)
    k = N/n |> Int

    dist_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cnt = 0
    t = length(stuff)
    for i = 1:t
        t2 = length(stuff[i]["sweeping"])
        for j = 1:t2 # loop for index update strategy  
            LFinito = stuff[i]["LFinito"]
            DeepLFinito = stuff[i]["DeepLFinito"]

            lbfgs = stuff[i]["lbfgs"]
            adaptive = stuff[i]["adaptive"]
            single_stepsize = stuff[i]["single_stepsize"]
            label = stuff[i]["label"]
            sweeping = stuff[i]["sweeping"][j]
            t3 = length(stuff[i]["minibatch"])

            for l = 1:t3 # loop for minibatch numbers
                cnt += 1
                minibatch = stuff[i]["minibatch"][l]
                size_batch = minibatch[2]
                println("solving using $(label) and with $(Labelsweep[sweeping]) sweeping.....")

                solver = CIAOAlgorithms.Finito{T}(
                    sweeping = sweeping,
                    LFinito = LFinito,
                    DeepLFinito = DeepLFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                    adaptive = adaptive,
                    β = plot_extras.β,
                    # γ = plot_extras.f.γ,
                )

                # if single_stepsize
                if single_stepsize
                    L_s = maximum(L)   # has to be divided since 1/L_F = \bar γ = (\sum 1/γ_i)^-1 = γ/N
                else
                    L_s = L
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
                    S = func.S,
                )

                # iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = H, L = L_s, N = N)
                dist_hist, it_hist, z_sol, dist2_hist, cost_hist  = getresidue!(iter, tol, factor, Maxit, freq, γ, res_xz, H, plot_extras)
                
                save_str = "ciao_N_$(N)_n_$(n)_p_$(p)_batch_$(size_batch)_$(stuff[i]["label"])_$(Labelsweep[sweeping])_Lratio_$(Int(floor(maximum(L) / minimum(L))))_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)_k_$(k)"

                #### ------------------------------------ solution visualization ---------------------------------
                println("sol min: ", minimum(z_sol))
                println("sol max: ", maximum(z_sol))
                Gray.(reshape(z_sol, 16,16))
                z_clamp = map(clamp01nan, z_sol)   # bring back to 0-1 range!
                z_clamp_ = map(clamp01nan, -z_sol)   # bring back to 0-1 range!
                save(string("solutions/datasets/",digits,"/",save_str,".png"), colorview(Gray, reshape(abs.(z_clamp), 16, 16)'))
                save(string("solutions/datasets/",digits,"/",save_str,"_.png"), colorview(Gray, reshape(abs.(z_clamp_), 16, 16)'))
                #### ---------------------------------------------------------------------------------------------

                push!(dist_history, dist_hist)
                push!(it_history, it_hist)
                # saving 
                output = [it_history[end] dist_history[end]]
                d = length(dist_history[end])
                rr = Int(ceil(d / 50)) # keeping at most 50 data points
                red_output = output[1:rr:end, :]

                open(
                    string(
                        "plot_data/",
                        str,
                        "zzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, red_output)
                end

                output = [it_history[end] dist2_hist]
                red_output = output[1:rr:end, :]

                open(
                    string(
                        "plot_data/",
                        str,
                        "Hzzplus/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, red_output)
                end

                output = [it_history[end] cost_hist]
                red_output = output[1:rr:end, :]

                open(
                    string(
                        "plot_data/",
                        str,
                        "cost/",
                        save_str,
                        ".txt",
                    ),
                    "w",
                ) do io
                    writedlm(io, red_output)
                end
            end
        end
    end
    return dist_history, it_history
end

function it_counter(iter::Union{CIAOAlgorithms.FINITO_lbfgs_iterable, CIAOAlgorithms.FINITO_lbfgs_adaptive_iterable}, state::Union{CIAOAlgorithms.FINITO_lbfgs_state, CIAOAlgorithms.FINITO_lbfgs_adaptive_state}, plot_extras)
    # return (3 + round(log(state.τ)/log(plot_extras.β))) * iter.N
    return (3 + round(log(1 / plot_extras.β, 1 / CIAOAlgorithms.epoch_count(state)))) * iter.N
    # return 2 * iter.N
end