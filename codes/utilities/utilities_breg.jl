
# function Residue_z!(H, F, g, z, γ::Array{Float64})

#     N = length(F)
#     NHZ = computeHatH(H, F, z, γ, N)  # nabla hat H(z) 
#     zplus, ~ = prox_Breg(H, g, NHZ, γ) # v= z^+

#     return norm( zplus .- z )^2
# end


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


function Residue_z!(H, F, g, z, γ::Array{Float64})

    N = length(F)
    NHZ = computeHatH(H, F, z, γ, N)  # nabla hat H(z) 
    v, ~ = prox_Breg(H, g, NHZ, γ) # v= z^+

    # Op = computeOp(H, F, z, v, γ, N)  # nabla hat H(z) 

    NHZ2 = computeHatH(H, F, v, γ, N)  # nabla hat H(z) 

    return norm( v .- z ), norm(NHZ2 .- NHZ)
end


function computeHatH(H::Array{Th}, F, x, γ::Array{Float64}, N) where {Th}
    temp = zero(x)
    for i = 1:N
        ∇h, ~ = BregmanBC.gradient(H[i], x)
        temp .+= ∇h ./ γ[i]
        ∇f, ~ = BregmanBC.gradient(F[i], x)
        temp .-= ∇f ./ N
    end
    return temp
end



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

function getresidue_breg!(iter, tol, factor, Maxit, freq, γ::Array{Float64}, res_xz, H)

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
    
    maxit = Maxit * factor / N # number of epochs
    cnt2 = -1
    epoch_cnt = 0
    for state in take(iter, Maxit + 2 |> Int)

        if epoch_cnt - factor/N >= maxit
            break
        end

        if mod(cnt2, freq) == 0 && cnt2 > 0
            # gamma = solution_γ(state)
            
            z = BregmanBC.solution(state)
            epoch_cnt = solution_epoch(state)
            
            z_ = z #/ maximum(z)

            cost = iter.g(z_)
            for i = 1:N
                cost += iter.F[i](z_) / N
            end

            D1, D2 = Residue_z!(H, iter.F, iter.g, z, γ)

            push!(dist_hist, D1)    # z - v
            push!(dist2_hist, D2)   # nabla H(z) - nabla H(v) for v in prox(z)
            push!(cost_hist, cost)  # cost

            push!(it_hist, epoch_cnt)  # indicating number of epochs

            println("iter $(cnt2) epoch $(epoch_cnt) cost is $(cost) residue1 is $(D1) residue2 is $(D2)") 
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


function Comparisons_breg!(
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
    x_star
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
            LBFinito = stuff[i]["LBFinito"]
            single_stepsize = stuff[i]["single_stepsize"]
            label = stuff[i]["label"]
            sweeping = stuff[i]["sweeping"][j]
            t3 = length(stuff[i]["minibatch"])
            lbfgs = stuff[i]["lbfgs"]
            for l = 1:t3
                cnt += 1
                minibatch = stuff[i]["minibatch"][l]
                size_batch = minibatch[2]
                println("solving using $(label) and with $(Labelsweep[sweeping]) sweeping.....")
                solver = BregmanBC.Bregman_Finito{R}(
                    sweeping = sweeping,
                    LBFinito = LBFinito,
                    minibatch = minibatch,
                    lbfgs = lbfgs,
                )

                # if single_stepsize L = maximum(L) end # single stepsize
                if single_stepsize
                    L_s = sum(L) / N   # has to be divided since 1/L_F = \bar γ = (\sum 1/γ_i)^-1 = γ/N 
                    # L_s = maximum(L)
                    println("single step size")
                else
                    L_s = L
                end
                # if low memory then every itr is 2N updates
                if LBFinito
                    factor = 2 * N
                    Maxit = Int(ceil(maxit * N/ factor))
                    freq = 1
                elseif lbfgs
                    println("batch sizee is $size_batch")
                    factor = 3 * N
                    Maxit = Int(ceil(maxit * N/ factor))
                    freq = 1
                else
                    println("batch size is $size_batch")
                    factor = 1
                    Maxit = Int(ceil(maxit * N/ factor))
                    freq = N # keeping at most 50 data points
                end

                iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = H, L = L_s, N = N)
                dist_hist, it_hist, z_sol, dist2_hist, cost_hist  = getresidue_breg!(iter, tol, factor, Maxit, freq, γ, res_xz, H)
                
                save_str = "breg_N_$(N)_n_$(n)_p_$(p)_batch_$(size_batch)_$(stuff[i]["label"])_$(Labelsweep[sweeping])_Lratio_$(Int(floor(maximum(L) / minimum(L))))_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)_k_$(k)"

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
                output = [it_history[cnt] dist_history[cnt]]
                d = length(dist_history[cnt])
                rr = Int(ceil(d / 50)) # keeping at most 50 data points
                red_output = output[1:rr:end, :]

                mkpath(string("plot_data/",str,"zzplus/"))
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

                output = [it_history[cnt] dist2_hist]
                red_output = output[1:rr:end, :]

                mkpath(string("plot_data/",str,"Hzzplus/"))
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

                output = [it_history[cnt] cost_hist]
                red_output = output[1:rr:end, :]

                mkpath(string("plot_data/",str,"cost/"))
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



function SMD!(
    p::Real,
    H,
    h,
    func,
    L,
    maxit::Int,
    tol::Float64,
    x0::Array{R},
    str,
    alg,
    diminishing,
    γ,
    res_xz,
    p_fail,
    λ
) where {R<:Real}
    
    # λ = func.g.lambda
    n = size(func.F[1].a, 1)
    N = size(func.F, 1)
    k = N/n |> Int

    dist_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cnt = 0

    println("solving using stochastic mirror descent.....")
    if alg == "SMD"
        if diminishing
            solver = BregmanBC.SMD{R}(diminishing = true)
            println("-----------------solving with diminishing stepsize------------")
        else
            solver = BregmanBC.SMD{R}()
            println("-----------------solving with constant stepsize------------")
        end
    else
        solver = BregmanBC.PLIAG{R}()
        println("-----------------solving with PLIAG------------")
    end
    size_batch = 1 # for now 
    factor = size_batch
    Maxit = floor(maxit / factor)
    freq = Int(ceil(Maxit / 100)) # keeping at most 50 data points

    iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = h, L = L, N = N)

    dist_hist, it_hist, ~, dist2_hist, cost_hist= getresidue!(iter, tol, factor, Maxit, freq, γ, res_xz, H)


    push!(dist_history, dist_hist)
    push!(it_history, it_hist)
    # saving 
    cnt = 1
    output = [it_history[cnt] dist_history[cnt]]
    d = length(dist_history[cnt])
    rr = Int(ceil(d / 50)) # keeping at most 50 data points
    red_output = output[1:rr:end, :]

    open(
        string(
            "plot_data/",
            str,
            "zzplus/",
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 5),
            "_pfail_",
            p_fail,
            "_k_",
            k,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, red_output)
    end
        output = [it_history[cnt] dist2_hist]
        red_output = output[1:rr:end, :]

        open(
           string(
            "plot_data/",
            str,
            "Hzzplus/",
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 5),
            "_pfail_",
            p_fail,
            "_k_",
            k,
            ".txt",
        ),
            "w",
        ) do io
            writedlm(io, red_output)
        end

        output = [it_history[cnt] cost_hist]
        red_output = output[1:rr:end, :]

        open(
           string(
            "plot_data/",
            str,
            "cost/",
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 5),
            "_pfail_",
            p_fail,
            "_k_",
            k,
            ".txt",
        ),
            "w",
        ) do io
            writedlm(io, red_output)
        end
end


function Comparisons_SMD!(
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
    x_star
) where {R<:Real}
    # λ = func.g.lambda
    n = size(func.F[1].a, 1)
    N = size(func.F, 1)
    k = N/n |> Int

    dist_history = Vector{Vector{R}}(undef, 0)
    it_history = Vector{Vector{R}}(undef, 0)
    cnt = 1
    

    println("solving using SMD ...")
    solver = BregmanBC.SMD{R}(
        diminishing = true,
    )

    factor = 1/N
    Maxit = floor(N * maxit / factor)
    freq = 1 # keeping at most 50 data points

    iter = BregmanBC.iterator(solver, x0, F = func.F, g = func.g, H = H[1], L = L[1], N = N)
    dist_hist, it_hist, z_sol, dist2_hist, cost_hist  = getresidue_breg!(iter, tol, factor, Maxit, freq, γ, res_xz, H)
    
    save_str = "SMD_N_$(N)_n_$(n)_p_$(p)_Lratio_$(Int(floor(maximum(L) / minimum(L))))_lambda_$(round(λ * N, digits = 5))_pfail_$(p_fail)_k_$(k)"

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
    output = [it_history[cnt] dist_history[cnt]]
    d = length(dist_history[cnt])
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

    output = [it_history[cnt] dist2_hist]
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

    output = [it_history[cnt] cost_hist]
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
    return dist_history, it_history
end


function accurate_sol(F, H, g, L, maxit::Int, x0::Array{R}, str, λ, p) where {R<:Real}
    N = size(F, 1)
    solver = BregmanBC.Bregman_Finito{R}(maxit = maxit, sweeping = 2)
    @time x_finito, it_finito = solver(x0, F = F, g = g, H = H, L = L, N = N)

    # computing the cost 
    cost = g(x_finito)
    for i = 1:N
        cost += F[i](x_finito) / N
    end
    println("cost is $(cost)")

    # #### saving

    open(
        string(
            str,
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 4),
            "_cost_",
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, cost)
    end
    open(
        string(
            str,
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 4),
            "_x_",
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, x_finito)
    end
    save(
        string(
            str,
            "_N_",
            N,
            "_n_",
            n,
            "_p_",
            p,
            "_Lratio_",
            Int(floor(maximum(L) / minimum(L))),
            "_lambda_",
            round(λ * N, digits = 4),
            ".png",
        ),
        colorview(Gray, reshape(x_finito, 16, 16)),
    )
    return cost, x_finito
end



function initializeX(A, b, N)
    # initial point according to Wang!

    # # # # # # Duchi alg2.....see Davis 2018 Thoerem 3.8 also
    # N = size(A,1)
    # idx = findall(x -> x <= mean(b), b)
    # X = zeros(n,n)
    # for i in idx
    #   a_i = A[i,:]
    #   X .+=  a_i * a_i' 
    # end
    # en = eigen(X)
    # d = en.vectors[:, 1]
    # x0 = sqrt(mean(b))*d
    
    # # # # # # # # # # initial point according to Wang!
    temp = Vector{R}(undef, 0)
    for i = 1:N
        push!(temp, b[i] / norm(A[i, :])^2)
    end
    # stemp = sort(temp)
    # quant56 = stemp[Int(ceil(5* N/6))]
    quant56 = quantile(temp, 5 / 6)

    idx = findall(x -> x >= quant56, temp)
    X = zeros(n, n)
    for i in idx
        a_i = A[i, :]
        X .+= a_i * a_i' ./ norm(a_i)^2
    end
    en = eigen(X)
    d = en.vectors[:, end]
    # @test d'*X*d - maximum(en.values) < 1e-6
    x0 = sqrt(sum(b) / N) * d

    #     # # # # # # initial point according to Zhang!

    # lam =  sqrt( quantile(b, 1/2) / 0.455 )  
    # idx = findall(x -> abs.(x) <= 9 * lam, b)    

    # X = zeros(n, n)
    # for i in idx
    #     X .+= b[i] * A[i, :] * A[i, :]'
    # end
    # X ./= N 
    # en = eigen(X)
    # d = en.vectors[:, end]
    # # @test d'*X*d - maximum(en.values) < 1e-6
    # x0 = lam * d

    # x0 ./= maximum(x0) ########### I added!     
    return x0
end




function Ab_image(x_star,k, p_fail)
    n = length(x_star)
    N = k * n

    # Hadamand matrices
    Hdmd = hadamard(n) ./ sqrt(n)
    HS = Vector{Array}(undef, 0)
    for i = 1:k
        # S = diagm(0 => rand([1, -1], n))
        S = diagm(0 => sign.(randn(n)))
        # println(S[1:3,1:3])
        push!(HS,  Hdmd * S)
        # push!(HS,  S * Hdmd )
    end
    A = vcat(HS...)

    # generating b
    b = (A * x_star) .^ 2
    # curropt with probability p_fail
    for i in eachindex(b)
        # println("b_i")
        if rand() < p_fail
            b[i] = 0
            # println("b_i zero")
        end
    end
    return A, b
end 



function Ab_rand(x_star,k, p_fail)
    n = length(x_star)
    N = k * n

    A = randn(N, n)
    # generating b
    b = (A * x_star) .^ 2
    # curropt with probability p_fail
    for i in eachindex(b)
        if rand() < p_fail
            b[i] = 0
        end
    end
    return A, b
end 