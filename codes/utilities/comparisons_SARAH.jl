

R = eltype(x0)

stuff = [
  Dict( #SARAH
        "label"             =>  "SARAH",
        "γ"                 =>  nothing,
        "m"                 =>  N,
        "DNN"               =>  false,
        "L"                 => [maximum(L)]
      ),
      ]

Labelsweep =["SARAH"] 
t = length(stuff)
cost_history = Vector{Vector{R}}(undef, 0)
res_history = Vector{Vector{R}}(undef, 0)
it_history = Vector{Vector{R}}(undef, 0)

for i in 1:t
  t2= length(stuff[i]["L"])
  for j in 1:t2 # loop for different number of inner iterations
    label = stuff[i]["label"]
    γ_sarah = stuff[i]["γ"]
    N_ = stuff[i]["m"]
    L = stuff[i]["L"][j]
    DNN_flag = stuff[i]["DNN"]
    println("using $(label) with L = $(L) ...")

    m_inner = N_ # number of inner cycles
    solver = CIAOAlgorithms.SARAH{R}(γ = γ_sarah, m = m_inner, DNN = DNN_flag)
    iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, L = L, N = N)

    factor = m_inner + N # number of grad evals per iteration
    Maxit = Int(ceil(maxit * N / factor))
    freq = 1 #Int(ceil(Maxit / 100)) # keeping at most 100 data points

    it_hist, cost_hist, res_hist, sol =
                    loopnsave(iter, factor, Maxit, freq, plot_extras) #in utilities

    push!(cost_history, cost_hist)
    push!(res_history, res_hist)
    push!(it_history, it_hist)

    # saving 
    output = [it_history[end] cost_history[end]]
    d = length(cost_hist)
    rr = 1 #Int(ceil(d / 50)) # keeping at most 50 data points
    red_output = output[1:rr:end, :]

    # if λ === 0
    #     par_λ = 0
    # else
    #     if plot_extras.x_star ===nothing
    #         par_λ = log10(round(λ * N, digits = 5)) |> Int
    #     else
    #         par_λ = log10(round(λ, digits = 5)) |> Int
    #     end
    # end

    open(
        string(
            "plot_data/",
            str,
            "SARAH/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_L_",
            L,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, red_output)
    end

    #
    output = [it_history[end] res_history[end]]
    d = length(res_hist)
    rr = 1 #Int(ceil(d / 50)) # keeping at most 50 data points
    red_output = output[1:rr:end, :]


    open(
        string(
            "plot_data/",
            str,
            "SARAH/res/",
            "N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_L_",
            L,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, red_output)
    end
    # error_cnt = 0
    # for i in 1:N
    #     if (argmax(softmax(predict(x_train[i,:]',sol)')) != argmax(y_train[i,:]))
    #         error_cnt += 1
    #     end
    # end
    # println("SARAH: Number of errors is $(error_cnt) out of $(N) samples\n")
    # flush(stdout)
  end 
end