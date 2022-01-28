

R = eltype(x0)

stuff = [
  Dict( #SGD
        "label"             => "SGD",
        "γ"                 => nothing,
        "plus"              => true, # true for diminishing stepsize, false for constant stepsize
        "DNN"               => false,
        "η0"                => 0.1, # for the stepsize: η0/(η_tilde + iter)
        "η_tilde"           => 0.5,
      ),
      ]

Labelsweep =["SGD"] 
t = length(stuff)
cost_history = Vector{Vector{R}}(undef, 0)
res_history = Vector{Vector{R}}(undef, 0)
it_history = Vector{Vector{R}}(undef, 0)

for i in 1:t
    label = stuff[i]["label"]
    γ_sgd = stuff[i]["γ"]
    plus = stuff[i]["plus"]
    DNN_flag = stuff[i]["DNN"]
    η0 = stuff[i]["η0"]
    η_tilde = stuff[i]["η_tilde"]
    println("using $(label) ...")

    solver = CIAOAlgorithms.SGD{R}(γ=γ_sgd, plus=plus, DNN=DNN_flag, η0=η0, η_tilde=η_tilde)
    iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, L = nothing, N = N, DNN_config = false)

    factor = N # (SGD is implemented in this way for speed) number of grad evals per iteration
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
    # if L == nothing
    #     L = 1
    # end
    L = 1

    open(
        string(
            "plot_data/",
            str,
            "SGD/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_",
            "η_",
            η0,
            "η_tilde_",
            η_tilde,
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
            "SGD/res/",
            "N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_",
            "η_",
            η0,
            "η_tilde_",
            η_tilde,
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
    # println("SGD: Number of errors is $(error_cnt) out of $(N) samples\n")
    # flush(stdout)
end 