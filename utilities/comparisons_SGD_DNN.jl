

R = eltype(λ)

stuff = [
  Dict( #SGD
        "label"             => "SGD",
        "γ"                 => nothing,
        "plus"              => true, # true for diminishing stepsize, false for constant stepsize
        "GD"                => false,
        "DNN"               => true,
        "η0"                => η0,
        "η_tilde"           => η_tilde,
      ),
      ]

println("version: $(stuff[1]["GD"])")
Labelsweep =["SGD"] 
t = length(stuff)
cost_history = Vector{Vector{R}}(undef, 0)
res_history = Vector{Vector{R}}(undef, 0)
it_history = Vector{Vector{R}}(undef, 0)

for i in 1:t
    label = stuff[i]["label"]
    γ_sgd = stuff[i]["γ"]
    plus = stuff[i]["plus"]
    GD = stuff[i]["GD"]
    DNN_flag = stuff[i]["DNN"]
    η0 = stuff[i]["η0"]
    η_tilde = stuff[i]["η_tilde"]
    println("using $(label) ...")

    solver = CIAOAlgorithms.SGD{R}(γ=γ_sgd, plus=plus, GD=GD, DNN=DNN_flag, η0=η0, η_tilde=η_tilde)
    iter = CIAOAlgorithms.iterator(solver, x0, F=NewLoss, g=g, N=N, data=data, DNN_config=DNN_config!)

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
    mkpath(string("plot_data/",str,"/SGD/cost"))
    open(
        string(
            "plot_data/",
            str,
            "SGD/cost/",
            "DNN_N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_",
            "eta_",
            η0,
            "eta_tilde_",
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
    
    mkpath(string("plot_data/",str,"/SGD/res"))
    open(
        string(
            "plot_data/",
            str,
            "SGD/res/",
            "DNN_N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_",
            "eta_",
            η0,
            "eta_tilde_",
            η_tilde,
            ".txt",
        ),
        "w",
    ) do io
        writedlm(io, red_output)
    end
end 