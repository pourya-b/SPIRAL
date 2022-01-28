
R = eltype(x0) # element type

stuff = [
    # Dict( #SAG
    #     "label" => "SAG",
    #     "m"                 => [N]
    # ),
    Dict( #SAGA
        "label" => "SAGA",
        "m"                 => [N],
        "prox" => true,
    ),
]

γ_SAGA = 1 / (5 * N * maximum(L))

data = [];

Labelsweep = ["SAG", "SAGA"]

t = length(stuff)

numlines = 0
for i = 1:t
    global numlines += length(stuff[i]["m"])
end

cost_history = Vector{Vector{R}}(undef, 0)
res_history = Vector{Vector{R}}(undef, 0)
it_history = Vector{Vector{R}}(undef, 0)

cnt = 0
for i = 1:t # loop for the stuff
    t2 = length(stuff[i]["m"])
    for j = 1:t2
        # plus = stuff[i]["plus"]
        label = stuff[i]["label"]
        # γ = stuff[i]["γ"]
        N_ = stuff[i]["m"][j]
        prox_ = stuff[i]["prox"]
        global cnt += 1
        println("solving using $(label) and with m= $(N_).....")


        if label == "SAGA"
            println("prox_", prox_)
            solver = CIAOAlgorithms.SAGA{R}(γ = γ_SAGA, prox_flag = prox_)
            iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, N = N)
            str2 = "SAGA/"
        elseif label == "SAG"
            # γ = 1 / (16 * maximum(L))
            solver = CIAOAlgorithms.SAG(R)
            iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, N = N, L=L)
            str2 = "SAG/"
        end

        factor = 2 * N #(2 gradient evaluation per loop, N loops per iteration for speed) 
        Maxit = Int(ceil(maxit * N / factor))
        freq = 1 # keeping at most 100 data points

        lbfgs = false

        it_hist, cost_hist, res_hist, ~ = loopnsave(iter, factor, Maxit, freq, plot_extras)

        push!(cost_history, cost_hist)
        push!(res_history, res_hist)
        push!(it_history, it_hist)

        # saving
        output = [it_history[end] cost_history[end]]
        d = length(cost_hist)
        rr = Int(ceil(d / 50)) # keeping at most 50 data points
        red_output = output[1:rr:end, :]

        if λ === 0
            par_λ = 0
        else
            if plot_extras.x_star ===nothing
                par_λ = log10(round(λ * N, digits = 5)) |> Int
            else
                par_λ = log10(round(λ, digits = 5)) |> Int
            end
        end

        open(
            string(
                "plot_data/",
                str,
                str2,
                "cost/",
                "_N_",
                N,
                "_n_",
                n,
                "_",
                stuff[i]["label"],
                "_",
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


        # residual
        output = [it_history[end] res_history[end]]
        d = length(cost_hist)
        rr = Int(ceil(d / 50)) # keeping at most 50 data points
        red_output = output[1:rr:end, :]


        open(
            string(
                "plot_data/",
                str,
                str2,
                "res/",
                "_N_",
                N,
                "_n_",
                n,
                "_",
                stuff[i]["label"],
                "_",
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
