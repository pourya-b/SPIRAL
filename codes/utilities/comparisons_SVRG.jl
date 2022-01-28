

R = eltype(x0)

# γ_SVRG = γ_Reddi

stuff = [
  Dict( #SVRG
        "label"             =>  "SVRG",
        "γ"                 =>  nothing,
        "plus"              => true, # true for convex SVRG or SVRG+ with m=sqrt(b) - false for nonconvex Reddi
        "m"                 => N,
        "DNN"               => false,
        "L"                 => [maximum(L)]
      ),
  #   Dict( #SVRG
  #       "label"             =>  "Reddi",
  #       "γ"                 =>  γ_Reddi,
  #       "plus"              => false, 
  #       "m"                 => [N]
  #     ),
  # Dict( #SVRG++
  #       "label"             =>  "Zhu",
  #       "γ"                 =>  γ_Zhu,
  #       "plus"              => true,
  #       # "m"                 => [N]
  #       "m"                 => Int64[floor(N),floor(N/2), floor(N/4), floor(N/10)]
  #     ),
      ]
data = []; 

Labelsweep =["SVRG_Reddi","SVRG++"] 

t = length(stuff)

numlines = 0 
for i in 1:t
  global numlines += length(stuff[i]["m"])
end 

cost_history = Vector{Vector{R}}(undef, 0)
res_history = Vector{Vector{R}}(undef, 0)
it_history = Vector{Vector{R}}(undef, 0)

cnt = 0
for i in 1:t
  t2= length(stuff[i]["L"])
  for j in 1:t2 # loop for index update strategy  
    plus = stuff[i]["plus"]
    label = stuff[i]["label"]
    γ_svrg = stuff[i]["γ"]
    DNN_flag = stuff[i]["DNN"]
    N_ = stuff[i]["m"]
    L_M = stuff[i]["L"][j]
    global  cnt += 1
    println("using $(label) with L = $(L_M) ...")

    γ_SVRG = 1/(7*maximum(L)) 
    γ_Reddi = 1/(3*N*maximum(L)) # according to thm 1 of "Proximal Stochastic Methods for Nonsmooth Nonconvex Finite-Sum Optimization "

    if convex_flag 
      γ_svrg = γ_SVRG
    else
      γ_svrg = γ_Reddi
    end

    m_inner = N_ # number of inner cycles
    solver = CIAOAlgorithms.SVRG{R}(γ = γ_svrg, m = m_inner, DNN = DNN_flag)
    iter = CIAOAlgorithms.iterator(solver, x0, F = F, g = g, N = N)

    factor = m_inner + N # number of grad evals per iteration
    Maxit = Int(ceil(maxit * N / factor))
    freq = 1 #Int(ceil(Maxit / 1000)) # keeping at most 50 data points

    lbfgs = false

    it_hist, cost_hist, res_hist, sol =
                    loopnsave(iter, factor, Maxit, freq, plot_extras)

    #--------------------- visualization for lasso -------------------
    # println("visualizing the test data ...")
    # data_ = CSV.read("../../../../datasets/mnist_test.csv", DataFrame)
    # data_ = Matrix(data_)
    # A_ = 1.0 * (data_[1:10,2:end]) # removing labels (MNIST)
    # b_ = 1.0 * (data_[1:10,1])

    # N_, n_ = size(A_, 1), size(A_, 2)
    # println(N_)
    # println(n_)

    # N_, n_ = size(sol, 1), size(sol, 2)
    # println(N_)
    # println(n_)

    
    # b_pre = real(dot(A_[1,:],sol))
    # println(b_pre)
    # println(b[1])
    #--------------------- visualization for lasso -------------------
    
    
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
    par_λ = 1

    open(
        string(
            "plot_data/",
            str,
            "SVRG/cost/",
            "N_",
            N,
            "_n_",
            n,
            "_",
            stuff[i]["label"],
            "_",
            "plus",
            "_",
            stuff[i]["plus"],
            "L_",
            L_M,
            ".txt",
        ),
      "w",
    ) do io
        writedlm(io, red_output)
    end

    #
    output = [it_history[end] res_history[end]]
    d = length(res_hist)
    rr = 1 #Int(ceil(d / 500)) # keeping at most 50 data points
    red_output = output[1:rr:end, :]


    open(
        string(
          "plot_data/",
          str,
          "SVRG/res/",
          "N_",
          N,
          "_n_",
          n,
          "_",
          stuff[i]["label"],
          "_",
          "plus",
          "_",
          stuff[i]["plus"],
          "L_",
          L_M,
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
    # println("SVRG: Number of errors is $(error_cnt) out of $(N) samples\n")
    # flush(stdout)
  end 
end



# using Plots
# plt=plot(xlabel= "iterations", ylabel= "cost",
#  legendfontsize = 6, legend = :topright, titlefontsize = 8)
# linestyles= [:solid, :solid, :dash, :dash, :dashdot, :dashdotdot, :dashdotdot ];
# bmarkerstyles= [:circle, :rect, :star5, :diamond] #, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x];
# linecolor = [:cornflowerblue, :red, :green1,:gray, :violet, :cyan, :magenta ]

# for i = 1:numlines
#          plot!(plt, red_output[i][:,1],red_output[i][:,2], yaxis =:log,linestyle = linestyles[i],
#            linecolor = linecolor[i], markersize = 2, markeralpha = 0.8)
#        gui()
#        sleep(0.3)
# end 
# savefig("SVRG.pdf")

# using DelimitedFiles

# cnt = 0 
# for i = 1:t
#     for j in 1: length(stuff[i]["m"])
#       global cnt +=1
#    open(string("plot_data/",str,"SVRG/","_N_",N,"_",stuff[i]["label"]
#     ,"_m_",stuff[i]["m"][j],"_Lratio_",Int(floor(maximum(L)/minimum(L))),"_lambda_",Int(round(lam*10)),".txt"),"w") do io 
#           writedlm(io, red_output[cnt])
#   end
# end 
# end





















# solver = StoInc.SVRG{Float64}(maxit=maxit |> Int, tol=1e-6, γ= γ_Zhu, m= N/10 |> Int, 
#       verbose=true,freq=10000, report_data = (true, data_freq), plus = plus ) 
#           @time x_SVRG, it_SVRG, sol_hist, = solver(F, g, x0, N=N)
# println(norm(A*x0-b)^2/2+ g(x_SVRG))





# using Plots
# plt=plot(xlabel= "passes through the data", ylabel= "cost",
#  legendfontsize = 6, legend = :topright, titlefontsize = 8)
# linestyles= [:solid, :solid, :dash, :dash, :dashdot, :dashdotdot ];
# bmarkerstyles= [:circle, :rect, :star5, :diamond] #, :hexagon, :cross, :xcross, :utriangle, :dtriangle, :rtriangle, :ltriangle, :pentagon, :heptagon, :octagon, :star4, :star6, :star7, :star8, :vline, :hline, :+, :x];
# linecolor = [:cornflowerblue, :red, :green1,:gray, :violet, :cyan ]


# loss = Vector{T}(undef,0)  
#   for k in 1:length(sol_hist[2])
#           push!(loss,max.(norm(A*sol_hist[1][k]-b)^2/2 + g(sol_hist[1][k])- f_star, eps(1e-20))/(f_0-f_star)  )  
# end
# plot!(plt, sol_hist[2]./N,loss, yaxis =:log, markersize = 2, markeralpha = 0.8)
# gui()
# savefig("SVRG.pdf")





# #### saving
# using DelimitedFiles

#   output = [sol_hist[2]./N loss]
#   d = length(loss)
#   rr = Int(ceil(d/50)) # keeping at most 50 data points
#   red_output =  output[1:rr:end,:]

#   # open(string("plot_data/","_N_",N,"_batch_",τ,"Richtaric_","lMlm",Int(floor(maximum(L)/minimum(L))),".txt"),"w") do io 
#    open(string("plot_data/",str,"SVRG/","_N_",N,"_batch_",1,"_","SVRG_","Lratio_",Int(floor(maximum(L)/minimum(L))),"_lambda_",Int(round(lam*10)),".txt"),"w") do io 
#           writedlm(io, red_output)
#     writedlm(io, red_output)
#   end


# open(
#   string("output.txt"),
#   "w",
# ) do io
#   writedlm(io, 88)
# end

