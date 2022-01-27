using Flux: train!

R = eltype(w0)
η_adam, β_adam = 0.001, (0.9,0.999)
opt = ADAM(η_adam, β_adam)
func = plot_extras.f

it_sum = 0
println("using ADAM ...")

res_hist = Vector{R}(undef, 0)
cost_hist = Vector{R}(undef, 0)
it_hist = Vector{R}(undef, 0)
# resz = eval_gradient(w)
resz = eval_gradient()
# initial point
push!(cost_hist, func(data))
# push!(cost_hist, func(z=w)[1])
push!(res_hist, resz)
push!(it_hist, 0)

println("init point res: ", resz); flush(stdout)

##### training using the built in functions --------
for epoch in 1:maxit
    for i in 1:N
        data_batch = [(data[1][:,i],data[2][:,i])]
        train!(ADAMLoss, ps, data_batch, opt)
    end
    global it_sum += N
    ww = DNN_config!()
    resz = eval_gradient()
    # resz = eval_gradient(ww)
    cost = func(data)
    # cost = func(z=ww)
    push!(res_hist, resz)
    push!(cost_hist, cost)  # cost
    push!(it_hist, it_sum/N)
    # println("epoch: $(it_sum/N) cost: $(cost) res: $(resz) - norm_0: $(sum(q -> norm(q,0), ps))")
    @printf("epoch: %i  cost: %.14f res: %.2e - norm_0: %i \n", (it_sum/N), cost, resz, sum(q -> norm(q,0), ps)); flush(stdout)
end
##### training using the built in functions --------

output = [it_hist cost_hist]
d = length(cost_hist)
rr = 1 #Int(ceil(d / 50)) # keeping at most 50 data points
red_output = output[1:rr:end, :] #reduced

mkpath(string("plot_data/",str,"/ADAM/cost"))
open(
    string(
        "plot_data/",
        str,
        "ADAM/cost/",
        "DNN_N_",
        N,
        "_n_",
        n,
        "_batch_",
        1, # for l
        "_",
        "ADAM",
        ".txt",
    ),
    "w",
) do io
    writedlm(io, red_output)
end

# residual |z- prox(z)|
output = [it_hist res_hist]
d = length(res_hist)
rr = Int(ceil(d / 50)) # keeping at most 50 data points
red_output = output[1:rr:end, :]

mkpath(string("plot_data/",str,"/ADAM/res"))
open(
    string(
        "plot_data/",
        str,
        "ADAM/res/",
        "DNN_N_",
        N,
        "_n_",
        n,
        "_batch_",
        1, # for l
        "_",
        "ADAM",
        ".txt",
    ),
    "w",
) do io
    writedlm(io, red_output)
end