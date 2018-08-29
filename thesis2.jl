if something(findfirst(isequal("./"),LOAD_PATH),0) == 0
    push!(LOAD_PATH, "./")
end
using Distributed
using Distributions
using ForwardDiff
using JuMP
using NLopt
using GR
using DataFrames
using MultivariateStats
using Flux
using DelimitedFiles
using Statistics
using Parameters
using Simulator

#Basis functions
ϕ(x) = [1, 1e-5*v(x),
        2.5e-9*(q(x)+h(x))*v(x),
        2.5e-3*s(x),
        1e-2*(s(x)-p(x)),
        #0.5e-3*(q(x)-h(x)),
        1e-5*q(x)*h(x),
        0.5e-3*(q(x)+h(x)),
        1e-7*(q(x)+h(x))^2,
        1e-7*(q(x)^2 + h(x)^2)]
ϕ(X::Array{Array{Float64,1},1}) = permutedims(Flux.batch(ϕ.(X)))

#Function approximations
V(θ,x) =  sum(θ .* ϕ(x))
δᵃ(θ,x) = sum(θ .* ϕ(x))
δᵇ(θ,x) = sum(θ .* ϕ(x))
ξ(θ,x) =  sum(θ .* ϕ(x))

function get_param(X)
    x = X[1:5]
    θ = X[6:length(X)]
    x,θ
end

function get_x(X)
    X[1:5]
end

#Value funcion as function of one master argument
#with both the state x and the parameters θᵛ
function V(X)
    x,θ = get_param(X)
    V(θ,x)
end

#Reward function partial derivatives
∇R(x) = ForwardDiff.gradient(R,x)
∇²R(x) = ForwardDiff.hessian(R,x)
∂Ry(x) = ∇R(x)[1]
∂Rs(x) = ∇R(x)[2]
∂Rp(x) = ∇R(x)[3]
∂Rq(x) = ∇R(x)[4]
∂Rh(x) = ∇R(x)[5]
∂∂Rss(x) = ∇²R(x)[2,2]
∂∂Rpp(x) = ∇²R(x)[3,3]
∂∂Rsp(x) = ∇²R(x)[2,3]

#Value function partial derivatives
∇V(θ,x) = ForwardDiff.gradient(V,[x;θ])
∇Vx(θ,x) = ∇V(θ,x)[1:5]
∇²V(θ,x) = ForwardDiff.hessian(V,[x;θ])
∇²Vx(θ,x) = ∇²V(θ,x)[1:5,1:5]
∂Vy(θ,x) = ∇V(θ,x)[1]
∂Vs(θ,x) = ∇V(θ,x)[2]
∂Vp(θ,x) = ∇V(θ,x)[3]
∂Vq(θ,x) = ∇V(θ,x)[4]
∂Vh(θ,x) = ∇V(θ,x)[5]
∂∂Vss(θ,x) = ∇²V(θ,x)[2,2]
∂∂Vpp(θ,x) = ∇²V(θ,x)[3,3]
∂∂Vsp(θ,x) = ∇²V(θ,x)[2,3]

#Value ingredients
Δᵃ(x,δᵃ,rᵃ) = [y(x) + (rᵃ>=δᵃ ? s(x)*(1+δᵃ) : 0), s(x) + s(x)*rᵃ, p(x) + s(x)*rᵃ, q(x) + (rᵃ>=δᵃ ? -1 : 0), h(x)]
Δᵇ(x,δᵇ,rᵇ) = [y(x) + (rᵇ>=δᵇ ? -s(x)*(1-δᵇ) : 0), s(x) - s(x)*rᵇ, p(x) - s(x)*rᵇ, q(x) + (rᵇ>=δᵇ ?  1 : 0), h(x)]
ΔᵃV(θ,x,δᵃ,rᵃ) = V(θ,Δᵃ(x,δᵃ,rᵃ)) - V(θ,x)
ΔᵇV(θ,x,δᵇ,rᵇ) = V(θ,Δᵇ(x,δᵇ,rᵇ)) - V(θ,x)
ΔᵃᵖV(θ,x,rᵖ) = cheap(x) ? V(θ,[y(x),s(x),p(x)+s(x)*rᵖ,q(x),h(x)]) - V(θ,x) : 0
ΔᵇᵖV(θ,x,rᵖ) = rich(x) ? V(θ,[y(x),s(x),p(x)-s(x)*rᵖ,q(x),h(x)]) - V(θ,x) : 0
EΔᵃV(θ,x,δᵃ) = mean([ΔᵃV(θ,x,δᵃ,r) for r in rᵃ(x)])
EΔᵇV(θ,x,δᵇ) = mean([ΔᵇV(θ,x,δᵇ,r) for r in rᵇ(x)])
EΔᵃᵖV(θ,x) = mean([ΔᵃᵖV(θ,x,r) for r in rᵃ(x)])
EΔᵇᵖV(θ,x) = mean([ΔᵇᵖV(θ,x,r) for r in rᵇ(x)])

#V(θᵛ,x₀), V(θᵛ,Δᵃ(x₀,0.0008,0.0008)), V(θᵛ,Δᵇ(x₀,0.0001,0.0001))

#Reward ingredients
ΔᵃR(x,δᵃ,rᵃ) = R(Δᵃ(x,δᵃ,rᵃ)) - R(x)
ΔᵇR(x,δᵇ,rᵇ) = R(Δᵇ(x,δᵇ,rᵇ)) - R(x)
ΔᵃᵖR(x,rᵖ) = cheap(x) ? R([y(x),s(x),p(x)+s(x)*rᵖ,q(x),h(x)]) - R(x) : 0
ΔᵇᵖR(x,rᵖ) = rich(x) ? R([y(x),s(x),p(x)-s(x)*rᵖ,q(x),h(x)]) - R(x) : 0
EΔᵃR(x,δᵃ) = mean([ΔᵃR(x,δᵃ,r) for r in rᵃ(x)])
EΔᵇR(x,δᵇ) = mean([ΔᵇR(x,δᵇ,r) for r in rᵇ(x)])
EΔᵃᵖR(x) = mean([ΔᵃᵖR(x,r) for r in rᵃ(x)])
EΔᵇᵖR(x) = mean([ΔᵇᵖR(x,r) for r in rᵇ(x)])

#Infinitesimal operator for V
function AV(θ,x,δᵃ,δᵇ)
    θ*(𝔭*s(x)-p(x))*∂Vp(θ,x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Vss(θ,x) + κ^2 * ∂∂Vpp(θ,x) + 2 * ϱ * ς * κ * ∂∂Vsp(θ,x))
    + λᵃ^-1 * EΔᵃV(θ,x,δᵃ) + λᵇ^-1 * EΔᵇV(θ,x,δᵇ) + λᵖ^-1 * EΔᵃᵖV(θ,x) + λᵖ^-1 * EΔᵇᵖV(θ,x)
end

#Infinitesimal operator for R
function AR(x,δᵃ,δᵇ)
    Parameters.θ*(𝔭*s(x)-p(x))*∂Rp(x)
    + 0.5 * s(x)^2 *(ς^2 * ∂∂Rss(x) + κ^2 * ∂∂Rpp(x) + 2 * ϱ * ς * κ * ∂∂Rsp(x))
    + λᵃ^-1 * EΔᵃR(x,δᵃ) + λᵇ^-1 * EΔᵇR(x,δᵇ) + λᵖ^-1 * EΔᵃᵖR(x) + λᵖ^-1 * EΔᵇᵖR(x)
end

#L operator as described in the thesis
function LV(θ,x,δᵃ,δᵇ)
    AR(x,δᵃ,δᵇ) + AV(θ,x,δᵃ,δᵇ) - ρ*V(θ,x)
end

#Intervention function as described in the thesis
function Γ(x,ξ)
    x .+ [-(s(x)-p(x))*ξ - abs(ξ)*s(x)*χ, 0, 0, 0, ξ]
end

#Intervention operator
function MV(θ,x,ξ)
    V(θ,Γ(x,ξ))-V(θ,x)
end

#Establishes valid values for ξ depending on the position
function validξ(q,h)
    if q+h> hedge_limit #long
        max(-q-h,-5):0
    elseif q+h < -hedge_limit
        0:min(-q-h,5)
    else
        0:0
    end
end

#Greedy controls
maxδᵃ(θ,x) = 0.0001*argmax([EΔᵃV(θ,x,δ)+EΔᵃR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxδᵇ(θ,x) = 0.0001*argmax([EΔᵇV(θ,x,δ)+EΔᵇR(x,δ) for δ in 0.0001:0.0001:0.0020])
maxξ(θ,x)  = validξ(q(x),h(x))[argmax([V(θ,Γ(x,ξ)) for ξ in validξ(q(x),h(x))])]

function total_reward(x₀;θᵛ=θᵛ,N=1000,T=1,Simulations=500,Safe=true)
    Δt = 1/N
    reward = 0
    reward = @distributed (+) for i = 1:Simulations
        reward_i = 0
        sim = simulate(x₀;N=N,T=T)
        t=1
        x = x₀
        for j in sim
            bid = Safe ? min(0.01,max(0.0001,maxδᵇ(θᵛ,x))) : maxδᵇ(θᵛ,x)
            offer = Safe ? min(0.01,max(0.0001,maxδᵃ(θᵛ,x))) : maxδᵃ(θᵛ,x)
            hedge = maxξ(θᵛ,x)
            if isnan(hedge)
                hedge=0
            end
            if isinf(hedge)
                hedge=0
            end
            new_x = [y(x), j[1], j[2], q(x), h(x)]
            new_rᵃ = j[3]
            new_rᵇ = j[4]
            if new_rᵃ >= offer
                new_x = new_x + [s(new_x),0,0,-1,0]
            end
            if new_rᵇ >= bid
                new_x = new_x + [-s(new_x),0,0,1,0]
            end
            if offer < 0 || bid < 0
                new_x = new_x + [- s(new_x)*χ,0,0,0,0]
            end
            posicao = q(x)+h(x)
            if posicao > hedge_limit
                hedge = max(-posicao,min(0,hedge))
            elseif posicao < -hedge_limit
                hedge = min(-posicao,max(0,hedge))
            end

            if abs(hedge) >= 1
                new_x = Γ(new_x,hedge)
            end
            reward_i += exp(-ρ*t*Δt)*R(new_x) - exp(-ρ*(t-1)*Δt)*R(x)
            if isnan(reward_i)
                println("NAN at ",x," -> ",new_x," hedge: ",hedge,", t: ",t)
                break
            end
            x = new_x
            t+=1
        end
        #reward += reward_i
        reward_i + exp(-ρ*t)*R(x)
    end
    reward/N
end


#Initial values for the function approximations parameters
θᵛ = rand(length(ϕ(ones(5))))
θᵃ = zeros(length(ϕ(ones(5))))
θᵇ = zeros(length(ϕ(ones(5))))
θʷ = zeros(length(ϕ(ones(5))))

V_estimate(θ,x) = max(LV(θ,x,maxδᵃ(θ,x),maxδᵇ(θ,x)),MV(θ,x,maxξ(θ,x))) + v(x) #V(θ,x)
#V_estimate(θ,x) = V(θ,x) + max(max([LV(θ,x,δᵃ,δᵇ) for δᵃ in 0.0001:0.0002:0.0016 for δᵇ in 0.0001:0.0002:0.0016]...), max([MV(θ,x,ξ) for ξ in validξ(q(x),h(x))]...))

 #max(LV(θ,x,maxδᵃ(θ,x),maxδᵇ(θ,x)),MV(θ,x,maxξ(θ,x)))
V_estimate(θ) = vcat([V_estimate(θ,x) for x in batch]...)
V_bar(θ) = vcat([V(θ,x) for x in batch]...)
maxδᵃ(θ) = mean([maxδᵃ(θ,x) for x in batch])
maxδᵇ(θ) = mean([maxδᵇ(θ,x) for x in batch])
maxξ(θ) = mean([maxξ(θ,x) for x in batch])

#max( max([LV(θ,x,δᵃ,δᵇ) for δᵃ in 0.0001:0.0001:0.0020 for δᵇ in 0.0001:0.0001:0.0020]...), max([MV(θ,x,ξ) for ξ in validξ(q(x),h(x))]...))

function second_order_condition(grad1,grad2)
    [grad1[i]*grad2[i] + grad1[j]*grad2[j] - abs(grad1[i]*grad2[j]+grad1[j]*grad2[i]) for i in 1:5 for j in i:5 if i!=j]
end

function third_order_condition(grad1, grad2)
    [grad1[i]*grad2[i] + grad1[j]*grad2[j] + grad1[k]*grad2[k] - abs(grad1[i]*grad2[j]+grad1[j]*grad2[i]) - abs(grad1[i]*grad2[k]+grad1[k]*grad2[i]) - abs(grad1[k]*grad2[j]+grad1[j]*grad2[k]) for i in 1:5 for j in i:5 for k in 1:5 if i<j && j<k]
end

function fourth_order_condition(grad1, grad2)
    [grad1[i]*grad2[i] + grad1[j]*grad2[j] + grad1[k]*grad2[k] + grad1[h]*grad2[h] - abs(grad1[i]*grad2[j]+grad1[j]*grad2[i]) - abs(grad1[i]*grad2[k]+grad1[k]*grad2[i]) - abs(grad1[k]*grad2[j]+grad1[j]*grad2[k]) - abs(grad1[h]*grad2[i]+grad1[i]*grad2[h]) - abs(grad1[h]*grad2[j]+grad1[j]*grad2[h]) - abs(grad1[h]*grad2[k]+grad1[k]*grad2[h]) for i in 1:5 for j in i:5 for k in 1:5 for h in 1:5 if i<j && j<k && k<h]
end

function fifth_order_condition(grad1, grad2)
    [
    grad1[1]*grad2[1] +
    grad1[2]*grad2[2] +
    grad1[3]*grad2[3] +
    grad1[4]*grad2[4] +
    grad1[5]*grad2[5]
    - abs(grad1[1]*grad2[2]+grad1[2]*grad2[1])
    - abs(grad1[1]*grad2[3]+grad1[3]*grad2[1])
    - abs(grad1[3]*grad2[2]+grad1[2]*grad2[3])
    - abs(grad1[4]*grad2[1]+grad1[1]*grad2[4])
    - abs(grad1[4]*grad2[2]+grad1[2]*grad2[4])
    - abs(grad1[4]*grad2[3]+grad1[3]*grad2[4])
    - abs(grad1[5]*grad2[1]+grad1[1]*grad2[5])
    - abs(grad1[5]*grad2[2]+grad1[2]*grad2[5])
    - abs(grad1[5]*grad2[3]+grad1[3]*grad2[5])
    - abs(grad1[5]*grad2[4]+grad1[4]*grad2[5])
    ]
end

function V_constraint1(θ)
    vcat([∇Vx(θ,x) .* ∇R(x) for x in batch]...)
end

function V_constraint2(θ)
    vcat([second_order_condition(∇Vx(θ,x),∇R(x)) for x in batch]...)
end

function V_constraint3(θ)
    vcat([third_order_condition(∇Vx(θ,x),∇R(x)) for x in batch]...)
end

function V_constraint4(θ)
    vcat([fourth_order_condition(∇Vx(θ,x),∇R(x)) for x in batch]...)
end

function V_constraint5(θ)
    vcat([fifth_order_condition(∇Vx(θ,x),∇R(x)) for x in batch]...)
end

function constraint(θ)
    [V_constraint1(θ);V_constraint2(θ);V_constraint3(θ);V_constraint4(θ);V_constraint5(θ)]
end

function objective_breakdown(z)
    θ = z[1:length(θᵛ)]
    ϵ = z[length(θ)+1:length(θ)+41*length(batch)]
    λ = z[length(θ)+1+41*length(batch):length(θ)+82*length(batch)]
    η = z[length(θ)+82*length(batch)+1]
    sum((V_estimate(θ)-V_bar(θ)).^2) , sum(abs.(θ)), sum(θ.^2), - λ'*(constraint(θ) - ϵ.^2) , 0.5 * η * sum((constraint(θ) - ϵ.^2) .^2)
end
#Lagrangian objective
function objective(z)
    θ = z[1:length(θᵛ)]
    ϵ = z[length(θ)+1:length(θ)+41*length(batch)]
    λ = z[length(θ)+1+41*length(batch):length(θ)+82*length(batch)]
    η = z[length(θ)+82*length(batch)+1]
    sum((V_estimate(θ)-V_bar(θ)).^2) + 0.5sum(abs.(θ)) + 0.5sum(θ.^2) - λ'*(constraint(θ) - ϵ.^2) + 0.5 * η * sum((constraint(θ) - ϵ.^2) .^2)
end
objective(θ,ϵ,λ,η) = objective([θ;ϵ;λ;η])
∇objective(θ,ϵ,λ,η) = ForwardDiff.gradient(objective,[θ;ϵ;λ;η])[1:length(θ)+41*length(batch)]

function minimize_lagrangian(θ₀,λ;gradient_step = 10^-10, penalty_parameter = 1, n_iter = 40, debug = false)
    F = zeros(n_iter)
    θ = [θ₀ for i in 1:n_iter]
    ϵ = [zeros(length(λ)) for i in 1:n_iter]
    improvement = [0.0 for i in 1:n_iter]
    restarted = false
    for iter = 1:n_iter
        F[iter] = objective(θ[iter],ϵ[iter],λ,penalty_parameter)
        if iter>1
            if F[iter]>F[iter-1]
                gradient_step /= 4
                restarted = true
                #restart the search
                println("> F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1]), gradient_step was $gradient_step")
                println("> Restarting search")
                F[iter] = F[iter-1]
                θ[iter] = θ[iter-1]
                #return F[iter-1],θ[iter-1],ϵ[iter-1],gradient_step
            elseif gradient_step < 10^-5 && !restarted
                gradient_step *= 2 #max(1,log(abs(F[iter]))/2)
            end
        end
        grad = ∇objective(θ[iter],ϵ[iter],λ,penalty_parameter)
        if iter<n_iter
            θ[iter+1] = θ[iter] - gradient_step * grad[1:length(θ₀)]
            ϵ[iter+1] = ϵ[iter] - gradient_step * grad[length(θ₀)+1:end]
        elseif iter==n_iter
            return F[iter],θ[iter],ϵ[iter],gradient_step
        end
        improvement[iter] = iter>1 ? max((F[iter-1]-F[iter])/abs(F[iter-1]), 0.0) : 0.0
        if improvement[iter] < 0.01 && !restarted
            gradient_step *= 10
        end
        if restarted && improvement[iter] < 2e-4 && improvement[iter] < improvement[iter-1]
            return F[iter],θ[iter],ϵ[iter],gradient_step
        end
        if debug
            println("> Objective F(θ[$iter])): $(F[iter])", improvement[iter] > 0 ? ". Improvement: $(@sprintf("%.4f",100*improvement[iter]))%" : ". No improvemnt with gradient step $gradient_step.")
        end
    end
end

function solve_problem(θ₀;
    n_outer_iter = 20, n_inner_iter = 20, gradient_step = 10^-10, adjoint_step=10^-5,
    penalty_parameter=1, current_epoch = 1, debug=false)
    #Lagrange multipliers
    λ = ones(41*length(batch))
    F = zeros(n_outer_iter)
    θ = [θ₀ for i in 1:n_outer_iter]
    ϵ = [ones(length(λ)) for i in 1:n_outer_iter]
    η = penalty_parameter
    for iter = 1:n_outer_iter
        println("Iteration $current_epoch.$iter:")
        F[iter], θ[iter], ϵ[iter], gradient_step = minimize_lagrangian(θ[iter],λ;gradient_step=gradient_step,penalty_parameter=η,n_iter=n_inner_iter,debug=debug)
        λ += adjoint_step * (constraint(θ[iter]) - ϵ[iter].^2)
        λ = [max(k,0) for k in λ]
        println("Best F[$current_epoch.$iter] = $(F[iter])")
        if iter>1
            if F[iter] > F[iter-1]
                println("F[$current_epoch.$iter] is worse than F[$current_epoch.$(iter-1)]: $(F[iter]) > $(F[iter-1])")
                return θ[iter-1],gradient_step
            elseif iter==n_outer_iter
                return θ[iter],gradient_step
            elseif adjoint_step < 0.05
                adjoint_step *= 2
            end
        end
        θ[iter+1] = θ[iter]
    end
end

#The learning rates below indicate how much % we want to learn from the current best
#and how much we want to learn from the new optimized function, given the new data
learning_rate = 1.0

master_batch = randx(100)
master_batch_matrix = permutedims(Flux.batch(ϕ.(master_batch)))
master_mean = [abs(mean(master_batch_matrix[:,i])) for i in 1:length(θᵛ)]

batch = sample(master_batch,4,replace=false)
ϕ_X = ϕ(batch)
θᵛ = rand(length(ϕ(ones(5)))) - 0.5
penalty_parameter = 1 #2e-7
gradient_step = 10^-6
θᵛhistory = [θᵛ]

q_coords = vcat([q for q in -20:1:20, h in -20:1:20]...)
h_coords = vcat([h for q in -20:1:20, h in -20:1:20]...)
bids = vcat([maxδᵇ(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
offers = vcat([maxδᵃ(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
v_values = vcat([V(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
surface(q_coords,h_coords,bids, title="Bid - Start")
surface(q_coords,h_coords,offers, title="Offer - Start")
surface(q_coords,h_coords,v_values, title="Value - Start")

for iter = 1:20
    println("Epoch: $iter")
    θᵛ,gradient_step = solve_problem(θᵛ,
        gradient_step = gradient_step,
        adjoint_step = 0.001,
        penalty_parameter = iter <= 10 ? penalty_parameter : 0.5*penalty_parameter,
        n_outer_iter = iter==1 ? 10 : 5,
        n_inner_iter = iter==1 ? 20 : 20,
        current_epoch = iter,
        debug = true)
    current_learning_rate = iter == 1 ? 1 : learning_rate*(1 - sqrt(iter)/4.9)
    global θᵛ = last(θᵛhistory)*(1-current_learning_rate) + θᵛ * current_learning_rate
    global θᵛhistory = [θᵛhistory..., θᵛ]
    gradient_step /= 2
    println("V(x₀) = ",V(θᵛ,x₀))
    println("θ[$iter] = $θᵛ")
    bids = vcat([maxδᵇ(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
    offers = vcat([maxδᵃ(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
    v_values = vcat([V(θᵛ,x₀+[-100*(q+h),0,0,q,h]) for q in -20:1:20, h in -20:1:20]...)
    surface(q_coords,h_coords,bids, title="Bid - $iter")
    surface(q_coords,h_coords,offers, title="Offer - $iter")
    surface(q_coords,h_coords,v_values, title="Value - $iter")
    #Replace half of the samples
    batch = sample(master_batch,4,replace=false)
    ϕ_X = ϕ(batch)
    #penalty_parameter *= 1/log(1+sum(θᵛ.^2))
end

function generate_scenario(θᵛ,f::Function, name::String, should_plot)
    scenario = DataFrame()
    scenario[:q] = vcat([y for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:h] = vcat([x for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:bid] = vcat([maxδᵇ(θᵛ,f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:offer] = vcat([maxδᵃ(θᵛ,f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:hedge] = vcat([maxξ(θᵛ,f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:wealth] = vcat([v(f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:utility] = vcat([R(f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    scenario[:value] = vcat([V(θᵛ,f(y,x)) for x in -2000:100:2000, y in -2000:100:2000]...)
    if should_plot
        surface(scenario[:q],scenario[:h],scenario[:bid], title="Bid - $name")
        surface(scenario[:q],scenario[:h],scenario[:offer],  title="Offer - $name")
        surface(scenario[:q],scenario[:h],scenario[:hedge],  title="Hedge - $name")
        surface(scenario[:q],scenario[:h],scenario[:wealth], title="Wealth - $name")
        surface(scenario[:q],scenario[:h],scenario[:utility], title="Utility - $name")
        surface(scenario[:q],scenario[:h],scenario[:value], title="Value - $name")
    end
    return scenario
end

function generate_scenario2(θᵛ,f::Function, name::String, should_plot)
    scenario = DataFrame()
    scenario[:q] = vcat([y for x in -50:2:50, y in -50:2:50]...)
    scenario[:h] = vcat([x for x in -50:2:50, y in -50:2:50]...)
    scenario[:bid] = vcat([maxδᵇ(θᵛ,f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    scenario[:offer] = vcat([maxδᵃ(θᵛ,f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    #scenario[:hedge] = vcat([maxξ(θᵛ,f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    #scenario[:wealth] = vcat([v(f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    #scenario[:utility] = vcat([R(f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    scenario[:value] = vcat([V(θᵛ,f(y,x)) for x in -50:2:50, y in -50:2:50]...)
    if should_plot
        surface(scenario[:q],scenario[:h],scenario[:bid], title="Bid - $name")
        surface(scenario[:q],scenario[:h],scenario[:offer],  title="Offer - $name")
        #surface(scenario[:q],scenario[:h],scenario[:hedge],  title="Hedge - $name")
        #surface(scenario[:q],scenario[:h],scenario[:wealth], title="Wealth - $name")
        #surface(scenario[:q],scenario[:h],scenario[:utility], title="Utility - $name")
        surface(scenario[:q],scenario[:h],scenario[:value], title="Value - $name")
    end
    return scenario
end

#generate_scenario2([-6459.41, 4817.17, 3239.14, 2914.76, 3803.45, -7771.34*0, 1182.54, 5644.38, -2739.32, -2762.34],(q,h) -> x₀+[-100*(q+h),0,0,q,h], "Scenario 1 (*)", true)

#Same wealth scenario, different arrangements of position
generate_scenario(θᵛ,(q,h) -> x₀+[-100*(q+h),0,0,q,h], "Scenario 1", true)
#Zero cash scenarion, different arrangements of position
generate_scenario(θᵛ,(q,h) -> x₀+[0,0,0,q,h], "Scenario 2", true)



#random_direction = vcat(randx(1)...)
# random_x = sample(batch)
# sign.(∇R(random_x) .* ∇V(θᵛ,random_x)[1:5])
# sign.(∇R(random_x) .* ∇V([0,1,0.00000001,0,0,0,0,0,0,-0.0001],random_x)[1:5])



#plot(-2000:100:2000,[R(x₀+i*random_direction) for i in -2000:100:2000], title="R")
#plot(-2000:100:2000,[V(θᵛ,x₀+i*random_direction) for i in -2000:100:2000], title="V")
