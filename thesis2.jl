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
using Printf
using Parameters
using Simulator

#Basis functions
ϕ(x) = [1,
        v(x),
        s(x),
        p(x),
        q(x)+h(x),
        1e-4*(q(x)+h(x))^2,
        1e-4*(q(x)^2 + h(x)^2)]#6.25e-8
        #1e-16*v(x)^3,
        #1e-10*(q(x)+h(x))^3,

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
∇Vθ(θ,x) = ∇V(θ,x)[6:end]
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

#[(δ,round(EΔᵃV(last(θ),x₀,δ),digits=2),EΔᵃR(x₀,δ)) for δ in 0.0001:0.0001:0.0020]
#[(δ,round(EΔᵇV(last(θ),x₀,δ),digits=2),EΔᵇR(x₀,δ)) for δ in 0.0001:0.0001:0.0020]
#[(ξ,round(V(θ1,Γ(x₀,ξ)),digits=2)) for ξ in validξ(-2000000,-2000000)]

#maxξ(θ1,x₀+[0,0,0,20000,20000])

function total_reward(x₀;θᵛ=θᵛ,N=100,T=1,Simulations=10,Safe=true)
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
#θᵛ = rand(length(ϕ(ones(5))))
θᵃ = zeros(length(ϕ(ones(5))))
θᵇ = zeros(length(ϕ(ones(5))))
θʷ = zeros(length(ϕ(ones(5))))

hjbqvi(θ,x) = max(LV(θ,x,maxδᵃ(θ,x),maxδᵇ(θ,x)),MV(θ,x,maxξ(θ,x)))
hjbqvi(θ) = vcat([hjbqvi(θ,x) for x in batch]...)

V_estimate(θ,x) = hjbqvi(θ,x) + V(θ,x)
V_estimate(θ) = vcat([V_estimate(θ,x) for x in batch]...)

V_bar(θ) = vcat([V(θ,x) for x in batch]...)
maxδᵃ(θ) = mean([maxδᵃ(θ,x) for x in batch])
maxδᵇ(θ) = mean([maxδᵇ(θ,x) for x in batch])
maxξ(θ) = mean([maxξ(θ,x) for x in batch])

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

function constraint_eigen(θ,x)
  v = ∇Vx(θ,x)
  r = ∇R(x)
  A=[v[1]*r[1] v[1]*r[2] v[1]*r[3] v[1]*r[4] v[1]*r[5];
  v[2]*r[1] v[2]*r[2] v[2]*r[3] v[2]*r[4] v[2]*r[5];
  v[3]*r[1] v[3]*r[2] v[3]*r[3] v[3]*r[4] v[3]*r[5];
  v[4]*r[1] v[4]*r[2] v[4]*r[3] v[4]*r[4] v[4]*r[5];
  v[5]*r[1] v[5]*r[2] v[5]*r[3] v[5]*r[4] v[5]*r[5]]
  eigen(0.5*(A+A')).values
end

function constraint_eigen(z)
    θ = z[1:length(θᵛ)]
    x = z[length(θ)+1:end]
    sum(constraint_eigen(θ,x))
end

∇constrain_eigen(θ,x) = ForwardDiff.gradient(constraint_eigen,[θ;x])

# function constraint(θ)
#     vcat([constraint_eigen(θ,x) for x in batch]...)
# end

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

regularization_parameter = 1e5

function objective_breakdown(z)
    θ = z[1:length(θᵛ)]
    ϵ = z[length(θ)+1:length(θ)+41*length(batch)]
    λ = z[length(θ)+1+41*length(batch):length(θ)+82*length(batch)]
    η = z[length(θ)+82*length(batch)+1]
    sum((V_estimate(θ)-V_bar(θ)).^2), regularization_parameter*sum(θ.^2) , - λ'*(constraint(θ) - ϵ.^2) , 0.5 * η * sum((constraint(θ) - ϵ.^2) .^2)
end

#+ regularization_parameter*sum(θ.^2)
#Lagrangian objective
function objective(z)
    θ = z[1:length(θᵛ)]
    ϵ = z[length(θ)+1:length(θ)+41*length(batch)]
    λ = z[length(θ)+1+41*length(batch):length(θ)+82*length(batch)]
    η = z[length(θ)+82*length(batch)+1]
    sum((V_estimate(θ)-V_bar(θ)).^2)  - λ'*(constraint(θ) - ϵ.^2) + 0.5 * η * sum((constraint(θ) - ϵ.^2) .^2) #0.5sum(abs.(θ))
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
                if debug
                    println(">> F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1]), gradient_step was $gradient_step")
                    println(">> Restarting search")
                end
                F[iter] = F[iter-1]
                θ[iter] = θ[iter-1]
                #return F[iter-1],θ[iter-1],ϵ[iter-1],gradient_step
            elseif gradient_step < 10^-6 && !restarted
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
        if improvement[iter] < 0.01 && !restarted && gradient_step < 0.1
            gradient_step *= 10
        end
        if restarted && improvement[iter] < 2e-4 && improvement[iter] < improvement[iter-1]
            return F[iter],θ[iter],ϵ[iter],gradient_step
        end
        if debug
            println(">> Objective F(θ[$iter])): $(F[iter])", improvement[iter] > 0 ? ". Improvement: $(@sprintf("%.4f",100*improvement[iter]))%" : ". No improvemnt with gradient step $gradient_step.")
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
        if debug println("> Iteration $current_epoch.$iter:") end
        F[iter], θ[iter], ϵ[iter], gradient_step = minimize_lagrangian(θ[iter],λ;gradient_step=gradient_step,penalty_parameter=η,n_iter=n_inner_iter,debug=debug)
        λ += adjoint_step * (constraint(θ[iter]) - ϵ[iter].^2)
        λ = [max(k,0) for k in λ]
        if debug println("> Best F[$current_epoch.$iter] = $(F[iter])") end
        if iter>1
            if F[iter] > F[iter-1]
                if debug println("> F[$current_epoch.$iter] is worse than F[$current_epoch.$(iter-1)]: $(F[iter]) > $(F[iter-1])") end
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

function td0_episode(θᵛ,x₀;N=1000,T=1,learning_rate=0.05,safe=true)
    Δt = 1/N
    reward = 0
    path = simulate(x₀;N=N,T=T)
    x = x₀
    t = 1
    ΔP = zeros(length(θᵛ))
    Δθ = zeros(length(θᵛ))
    for (sₜ,pₜ,rᵃ,rᵇ) in path
        bid = safe ? min(0.01,max(0.0001,maxδᵇ(θᵛ,x))) : maxδᵇ(θᵛ,x)
        offer = safe ? min(0.01,max(0.0001,maxδᵃ(θᵛ,x))) : maxδᵃ(θᵛ,x)
        hedge = maxξ(θᵛ,x)
        if isnan(hedge) || isinf(hedge)
            hedge=0
        end
        xₜ = [y(x), sₜ, pₜ, q(x), h(x)] .+ (rᵃ >= offer ? [sₜ,0,0,-1,0] : [0,0,0,0,0]) .+ (rᵇ >= bid ? [-sₜ,0,0,1,0] : [0,0,0,0,0])
        pos = q(x)+h(x)
        if pos > hedge_limit
            hedge = max(-pos,min(0,hedge))
        elseif pos < -hedge_limit
            hedge = min(-pos,max(0,hedge))
        end
        if abs(hedge) >= 1
            xₜ = Γ(xₜ,hedge)
        end
        rewardₜ = exp(-ρ*t*Δt)*R(xₜ) - exp(-ρ*(t-1)*Δt)*R(x)
        reward += rewardₜ
        ΔP = ϕ(x) #∇Vθ(θᵛ,x)
        P = V(θᵛ,x)
        Pₜ = t==length(path) ? reward + exp(-ρ*t*Δt)*R(xₜ) : V(θᵛ,xₜ)
        Δθ += learning_rate * (Pₜ - P) * ΔP
        x = xₜ
        if sum(isnan.(Δθ))> 0
            println("Found NaN. learning_rate = $learning_rate, x = $x, xₜ = $xₜ, rewardₜ = $rewardₜ, θᵛ = $θᵛ")
            break
        end
        t+=1
    end
    #println("Last x = $x, R(x) = $(R(x)), reward = $reward")
    θᵛ+Δθ, reward
end


function td0_episode_online(θᵛ,x₀;N=1000,T=1,learning_rate=0.05,safe=true)
    Δt = 1/N
    reward = 0
    path = simulate(x₀;N=N,T=T)
    x = x₀
    t = 1
    ΔP = zeros(length(θᵛ))
    Δθ = zeros(length(θᵛ))
    for (sₜ,pₜ,rᵃ,rᵇ) in path
        bid = safe ? min(0.01,max(0.0001,maxδᵇ(θᵛ,x))) : maxδᵇ(θᵛ,x)
        offer = safe ? min(0.01,max(0.0001,maxδᵃ(θᵛ,x))) : maxδᵃ(θᵛ,x)
        hedge = maxξ(θᵛ,x)
        if isnan(hedge) || isinf(hedge)
            hedge=0
        end
        xₜ = [y(x), sₜ, pₜ, q(x), h(x)] .+ (rᵃ >= offer ? [sₜ,0,0,-1,0] : [0,0,0,0,0]) .+ (rᵇ >= bid ? [-sₜ,0,0,1,0] : [0,0,0,0,0])
        pos = q(x)+h(x)
        if pos > hedge_limit
            hedge = max(-pos,min(0,hedge))
        elseif pos < -hedge_limit
            hedge = min(-pos,max(0,hedge))
        end
        if abs(hedge) >= 1
            xₜ = Γ(xₜ,hedge)
        end
        rewardₜ = exp(-ρ*t*Δt)*R(xₜ) - exp(-ρ*(t-1)*Δt)*R(x)
        reward += rewardₜ
        ΔP = ϕ(x) #∇Vθ(θᵛ,x)
        P = V(θᵛ,x)
        Pₜ = t==length(path) ? reward + exp(-ρ*t*Δt)*R(xₜ) : V(θᵛ,xₜ)
        θᵛ += learning_rate * (Pₜ - P) * ΔP
        x = xₜ
        if sum(isnan.(Δθ))> 0
            println("Found NaN. learning_rate = $learning_rate, x = $x, xₜ = $xₜ, rewardₜ = $rewardₜ, θᵛ = $θᵛ")
            break
        end
        t+=1
    end
    #println("Last x = $x, R(x) = $(R(x)), reward = $reward")
    θᵛ, reward
end

function td1_episode(θᵛ,x₀;N=1000,T=1,learning_rate=0.05,safe=true)
    Δt = 1/N
    reward = 0
    path = simulate(x₀;N=N,T=T)
    x = x₀
    t = 1
    ΔP = zeros(length(θᵛ))
    Δθ = zeros(length(θᵛ))
    for (sₜ,pₜ,rᵃ,rᵇ) in path
        bid = safe ? min(0.01,max(0.0001,maxδᵇ(θᵛ,x))) : maxδᵇ(θᵛ,x)
        offer = safe ? min(0.01,max(0.0001,maxδᵃ(θᵛ,x))) : maxδᵃ(θᵛ,x)
        hedge = maxξ(θᵛ,x)
        if isnan(hedge) || isinf(hedge)
            hedge=0
        end
        xₜ = [y(x), sₜ, pₜ, q(x), h(x)] .+ (rᵃ >= offer ? [sₜ,0,0,-1,0] : [0,0,0,0,0]) .+ (rᵇ >= bid ? [-sₜ,0,0,1,0] : [0,0,0,0,0])
        pos = q(x)+h(x)
        if pos > hedge_limit
            hedge = max(-pos,min(0,hedge))
        elseif pos < -hedge_limit
            hedge = min(-pos,max(0,hedge))
        end
        if abs(hedge) >= 1
            xₜ = Γ(xₜ,hedge)
        end
        rewardₜ = exp(-ρ*t*Δt)*R(xₜ) - exp(-ρ*(t-1)*Δt)*R(x)
        reward += rewardₜ
        #rewardₜ = exp(-ρ*Δt)*R(xₜ) - R(x)
        #θᵛ += learning_rate * (rewardₜ + exp(-ρ*Δt)*V(θᵛ,xₜ) - V(θᵛ,x) ) * ∇Vθ(θᵛ,x)
        #rewardₜ +
        #td_error = reward + exp(-ρ*t*Δt)*V(θᵛ,xₜ) - V(θᵛ,x₀)
        # hjbqvi_error = hjbqvi(θᵛ,xₜ)
        # if !isnan(hjbqvi_error) && !isinf(hjbqvi_error) && hjbqvi_error!=0
        #     td_error = hjbqvi_error
        # end
        ΔP += ϕ(x) # ∇Vθ(θᵛ,x)
        P = V(θᵛ,x)
        Pₜ = t==length(path) ? reward + exp(-ρ*t*Δt)*R(xₜ) : V(θᵛ,xₜ)
        Δθ += learning_rate * (Pₜ - P) * ΔP
        x = xₜ
        #θᵛ += learning_rate * td_error * ∇Vθ(θᵛ,x)
        #Δθᵛ = learning_rate * td_error * ∇Vθ(θᵛ,x)
        if sum(isnan.(Δθ))> 0
            println("Found NaN. learning_rate = $learning_rate, x = $x, xₜ = $xₜ, rewardₜ = $rewardₜ, θᵛ = $θᵛ")
            break
        # else
        #     θᵛ += Δθᵛ
        end
        t+=1
    end
    #println("Last x = $x, R(x) = $(R(x)), reward = $reward")
    θᵛ+Δθ, reward
end

grid_space = [[x₀];[[0,100,0,2000,2000]];[[0,100,0,2000,-2000]];[[0,100,0,-2000,2000]];[[0,100,0,-2000,-2000]]]
shuffle!(grid_space)

batch = [x+[cash,price,premium,0,0] for x in grid_space for cash in [-10^5, 0, 10^5] for premium in [-1, -0.5, -0.2, 0, 0.2, 0.5, 1] for price in [-1, 0, 1]]
shuffle!(batch)

#θ1 = [-1.0001, 0.829139, 0.520647, 0.0970372, 0.521014, -0.134654, 0.387083, 0.115137, 0.0475305, 0.466884]
θ0 = [0., 1., 0., 0., 0., 0., 0.] #last(θ) #rand(7)-0.5 #[0.113619, 0.00993826, 0.0216627, 0.056741, 0.0100034, 0.00999995, 0.01] #[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
θ = [θ0 for i in 1:1000]
learning_rate = [i for i in range(0.5,stop=0.25,length=length(θ))]
gross_reward = 0
for iter in 1:length(θ)-1
    xₜ=vcat(sample(batch,1)...)
    θ[iter+1], rewardₜ = td0_episode(θ[iter],xₜ,learning_rate=learning_rate[iter]*1e-11)
    rewardₜ = round(rewardₜ,digits=4)
    gross_reward += rewardₜ
    error₀ = V(θ[iter+1],x₀) - V(θ[iter],x₀)
    errorₜ = V(θ[iter+1],xₜ) - V(θ[iter],xₜ)
    V₀ = round(V(θ[iter+1],x₀),digits=4)
    Vₜ = round(V(θ[iter+1],xₜ),digits=4)
    Rₜ = R(xₜ)
    error₀ = round(error₀,digits=4)
    errorₜ = round(errorₜ,digits=4)
    error =round(abs(error₀) + abs(errorₜ),digits=4)
    println("[$(iter+1)] V₀ = $V₀, Error = ($error), Reward = $rewardₜ, Reward avg = $(gross_reward/iter), θ = $(θ[iter+1])")
end

# #The learning rates below indicate how much % we want to learn from the current best
# #and how much we want to learn from the new optimized function, given the new data
# learning_rate = 1.0
#
# grid_space = [[x₀+[1000,0,0,0,0]];[[0,100,0,2000,2000]];[[0,100,0,2000,-2000]];[[0,100,0,-2000,2000]];[[0,100,0,-2000,-2000]]]
#
# master_batch = grid_space #[x+[cash,price,premium,0,0] for x in grid_space for cash in [-10^5, 0, 10^5] for premium in [-1, -0.5, -0.2, 0, 0.2, 0.5, 1] for price in [-1, 0, 1]]
# #shuffle!(master_batch)
# master_batch_matrix = permutedims(Flux.batch(ϕ.(master_batch)))
# master_mean = [abs(mean(master_batch_matrix[:,i])) for i in 1:length(θᵛ)]
#
# batch_iter = 1
# batch = master_batch[1:1] #sample(master_batch,1) #[sample(master_batch,3,replace=false);[x₀]]
# V_batch =[V(θᵛ,x) for x in batch]
# ϕ_X = ϕ(batch)
# θᵛ = rand(length(ϕ(ones(5)))) - 0.5
# penalty_parameter = 1 #2e-7
# gradient_step = 10^-6
# θᵛhistory = [θᵛ]
# Vhistory = [V(θᵛ,x₀)]
#
# for iter = 1:100
#     println("Epoch: $iter")
#     current_penalty_parameter = iter <= 10 ? penalty_parameter : 0.5*penalty_parameter
#     θᵛ,gradient_step = solve_problem(θᵛ,
#         gradient_step = gradient_step,
#         adjoint_step = 0.001,
#         penalty_parameter = current_penalty_parameter,
#         n_outer_iter = iter<5 ? 10 : 5,
#         n_inner_iter = iter<5 ? 20 : 10,
#         current_epoch = iter,
#         debug = false)
#     #current_learning_rate = iter == 1 ? 1 : max(0.05,learning_rate*(1 - sqrt(iter)/4.9))
#     #global θᵛ = last(θᵛhistory)*(1-current_learning_rate) + θᵛ * current_learning_rate
#     global θᵛhistory = [θᵛhistory..., θᵛ]
#     gradient_step /= 10
#     br1, br2, br3, br4 = objective_breakdown([θᵛ;ones(41*length(batch));ones(41*length(batch));current_penalty_parameter])
#     println("Objective breakdown = [$br1, $br2, $br3, $br4], gradient step: $(gradient_step)")
#     println("V(θ[$iter],x[$(batch[1])]) = ",V(θᵛ,batch[1]))#
#     global Vhistory = [Vhistory..., V(θᵛ,x₀)]
#     previous_V_batch = V_batch
#     V_batch = [V(θᵛ,x) for x in batch]
#     batch_objective = sum((V_batch - previous_V_batch) .^2)
#     println("Batch[$batch_iter] = $batch_objective")
#     println("θ[$iter] = $θᵛ")
# end

function generate_scenario(θᵛ,f::Function, name::String, interval, should_plot)
    scenario = DataFrame()
    scenario[:q] = vcat([q for q in interval, h in interval]...)
    scenario[:h] = vcat([h for q in interval, h in interval]...)
    scenario[:bid] = vcat([maxδᵇ(θᵛ,f(q,h)) for q in interval, h in interval]...)
    scenario[:offer] = vcat([maxδᵃ(θᵛ,f(q,h)) for q in interval, h in interval]...)
    scenario[:hedge] = vcat([maxξ(θᵛ,f(q,h)) for q in interval, h in interval]...)
    scenario[:wealth] = vcat([v(f(q,h)) for q in interval, h in interval]...)
    scenario[:utility] = vcat([R(f(q,h)) for q in interval, h in interval]...)
    scenario[:value] = vcat([V(θᵛ,f(q,h)) for q in interval, h in interval]...)
    if should_plot
        surface(scenario[:q],scenario[:h],scenario[:bid], title="Bid - $name")
        surface(scenario[:q],scenario[:h],scenario[:offer],  title="Offer - $name")
        surface(scenario[:q],scenario[:h],scenario[:hedge],  title="Hedge - $name")
        surface(scenario[:q],scenario[:h],scenario[:wealth], title="Wealth - $name")
        surface(scenario[:q],scenario[:h],scenario[:utility], title="Utility - $name")
        surface(scenario[:q],scenario[:h],scenario[:value], title="Value - $name")
        contour(scenario[:q],scenario[:h],scenario[:value], title="Value - $name")
    end
    return scenario
end

#last(θ)
#Same wealth scenario, different arrangements of position
generate_scenario(last(θ),(q,h) -> x₀+[-100*(q+h),0,0,q,h], "Scenario 1",-2000:100:2000, true)
# #Zero cash scenarion, different arrangements of position
generate_scenario(last(θ),(q,h) -> x₀+[0,0,0,q,h], "Scenario 2",-2000:100:2000, true)
# #Same wealth scenario, different arrangements of position
#generate_scenario(θᵛ,(q,h) -> x₀+[-100*(q+h),0,0,q,h], "Scenario 1",-2000:100:2000, true)
# #Zero cash scenarion, different arrangements of position
#generate_scenario(θᵛ,(q,h) -> x₀+[0,0,0,q,h], "Scenario 2",-2000:100:2000, true)


plot(1:length(θ),[V(θ[i],x₀) for i in 1:length(θ)],title="x0")
plot(1:length(θ),[V(θ[i],x₀+[1000,0,0,0,0]) for i in 1:length(θ)],title="x0 with initial cash 1k")
plot(1:length(θ),[V(θ[i],x₀+[0,0,0,2000,-2000]) for i in 1:length(θ)],title="x0 with hedged position")
