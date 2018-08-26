#include("./Parameters.jl")
#include("./Simulator.jl")
if something(findfirst(isequal("./"),LOAD_PATH),0) == 0
    push!(LOAD_PATH, "./")
end
#using Distributions
using ForwardDiff
using JuMP
using NLopt
using GR
using DataFrames
using MultivariateStats
using GLMNet
using Flux:batch
using DelimitedFiles
using Statistics
using Parameters
using Simulator

#Basis functions
ϕ(x) = [v(x), y(x), s(x), p(x), q(x), h(x), s(x)^2, p(x)^2, q(x)^2, h(x)^2, log(s(x)),
        s(x)*q(x), h(x)*s(x), h(x)*p(x), tanh(v(x)/1000), tanh((q(x)+h(x))/1000)]

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

function V_estimate(θ,x)
    V(θ,x) + max(LV(θ,x,maxδᵃ(θ,x),maxδᵇ(θ,x)),MV(θ,x,maxξ(θ,x)))
end

function total_reward(x₀;θᵃ=θᵃ,θᵇ=θᵇ,θʷ=θʷ,N=1000,T=1,Simulations=500,Safe=true)
    Δt = 1/N
    reward = 0
    reward = @parallel (+) for i = 1:Simulations
        reward_i = 0
        sim = simulate(x₀;N=N,T=T)
        t=1
        x = x₀
        for j in sim
            bid = Safe ? min(0.01,max(0.0001,δᵇ(x))) : δᵇ(θᵇ,x)
            offer = Safe ? min(0.01,max(0.0001,δᵃ(x))) : δᵃ(θᵃ,x)
            hedge = ξ(θʷ,x)
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
        reward_i
    end
    reward/N
end


X_support = [randx(19;support=true);[x₀]]
ϕ_X_support = ϕ.(X_support)
ϕ_X_data_support = permutedims(batch(ϕ_X_support))
V_support = R.(X_support)

X_sample = randx(100)
ϕ_X = ϕ.(X_sample)
ϕ_X_data = permutedims(batch(ϕ_X))

#The learning rates below indicate how much % we want to learn from the current best
#and how much we want to learn from the new optimized function, given the new data
learning_rate = 0.25
learning_best = 0.50

#Initial values for the function approximations parameters
θᵛ = zeros(length(ϕ(ones(5))))
θᵃ = zeros(length(ϕ(ones(5))))
θᵇ = zeros(length(ϕ(ones(5))))
θʷ = zeros(length(ϕ(ones(5))))

#The best reward is the best known expected total reward possible from state x₀
#we also keep track of the policies that generated the best reward
if !@isdefined best_reward
    best_θᵃ = θᵃ
    best_θᵇ = θᵇ
    best_θʷ = θʷ
    best_θᵛ = θᵛ
    best_reward = -Inf
end

function maxδᵃ(θ)
    mean([maxδᵃ(θ,x) for x in X_sample])
end

function maxδᵇ(θ)
    mean([maxδᵇ(θ,x) for x in X_sample])
end

function maxξ(θ)
    mean([maxξ(θ,x) for x in X_sample])
end

function V_estimate(θ)
    mean([V_estimate(θ,x) for x in X_sample])
end

function V_bar(θ)
    mean([V(θ,x) for x in X_sample])
end

function V_constraint1(θ)
    sum([∇Vx(θ,x) .* ∇R(x) for x in X_sample])
end

function second_order_condition(grad1,grad2)
    [(grad1[i]-grad1[j])*(min(grad2[i]-grad2[j],grad2[i]+grad2[j])) for i in 1:5 for j in i:5 if i!=j]
end

function V_constraint2(θ)
    sum([second_order_condition(∇Vx(θ,x),∇R(x)) for x in X_sample])
end

#Lagrangian objective
function objective(z)
    θ = z[1:length(θᵛ)]
    λ₁ = z[length(θᵛ)+1:length(θᵛ)+5]
    λ₂ = z[length(θᵛ)+6:length(θᵛ)+15]
    (V_estimate(θ)-V_bar(θ))^2 - λ₁'*V_constraint1(θ) - λ₂'*V_constraint2(θ) + sum(z.^2) #this is a penalty/regularization
end
objective(θ,λ₁,λ₂) = objective([θ;λ₁;λ₂])
∇objective(θ,λ₁,λ₂) = ForwardDiff.gradient(objective,[θ;λ₁;λ₂])[1:length(θᵛ)]

function minimize_lagrangian(θ₀,λ₁,λ₂;gradient_step = 0.000000002,n_iter = 40)
    F = zeros(n_iter)
    θ = [θ₀ for x in 1:n_iter]
    for iter = 1:n_iter
        F[iter] = objective(θ[iter],λ₁,λ₂)
        if iter>1
            if F[iter]>F[iter-1]
                println("> F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1])")
                return F[iter-1],θ[iter-1]
            elseif iter==n_iter
                return F[iter],θ[iter]
            else
                gradient_step *= 1.25
            end
        end
        θ[iter+1] = θ[iter] - gradient_step * ∇objective(θ[iter],λ₁,λ₂)
        println("> Objective F(θ[$iter])): $(F[iter])")
    end
end
function solve_problem(θ₀;n_outer_iter = 5,n_inner_iter=5,gradient_step = 0.000000002,adjoint_step=0.000000002)
    #Lagrange multipliers
    λ₁ = zeros(5)
    λ₂ = zeros(10)
    F = zeros(n_outer_iter)
    θ = [θ₀ for x in 1:n_outer_iter]
    for iter = 1:n_outer_iter
        println("Iteration $iter:")
        F[iter], θ[iter] = minimize_lagrangian(θ[iter],λ₁,λ₂;gradient_step=gradient_step,n_iter=n_inner_iter)
        λ₁ += adjoint_step * V_constraint1(θ[iter])
        λ₂ += adjoint_step * V_constraint2(θ[iter])
        println("Best F[$iter] = $(F[iter]) ")

        if iter>1
            if F[iter] > F[iter-1]
                println("F[$iter] is worse than F[$(iter-1)]: $(F[iter]) > $(F[iter-1])")
                return θ[iter-1]
            else iter==n_outer_iter
                return θ[iter]
            end
        end
        θ[iter+1] = θ[iter]
    end
    last(θ)
end

θᵛ = zeros(length(ϕ(ones(5))))

θᵛ = solve_problem(θᵛ,gradient_step = 0.00000000002)
