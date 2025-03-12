using JaxCall: JaxCall, compile, addto!
using Enzyme: Enzyme, Annotation, Duplicated, gradient, Reverse, Const
using Enzyme.EnzymeRules
Enzyme.Compiler.VERBOSE_ERRORS[] = true

using Test
using InteractiveUtils
using Serialization

# using CondaPkg
# using PythonCall
# CondaPkg.resolve()

L2(x) = sum(x .^ 2)
dot(x, y) = sum(x .* y)

# a⋅(b*c) = Tr(a'*b*c)
# d(a⋅(b*c)) = Tr(a'*db*c) + Tr(a'*b*dc)
#    = Tr((a*c')'*db) + Tr((b'*a)'*dc)
#    = (a*c')⋅db + (b'*a)⋅dc
# => (df/db)[a] = a*c'
#    (df/dc)[a] = b'*a

M, N, P = 10, 20, 40
a = randn(M, P)
b = randn(M, N)
c = randn(N, P)

function main1(a, b, c)
    fun(a, b, c) = dot(a, b * c)
    grad = @time gradient(Reverse, Const(fun), a, b, c)
    @assert grad[1] ≈ b * c
    @assert grad[2] ≈ a * c'
    @assert grad[3] ≈ b' * a
end

#===================== Custom Enzyme pullback ======================#

struct CustomFun
end
@noinline (::CustomFun)(b, c) = b * c

function EnzymeRules.augmented_primal(config::RevConfig,
                                      ::Annotation{<:CustomFun},
                                      ::Type{<:Annotation},
                                      inputs::Duplicated...)
    return custom_forward(config, inputs)
end

use_rarray() = false

@noinline function custom_forward(config, inputs)
    forward(b, c) = b * c
    args = map(x -> x.val, inputs)
    args = JaxCall.as_flat_tuple(args)
    if use_rarray()
        args = JaxCall.to_rarray(args) # this creates a copy
        primal = JaxCall.to_array(forward(args...))
    else
        args = map(copy, args)
        primal = forward(args...)
    end
    shadow = needs_shadow(config) ? Enzyme.make_zero(primal) : nothing
    primal = needs_primal(config) ? primal : nothing
    tape = shadow, args
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(::RevConfig,
                             ::Const{<:CustomFun},
                             ::Type{<:Annotation}, # return value
                             tape,
                             args::Duplicated...)
    return custom_backward(tape, args)
end

@noinline function custom_backward((douts, ins), args)
    backward(a, (b, c)) = a * c', b' * a
    if use_rarray()
        dins = backward(JaxCall.to_rarray(douts), ins)
    else
        dins = backward(douts, ins)
    end
    dvals = map(x -> x.dval, args)
    dvals = JaxCall.as_flat_tuple(dvals)
    foreach(addto!, dvals, dins)
    @info "here" typeof(dvals) typeof(dins)

    Enzyme.make_zero!(douts)
    return map(x -> nothing, args)
end

JaxCall.addto!(x::Array, y::Array) = @. x += y

function main2(a, b, c)
    backward(a, (b, c)) = a * c', b' * a
    loss(b, c) = dot(a, cfun(b, c))
    cfun = CustomFun()

    @time begin
        config = EnzymeRules.RevConfig{true,true,1,true,true}()
        b_dup = Duplicated(b, zero(b))
        c_dup = Duplicated(c, zero(c))
        augmented = EnzymeRules.augmented_primal(config, Const(cfun), Duplicated, b_dup,
                                                 c_dup)
        EnzymeRules.reverse(config, Const(cfun), Duplicated, augmented.tape, b_dup, c_dup)
    end

    grad = @time gradient(Reverse, Const(loss), b, c)
    @assert grad[1] ≈ a * c'
    @assert grad[2] ≈ b' * a
end

function main3(a, b, c)
    forward(b, c) = b * c
    backward(a, (b, c)) = a * c', b' * a
    aa, bb, cc = JaxCall.to_rarray((a, b, c))
    fwd = JaxCall.Reactant.@compile forward(bb, cc)
    bwd = JaxCall.Reactant.@compile backward(aa, (bb, cc))
    cfun = JaxCall.CustomFun(nothing, fwd, bwd)

    loss(b, c) = dot(a, cfun(b, c))
    grad = @time gradient(Reverse, Const(loss), b, c)
    @assert grad[1] ≈ a * c'
    @assert grad[2] ≈ b' * a
end

function main4(a, b, c)
    # @code_hlo messes up the order of indices => we need to transpose
    forward(b, c) = c * b
    aaa, bbb, ccc = map(JaxCall.to_rarray ∘ Array, (a', b', c'))
    code = JaxCall.Reactant.@code_hlo forward(bbb,ccc)
    cfun = JaxCall.compile_jax(code, b,c ; verbose=true)

    loss(b, c) = dot(a, cfun(b, c))
    grad = @time gradient(Reverse, Const(loss), b, c)
    @assert grad[1] ≈ a * c'
    @assert grad[2] ≈ b' * a
end

function main5(a, b, c)
    forward(b, c) = b*c
    loss(b, c) = dot(a, cfun(b, c))
    cfun = JaxCall.compile(forward, b,c ; verbose=true)
    grad = @time gradient(Reverse, Const(loss), b, c)
    @assert grad[1] ≈ a * c'
    @assert grad[2] ≈ b' * a
end

function main6()
    (; code, params, x) = deserialize("MLP.jld")
    model = JaxCall.compile_jax(code, params, x ; verbose=true)
    pred = model(params, x)
    grad = make_grad(model, params, x, pred)
    @time grad(params, x)
    @time grad(params, x)
end

function make_grad(model, params, x, pred)
    loss(params, x) = L2(model(params, x)-pred)
    return (params,x) -> gradient(Reverse, Const(loss), params, x)
end

function main7()
    loss(params, x) = L2(model(params, x)-pred)
    grad(params,x) = gradient(Reverse, Const(loss), params, x)
    (; code, params, x) = deserialize("MLP.jld")
    model = JaxCall.compile_jax(code, params, x ; verbose=true)
    pred = model(params, x)
    @time grad(params, x)
    @time grad(params, x)
end

# main1(a, b, c) # works
# main2(a, b, c) # works if use_rarray() == false
# main3(a, b, c) # fails
# main4(a,b,c) # fails
# main5(a,b,c) # fails
main6()

#==================================================================#

#=
function main2()
    fun(b,c) = b*c
    loss(b,c) = L2(fun(b,c)-a)
#    closs(b,c) = L2(cfun(b,c)-a)

    M, N, P = 10,20,40
    b = randn(M,N)
    c = randn(N,P)
    a = randn(M,P)

#    cfun = compile(fun, b, c ; verbose=true)
    @code_warntype cfun(b,c)

    grad = @time gradient(Reverse, Const(loss), b, c)
#    cgrad = @time gradient(Reverse, Const(closs), b, c)
#    @test fun(b,c) ≈ cfun(b,c)
#    @test grad[1] ≈ cgrad[1]
#    @test grad[2] ≈ cgrad[2]
end
=#