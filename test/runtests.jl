using Serialization
using Enzyme
using JaxCall
using Test

L2(x) = sum(x.^2)
L1(x) = mapreduce(abs, max, x)

rmax(x) = abs(x)
rmax(x::NamedTuple) = mapreduce(rmax, max, x)

diff(x::Array, y::Array) = L1(x-y)/(L1(x)+L1(y))
diff(x::N, y::N) where {N<:NamedTuple} = map(diff, x, y)

function make_grad(model, params, x, pred)
#    loss(params, x) = L2(model(params, x)-pred)
    loss(params, x) = L2(model(params, x))
    @info "make_grad" loss(params, x)
    return (params, x) -> gradient(Reverse, Const(loss), params, x)
end

function time_grad(fun, params, x)
    model = JaxCall.compile(fun, params, x ; verbose=true)
    pred = model(params, x)
    grad = make_grad(model, params, x, pred)
    @time grad(params, x)
    @time grad(params, x)
end

function test()
  fun(params, x) = @. sin(params)*cos(x)
  x = randn(Float32, 1000)
  params = randn(Float32, 1000)
  time_grad(fun, params, x)
end

# test()

function load_model(filename)
    (code, params, jgrads, x) = Serialization.deserialize(string(filename))
    return jgrads, time_grad(code, params, x)
end

jax_grads, (enz_grads, dx) = load_model("small.jld")
err = diff(enz_grads, jax_grads)
@info "check" err rmax(err)
@test rmax(err)<2e-7
