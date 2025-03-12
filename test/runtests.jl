using Serialization
using Enzyme
using JaxCall

L2(x) = sum(x.^2)
rmax(x::Union{Tuple, NamedTuple}) = map(rmax, x)
rmax(x::AbstractArray) = maximum(x)

function make_grad(model, params, x, pred)
#    loss(params, x) = L2(model(params, x)-pred)
    loss(params, x) = L2(model(params, x))
    @info "make_grad" loss(params, x)
    return (params, x) -> gradient(Reverse, Const(loss), params, x)
end

function time_grad(fun, params, x)
#    model = Local.compile(fun, params, x ; verbose=true)
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
@info "check" rmax(enz_grads) rmax(jax_grads)
