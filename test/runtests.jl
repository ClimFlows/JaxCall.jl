using Serialization
using Enzyme
using JaxCall: JaxCall, to_rarray, compile
using Reactant: @code_hlo, @compile
using Test

function timeit(N, fun, q) 
    fun(q)
    @time for _ in 1:N ; fun(q) ; end
end

f(q) = q*q
q = randn(Float32, 1000, 1000) # square, to avoid shape issues
code = string(@code_hlo f(to_rarray(q)))

@info "Time for 10 Julia 1000x1000 matmul"
timeit(10, f, q)

@info "Timing for 1 non-compiled Reactant 1000x1000 matmul"
f_hlo = JaxCall.HLOFun(code)
rq = to_rarray(q)
compiled_f = @compile f_hlo(rq) 
timeit(10, compiled_f, rq)

@info "Timing for 10 Reactant 1000x1000 matmul"
compiled_f = compile(code, q)
# timeit(10, compiled_f, rq)
timeit(10, compiled_f, q)

rq2 = compiled_f.forward(rq)
dq = compiled_f.backward(rq2, (rq,))

#=============================================================#

transp(x) = Array(x')

f(a,b) = a*b

a = randn(Float32, 100, 100) # square, to avoid shape issues
b = randn(Float32, 100, 100) # square, to avoid shape issues
ra, rb = to_rarray((a,b))
code = string(@code_hlo f(ra,rb))
compiled_f = compile(code, a, b)

rc = compiled_f.forward(ra,rb)
dc = compiled_f.backward(rc, (ra,rb))

g(cf, a,b) = sum(cf(a,b))
dup(x) = Duplicated(x, copy(x))

@info "g" g(f, a',b') g(compiled_f, a,b)

autodiff(Reverse, Const(g), Active, Const(f), dup(transp(a)), dup(transp(b)))

autodiff(Reverse, Const(g), Active, Const(compiled_f), dup(a), dup(b))

#===== check HLO functions can be differentiated as ordinary functions ====#

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
