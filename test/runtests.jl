module Local

using Enzyme
using Enzyme.EnzymeRules
using Reactant: Reactant, ConcreteRArray, @compile, @code_hlo

# We want to call JAX functions with their "natural" arguments
# but HLO expects that we flatten nested structures first.
# For this flattening to be type-stable on the Julia side,
# the "natural" arguments must provide their structure in the type domain.
# Nested NamedTuples do the job. So the procedure is to
# (i) convert any PyTree (nested dict) to a NamedTuple, once for all, using `to_namedtuple`
# (ii) provide these NamedTuples as args to JAX functions
# (iii) fun::CompiledFun transparently flattens arguments,
# (iv)    converts Arrays to ConcreteRArrays before passing them to hlo_call,
# (v)     and converts the returned result to Arrays.

#================ Compiled function and its adjoint ================#

struct CompiledFun{Fun,FunC,FunD}
    fun::Fun # original function
    forward::FunC # original function, compiled
    backward::FunD # adjoint, compiled
end
function (cfun::CompiledFun)(args...)
    args = as_flat_tuple(args)
    return to_array(cfun.forward(to_rarray(args)...))
end

function compile(code, args... ; verbose=false)
    code = String(code)
    fun(args...) = strip(Reactant.Ops.hlo_call(code, args...))
    # args is expected to be a (nested) Tuple / NamedTuple of Arrays
    cfun = compile(fun, as_flat_tuple(args)... ; verbose)
    return CompiledFun(code, cfun.forward, cfun.backward)
end

function compile(fun::Function, inputs...; verbose=false)
    # `fun` is a pure function which takes inputs and returns outputs
    dfun(douts, ins) = dotprod(douts, fun(ins...))
    function dfun!(douts, ins)
        _, dins = Enzyme.gradient(Reverse, dfun, Const(douts), ins)
        return dins
    end

    inputs = to_rarray(inputs)
    verbose && @info @code_hlo fun(inputs...)
    forward = @compile fun(inputs...)

    outputs = forward(inputs...)
    verbose && @info @code_hlo dfun!(outputs, inputs)
    backward = @compile dfun!(outputs, inputs)
    backward(make_zero(outputs), inputs)
    return CompiledFun(fun, forward, backward)
end

#===================== Custom Enzyme pullback ======================#

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Annotation{<:CompiledFun},
                                      ::Type{<:Annotation},
                                      inputs::Duplicated...)
    args = map(x -> x.val, inputs)
    args = as_flat_tuple(args)
    args = to_rarray(args) # this creates a copy
    primal = to_array(func.val.forward(args...))
    shadow = needs_shadow(config) ? make_zero(primal) : nothing
    primal = needs_primal(config) ? primal : nothing
    tape = shadow, args
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(::RevConfig,
                             func::Const{<:CompiledFun},
                             outs::Type{<:Annotation}, # return value
                             (douts, ins), # tape
                             args::Duplicated...)
    dins = func.val.backward(to_rarray(douts), ins)
    @info "reverse" rmax(douts) rmax(dins)
    dvals = as_flat_tuple(map(x -> x.dval, args))
    @info "reverse before addto!" rmax(dvals)
    foreach(addto!, dvals, dins)
    @info "reverse after addto!" rmax(dvals)
    make_zero!(douts)
    return map(x -> nothing, args)
end

#======================== Helper functions =========================#
rmax(x::Union{Tuple, NamedTuple}) = map(rmax, x)
rmax(x::AbstractArray) = Float32(maximum(x)) # mapreduce(abs, max, x)

#=
to_rarray(x::Array) = ConcreteRArray(x)
to_rarray(x::Tuple) = map(to_rarray, x)
to_array(x::AbstractArray{T,N}) where {T,N} = Array{T,N}(x)
to_array(x::Tuple) = map(to_array, x)

strip(x) = x
strip(x::Tuple{T}) where {T} = x[1]

dotprod(x::A, y::A) where {A<:Array} = sum(x .* y)

function dotprod(x::A, y::A) where {N,A<:Reactant.RArray{<:Number,N}}
    dims = collect(1:N)
    return Reactant.Ops.dot_general(x, y; contracting_dimensions=(dims, dims))
end

dotprod(x::Tuple, y::Tuple) = mapreduce(dotprod, +, x, y)
=#
using JaxCall: to_rarray, to_array, strip, dotprod

function addto!(x::Array, xx::ConcreteRArray{T}) where {T}
    @assert size(x) == size(xx)
    buf = Reactant.get_buffer(xx)
    ptr = Base.unsafe_convert(Ptr{T}, Reactant.XLA.unsafe_buffer_pointer(buf))
    @inbounds for i in eachindex(x)
        x[i] += unsafe_load(ptr, i)
    end
    return nothing
end

#=
function visit(TT::Type{<:NamedTuple}, var, leaves)
    # notice that we sort field names before visiting each field
    # this is consistent with observed JAX behavior when flattening PyTrees
    for (name, X) in sort(collect(zip(TT.parameters[1], TT.parameters[2].parameters)))
        visit(X, :($var.$name), leaves)
    end
    return leaves
end

function visit(TT::Type{<:Tuple}, var, leaves)
    for (i, X) in enumerate(TT.parameters)
        visit(X, :($var[$i]), leaves)
    end
    return leaves
end
visit(::Type, var, leaves) = push!(leaves, var)

# in this generator body, `x` is actually typeof(x)
@generated as_flat_tuple(x::NamedTuple) = Expr(:tuple, visit(x, :x, [])...)
@generated as_flat_tuple(x::Tuple) = Expr(:tuple, visit(x, :x, [])...)
=#
using JaxCall: as_flat_tuple

end # module Local

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
