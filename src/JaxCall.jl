module JaxCall

using Enzyme
using Enzyme.EnzymeRules
using Reactant: Reactant, ConcreteRArray, @compile, @code_hlo

using Serialization

# We want to call a JAX functions with their "natural" arguments
# but HLO expects that we flatten nested structures first.
# For this flattening to be type-stable on the Julia side, 
# the "natural" arguments must provide their structure in the type domain.
# Nested NamedTuples do the job. So the procedure is to 
# (i) convert any PyTree (nested dict) to a NamedTuple, once for all, using `to_namedtuple`
# (ii) provide these NamedTuples as args to JAX functions
# (iii) fun::CompiledFun transparently flattens arguments,
# (iv)    converts Arrays to ConcreteRArrays before passing them to hlo_call,
# (v)     and converts the returned result to Arrays.

#================= Save model to binary file =================#

function save_model(filename, code, params, x)
    model = (code=string(code),
             params=to_namedtuple(params),
             x=Array(x))
    return Serialization.serialize(string(filename), model)
end

#================ Compiled function and its adjoint ================#

struct CompiledFun{Fun,FunC,FunD}
    fun::Fun # original function
    compiled::FunC # original function, compiled
    compiled_diff::FunD # adjoint, compiled
end
function (cfun::CompiledFun)(args...)
    args = as_flat_tuple(args)
    return to_array(cfun.compiled(to_rarray(args)...))
end

function compile_jax(code, args...)
    code = String(code)
    fun(args...) = strip(Reactant.Ops.hlo_call(code, args...))
    # args is expected to be a (nested) Tuple / NamedTuple of Arrays
    cfun = compile(fun, as_flat_tuple(args)...)
    return CompiledFun(code, cfun.compiled, cfun.compiled_diff)
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
    compiled = @compile fun(inputs...)

    outputs = compiled(inputs...)
    verbose && @info @code_hlo dfun!(outputs, inputs)
    compiled_diff = @compile dfun!(outputs, inputs)
    compiled_diff(make_zero(outputs), inputs)
    return CompiledFun(fun, compiled, compiled_diff)
end

#===================== Custom Enzyme pullback ======================#

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Annotation{<:CompiledFun},
                                      ::Type{<:Annotation},
                                      inputs::Duplicated...)
    args = map(x -> x.val, inputs)
    args = as_flat_tuple(args)
    args = to_rarray(args) # this creates a copy
    primal = to_array(func.val.compiled(args...))
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
    dins = func.val.compiled_diff(to_rarray(douts), ins)
    dvals = as_flat_tuple(map(x -> x.dval, args))
    foreach(addto!, dvals, dins)
    make_zero!(douts)
    return map(x -> nothing, args)
end

#======================== Helper functions =========================#

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

function addto!(x::Array, xx::ConcreteRArray{T}) where {T}
    @assert size(x) == size(xx)
    buf = Reactant.get_buffer(xx)
    ptr = Base.unsafe_convert(Ptr{T}, Reactant.XLA.unsafe_buffer_pointer(buf))
    @inbounds for i in eachindex(x)
        x[i] += unsafe_load(ptr, i)
    end
    return nothing
end

function visit(TT::Type{<:NamedTuple}, var, leaves)
    # notice that we sort field names before visiting each field
    # this is consistent with observed JAX behavior
    # alternatively, `to_namedtuple` could sort names when generating the named tuple from the PyDict 
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


end # module JaxCall

#=

# this will go into an extension module

using PythonCall

to_namedtuple(x::PyArray) = Array(x)
function to_namedtuple(dict::PyDict)
    return NamedTuple(Symbol(key) => to_namedtuple(getindex(dict, key))
                      for key in keys(dict))
end
=#
