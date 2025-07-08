module JaxCall

using Enzyme
using Enzyme.EnzymeRules
using Reactant: Reactant, ConcreteRArray, @compile, @code_hlo

# We want to call a JAX functions with their "natural" arguments
# but HLO expects that we flatten nested structures first.
# For this flattening to be type-stable on the Julia side, 
# the "natural" arguments must provide their structure in the type domain.
# Nested NamedTuples do the job. So the procedure is to 
# (i) convert any PyTree (nested dict) to a NamedTuple, once for all, using `to_namedtuple` (see notebook)
# (ii) provide these NamedTuples as args to JAX functions
# (iii) fun::CustomFun transparently flattens arguments,
# (iv)    converts Arrays to ConcreteRArrays before passing them to hlo_call,
# (v)     and converts the returned result to Arrays.

#================ Compiled function and its adjoint ================#

struct CustomFun{Code,Fwd,Bwd}
    code::Code
    forward::Fwd
    backward::Bwd
end

@inline function (cfun::CustomFun)(args::Vararg{Any, N}) where N
    args = as_flat_tuple(args)
    rargs = to_rarray(args)
    result = cfun.forward(rargs...)
    return to_array(result)
end

function compile(code, args...; verbose=false)
    code = string(code)
    fun(args...) = strip(Reactant.Ops.hlo_call(code, args...))
    # args is expected to be a (nested) Tuple / NamedTuple of Arrays
    cfun = compile(fun, as_flat_tuple(args)...; verbose)
    return CustomFun(code, cfun.forward, cfun.backward)
end

function compile(fun::Function, inputs::Vararg{Any,N}; verbose=false) where N
    # `fun` is a pure function which takes inputs and returns outputs
    dfun(douts, ins) = dotprod(douts, fun(ins...))
    function dfun!(douts, ins)
        _, dins = Enzyme.gradient(Reverse, dfun, Const(douts), ins)
        return dins
    end

    inputs = to_rarray(inputs)
    code = @code_hlo fun(inputs...)
    verbose && @info code
    forward = @compile fun(inputs...)

    outputs = forward(inputs...)
    code = @code_hlo dfun!(outputs, inputs)
    verbose && @info code
    backward = @compile dfun!(outputs, inputs)
    backward(make_zero(outputs), inputs)
    return CustomFun(fun, forward, backward)
end

#===================== Custom Enzyme pullback ======================#

function EnzymeRules.augmented_primal(config::RevConfig,
                                      func::Annotation{<:CustomFun},
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
                             func::Const{<:CustomFun},
                             outs::Type{<:Annotation}, # return value
                             (douts, ins), # tape
                             args::Duplicated...)
    dins = func.val.backward(to_rarray(douts), ins)
    dvals = as_flat_tuple(map(x -> x.dval, args))
    foreach(addto!, dvals, dins)
    make_zero!(douts)
    return map(x -> nothing, args)
end

#======================== Helper functions =========================#

@inline to_rarray(x) = Reactant.to_rarray(x)
@inline to_rarray(x::Tuple) = map(to_rarray, x)

@inline to_array(x::AbstractArray) = convert(Array, x)
@inline to_array(x::Tuple) = map(to_array, x)

@inline strip(x) = x
@inline strip(x::Tuple{T}) where {T} = x[1]

@inline dotprod(x::A, y::A) where {A<:Array} = sum(x .* y)
@inline dotprod(x::F, y::F) where {F<:Number} = x*y
@inline dotprod(x::Tuple, y::Tuple) = mapreduce(dotprod, +, x, y)

function dotprod(x::A, y::A) where {N,A<:Reactant.RArray{<:Number,N}}
    dims = collect(1:N)
    return Reactant.Ops.dot_general(x, y; contracting_dimensions=(dims, dims))
end


function addto!(x::Array, yy::ConcreteRArray{T}) where {T}
    if false
        # this non-allocating variant seems to lead to memory corruption somehow
        @assert size(x) == size(yy)
        buf = Reactant.get_buffer(yy)
        ptr = Base.unsafe_convert(Ptr{T}, Reactant.XLA.unsafe_buffer_pointer(buf))
        @inbounds for i in eachindex(x)
            x[i] += unsafe_load(ptr, i)
        end
    else
        y = Array(yy)
        @. x += y
    end
    return nothing
end

function visit(TT::Type{<:NamedTuple}, var, leaves)
    # notice that we sort field names before visiting each field
    # this is consistent with observed JAX behavior
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
