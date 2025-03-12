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

struct CustomFun{Code, Fwd, Bwd}
    code::Code
    forward::Fwd
    backward::Bwd
end
function (cfun::CustomFun)(args...)
    args = as_flat_tuple(args)
    return to_array(cfun.forward(to_rarray(args)...))
end

function compile(code, args...; verbose=false)
    code = string(code)
    fun(args...) = strip(Reactant.Ops.hlo_call(code, args...))
    # args is expected to be a (nested) Tuple / NamedTuple of Arrays
    cfun = compile(fun, as_flat_tuple(args)... ; verbose)
    return CustomFun(code, cfun.forward, cfun.backward)
end

function compile(fun::Function, inputs...; verbose=false)
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

function addto!(x::Array, yy::ConcreteRArray{T}) where {T}
    @assert size(x) == size(yy)
    buf = Reactant.get_buffer(yy)
    ptr = Base.unsafe_convert(Ptr{T}, Reactant.XLA.unsafe_buffer_pointer(buf))
    @inbounds for i in eachindex(x)
        x[i] += unsafe_load(ptr, i)
    end
#    y = Array(yy)
#    @. x += y
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

