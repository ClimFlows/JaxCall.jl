module JaxCall

using Enzyme
using Enzyme.EnzymeRules
using Reactant: Reactant, ConcreteRArray, @compile, @code_hlo
using Reactant.Compiler: Thunk

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

# compiled, callable and differentiable HLO function => for user consumption
struct CustomFun{Fwd<:Thunk, Bwd<:Thunk}
#    code::String
    forward::Fwd   # @compiled, RArray inputs/outputs
    backward::Bwd  # @compiled, RArray inputs/outputs
end

@inline function (fun::CustomFun)(args::Vararg{Any, N}) where N
    flat_args = as_flat_tuple(args)
    rargs = to_rarray(flat_args)
    results = strip(fun.forward(rargs...))
    return to_array(results)
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
#    @info "reverse!" typeof(ins) typeof(dins) typeof(dvals) outs typeof(douts)
    foreach(addto!, dvals, dins)
    make_zero!(douts)
    return map(x -> nothing, args)
end

#================= @compile HLO code and its adjoint ================#

# callable objects for internal use only, not compiled and/or not differentiable

# RArray inputs/outputs, to be compiled
struct HLOFun
    code::String
end
(fun::HLOFun)(args...) = strip(Reactant.Ops.hlo_call(fun.code, args...))

# function whose gradient provides the adjoint
# RArray inputs/outputs, to be differentiated
struct DFun
    code::String
end
function (dfun::DFun)(douts, ins...)
    fun = HLOFun(dfun.code)
    return dotprod(douts, fun(ins...))
end

# applies the adjoint of `code` to `douts`
# RArray inputs/outputs, to be compiled
struct DFun!
    code::String
end
function (dfun!::DFun!)(douts, ins)
    dfun = DFun(dfun!.code)
    _, dins... = Enzyme.gradient(Reverse, Const(dfun), Const(douts), ins...)
    return dins
end

function compile(code::String, args...; verbose=false)
    flat_args = as_flat_tuple(args)
    rargs = to_rarray(flat_args)
    fun = HLOFun(code) # not compiled
    cfun = @compile fun(rargs...)

    outs = cfun(rargs...)
    dfun! = DFun!(code) # not compiled
    dfun = @compile dfun!(outs, rargs)
    return CustomFun(cfun, dfun)
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
