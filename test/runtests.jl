using JaxCall: compile
using Enzyme: gradient, Reverse, Const

using Test
using InteractiveUtils

using CondaPkg
using PythonCall

CondaPkg.resolve()

L2(x) = sum(x.^2)

function main1()
    fun(b,c) = b*c
    loss(b,c) = L2(cfun(b,c)-a)

    # L(b,c) = <bc-a | bc-a>
    # dL = 2 < b⋅c-a | b.dc + db.c >
    #    = 2 < b'⋅(bc-a) | dc >  2 < (b⋅c-a)⋅c' | db >
    # dL/db = (b⋅c-a)⋅c'
    # dL/dc = b'⋅(b⋅c-a)

    M, N, P = 10,20,40
    b = randn(M,N)
    c = randn(N,P)
    a = randn(M,P)
    cfun = compile(fun, b, c ; verbose=true)
    @code_warntype cfun(b,c)
    @info loss(b,c)
    grad = gradient(Reverse, Const(loss), b, c)
    @info typeof(grad)
    d = b*c-a
#    @test dLdb ≈ 2d*c'
#    @test dLdc ≈ 2b'*d
end

main1()
