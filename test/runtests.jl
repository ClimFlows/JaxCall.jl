import JaxCall
import Enzyme

using Test
using InteractiveUtils

fun(b,c) = b*c

function main1()
    M, N, P = 10,20,40
    b = randn(M,N)
    c = randn(N,P)
    cfun = JaxCall.compile(fun, b, c ; verbose=true)

    @code_warntype cfun(b,c)

    a = randn(M,P)
    loss(b,c) = sum(abs2, cfun(b,c)-a)
    # L(b,c) = <bc-a | bc-a>
    # dL = 2 < b⋅c-a | b.dc + db.c >
    #    = 2 < b'⋅(bc-a) | dc >  2 < (b⋅c-a)⋅c' | db >
    # dL/db = (b⋅c-a)⋅c'
    # dL/dc = b'⋅(b⋅c-a)
    @info loss(b,c)
    dLdb, dLdc = Enzyme.gradient(Reverse, Const(loss), b, c)
    d = b*c-a
    @test dLdb ≈ 2d*c'
    @test dLdc ≈ 2b'*d
end

main1()
