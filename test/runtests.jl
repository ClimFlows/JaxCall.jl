using JaxCall: compile
using Enzyme: Enzyme, gradient, Reverse, Const
Enzyme.Compiler.VERBOSE_ERRORS[] = true

using Test
using InteractiveUtils

using CondaPkg
using PythonCall

CondaPkg.resolve()

L2(x) = sum(x.^2)

function main1()
    fun(b,c) = b*c
    loss(b,c) = L2(fun(b,c)-a)
    closs(b,c) = L2(cfun(b,c)-a)

    M, N, P = 10,20,40
    b = randn(M,N)
    c = randn(N,P)
    a = randn(M,P)

    cfun = compile(fun, b, c ; verbose=true)
    @code_warntype cfun(b,c)

    grad = gradient(Reverse, Const(loss), b, c)
    cgrad = gradient(Reverse, Const(closs), b, c)
    @test fun(b,c) ≈ cfun(b,c)
    @test grad[1] ≈ cgrad[1]
    @test grad[2] ≈ cgrad[2]
end

main1()
