# QuasiStrided per-call contract! on cases.txt; prints `id dtype ns checksum`.
using QuasiStrided, StridedViews, Statistics
using QuasiStrided: contract!
gen(::Type{Float64}, i, s) = ((i*17+s*13+3)%97)/97 - 0.5
gen(::Type{ComplexF64}, i, s) = complex(((i*17+s*13+3)%97)/97 - 0.5, ((i*29+s*7+5)%89)/89 - 0.5)
ck(x::AbstractArray{Float64}) = sum(v*(1+(i-1)%7) for (i,v) in enumerate(vec(x)))
ck(x::AbstractArray{ComplexF64}) = sum((real(v)+0.5imag(v))*(1+(i-1)%7) for (i,v) in enumerate(vec(x)))
function timeit(f)
    for _ in 1:3; f(); end
    t = time_ns(); f(); one = max(time_ns()-t, 1)
    n = clamp(round(Int, 1e6/one), 1, 200_000)
    v = map(1:15) do _
        t = time_ns(); for _ in 1:n; f(); end; (time_ns()-t)/n
    end
    median(v)
end
function run(T, id, ca, cb, IA, IB, IC, ext)
    sz(l) = Tuple(ext[c] for c in l)
    A = reshape([gen(T, i-1, 1) for i in 1:prod(sz(IA))], sz(IA))
    B = reshape([gen(T, i-1, 2) for i in 1:prod(sz(IB))], sz(IB))
    C = zeros(T, sz(IC))
    lab(l) = Tuple(Int(c) for c in l)
    vA = ca ? conj(StridedView(A)) : StridedView(A)
    vB = cb ? conj(StridedView(B)) : StridedView(B)
    vC = StridedView(C)
    f = () -> contract!(vC, one(T), vA, lab(IA), vB, lab(IB), zero(T), lab(IC))
    f(); c = ck(C)
    ns = timeit(f)
    println(id, " ", T == Float64 ? "f64" : "c64", " ", round(ns, digits=1), " ", c); flush(stdout)
end
filt = length(ARGS) > 1 ? ARGS[2] : ""
for line in eachline(ARGS[1])
    isempty(line) && continue
    f = split(line)
    occursin(filt, f[1]) || continue
    ext = Dict(k[1] => parse(Int, v) for (k, v) in (split(kv, '=') for kv in split(f[8], ',')))
    IC = f[7] == "-" ? "" : f[7]
    run(f[2] == "f64" ? Float64 : ComplexF64, f[1], f[3] == "1", f[4] == "1", f[5], f[6], IC, ext)
end
