# Common case list: TO-suite synthetic :contract, tenferro MPS pairwise, MPS eff-H steps.
# Line: id dtype conjA conjB IA IB IC ext  (labels = single chars, col-major, IC = freeA ++ freeB)
import itertools
out = []
SHAPES = [(2,1,2),(2,2,2),(1,3,1),(1,0,1),(0,2,3),(3,2,3)]
def inter(a,b):
    n=min(len(a),len(b)); r=[]
    for i in range(n): r += [a[i],b[i]]
    return r+a[n:]+b[n:]
MAXEL, MAXFLOP = 1<<24, 4e9
for T in ["f64","c64"]:
    for dim in [4,8,16,32]:
        for (na,nc,nb) in SHAPES:
            oa=list("ABC"[:na]); c=list("klm"[:nc]); ob=list("xyz"[:nb])
            IC=oa+ob
            cands=[("gemm_ready",oa+c,c+ob),("a_permuted",inter(oa,c),c+ob),("b_permuted",oa+c,inter(c,ob)),
                   ("both_permuted",inter(oa,c),inter(c,ob)),("contract_scrambled",oa+c,c[::-1]+ob)]
            seen=set()
            for (lay,IA,IB) in cands:
                k=("".join(IA),"".join(IB))
                if k in seen: continue
                seen.add(k)
                if max(dim**len(IA),dim**len(IB),dim**max(len(IC),1))>MAXEL: continue
                if 2*dim**(na+nc+nb)>MAXFLOP: continue
                ext=",".join(f"{l}={dim}" for l in IA+IB if l not in [] )
                ext=",".join(sorted(set(ext.split(","))))
                out.append(f"syn_d{dim}_{na}{nc}{nb}_{lay} {T} 0 0 {''.join(IA)} {''.join(IB)} {''.join(IC) or '-'} {ext}")
# tenferro pairwise (c64, phys 2)
for chi in [4,8,16,32,64,128]:
    out.append(f"tf_env_bra_chi{chi} c64 0 1 ab asc bsc a={chi},b={chi},s=2,c={chi}")
    out.append(f"tf_tmp_ket_chi{chi} c64 0 0 bsc bsd cd b={chi},s=2,c={chi},d={chi}")
# MPS 1-site effective Hamiltonian, pairwise steps (d=2, w=6)
for T in ["f64","c64"]:
    for D in [16,32,64,128]:
        e=f"x={D},y={D},z={D},o={D},w=6,v=6,s=2,t=2"
        out.append(f"mps_L_ket_D{D} {T} 0 0 xwy ysz xwsz {e}")
        out.append(f"mps_T_mpo_D{D} {T} 0 0 xwsz wtsv xztv {e}")
        out.append(f"mps_T_R_D{D} {T} 0 0 xztv zvo xto {e}")
open("cases.txt","w").write("\n".join(out)+"\n")
print(len(out))
