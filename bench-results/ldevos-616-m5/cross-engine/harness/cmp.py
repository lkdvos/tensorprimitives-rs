import glob, math, collections, sys
E = ["tp","tfc","tf","qs"]
t = collections.defaultdict(dict); ck = collections.defaultdict(dict)
for e in E:
    for f in sorted(glob.glob(f"results/{e}_*.txt")):
        for l in open(f):
            i,d,ns,c = l.split(); k=(i,d)
            t[k].setdefault(e,[]).append(float(ns)); ck[k][e]=float(c)
best = {k:{e:min(v) for e,v in r.items()} for k,r in t.items()}
bad=[k for k in ck if any(abs(ck[k][e]-ck[k]["tp"])>1e-8*max(1,abs(ck[k]["tp"])) for e in ck[k])]
print("checksum mismatches:", bad)
def gm(x): return math.exp(sum(map(math.log,x))/len(x)) if x else float('nan')
def group(k):
    i,d=k
    if i.startswith("syn"):
        dim=int(i.split("_")[1][1:]); return f"TO synthetic {d} d={dim}"
    if i.startswith("tf_"): return f"tenferro MPS pairwise {d}"
    return f"MPS eff-H steps {d}"
G=collections.defaultdict(list)
for k in best: G[group(k)].append(k)
print(f"{'group':34s} {'n':>3s} {'tp ns(med)':>11s} {'tfc/tp':>14s} {'tf/tp':>14s} {'qs/tp':>14s}")
for g in sorted(G):
    ks=G[g]
    def col(e):
        r=[best[k][e]/best[k]["tp"] for k in ks if e in best[k]]
        return f"{gm(r):5.2f} [{min(r):.2f},{max(r):.2f}]"
    med=sorted(best[k]["tp"] for k in ks)[len(ks)//2]
    print(f"{g:34s} {len(ks):3d} {med:11.0f} {col('tfc'):>14s} {col('tf'):>14s} {col('qs'):>14s}")
if "-v" in sys.argv:
    for k in sorted(best, key=lambda k:(group(k),k)):
        b=best[k]; print(f"{k[0]:40s} {k[1]} tp {b['tp']:10.0f}  tfc {b['tfc']/b['tp']:5.2f} tf {b['tf']/b['tp']:5.2f} qs {b['qs']/b['tp']:5.2f}")
# spread between rounds for tp (noise)
sp=[max(v)/min(v) for k in t for v in [t[k]['tp']]]
print("tp round spread: median %.3f  p90 %.3f"%(sorted(sp)[len(sp)//2], sorted(sp)[int(.9*len(sp))]))
