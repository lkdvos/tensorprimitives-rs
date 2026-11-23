# ab2.py A_glob B_glob : per-case min over rounds, ratio B/A, per-group geomean.
import sys,re,glob,math,collections
R=re.compile(r'^(.*?)\s+m=\d+\s+n=\d+\s+k=\d+\s+mr=(\d+) nr=(\d+) swap=(\d)\s+(\d+) ns')
def load(g):
    d=collections.defaultdict(list)
    for f in sorted(glob.glob(g)):
        for l in open(f):
            m=R.match(l.rstrip())
            if m: d[m.group(1).strip()].append(float(m.group(5)))
    return d
A,B=load(sys.argv[1]),load(sys.argv[2])
rows=[]
for k in A:
    if k not in B: continue
    a,b=min(A[k]),min(B[k]); sp=max(max(A[k])/a, max(B[k])/b)
    rows.append((k,a,b,b/a,sp))
v="-v" in sys.argv
for r in rows:
    if v: print(f"{r[0]:<34} A={r[1]:>10.0f} B={r[2]:>10.0f} B/A={r[3]:.3f} spread={r[4]:.3f}")
g=collections.defaultdict(list)
for r in rows: g[" ".join(r[0].split()[:2])].append(r[3])
for k,x in sorted(g.items()):
    print(f"{k:<14} n={len(x):3d} geomean={math.exp(sum(map(math.log,x))/len(x)):.3f} min={min(x):.3f} max={max(x):.3f}")
