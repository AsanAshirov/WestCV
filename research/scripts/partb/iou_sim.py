import numpy as np
rng = np.random.default_rng(0)
taus = [0.3,0.5,0.7]
def iou(a0,a1,b0,b1):
    inter = np.clip(np.minimum(a1,b1)-np.maximum(a0,b0),0,None)
    union = (a1-a0)+(b1-b0)-inter
    return inter/union
print("Deterministic tolerance: max symmetric shift d (same length) for IoU>=tau: d <= L*(1-tau)/(1+tau)")
for L in [2,3,5,8]:
    print(L, [round(L*(1-t)/(1+t),2) for t in taus])
print("\nOne boundary wrong only: shorter pred d<=L(1-tau); longer pred d<=L(1/tau-1)")
for L in [2,3,5,8]:
    print(L, "short", [round(L*(1-t),2) for t in taus], "long", [round(L*(1/t-1),2) for t in taus])

print("\nExpected mean-over-tau match prob vs noise sigma (independent gaussian on start/end), and padding p added each side")
N=200000
for L in [2,3,5]:
    for sigma in [0.25,0.5,1.0,1.5]:
        best=None; row=[]
        for p in [-0.25,0,0.25,0.5,0.75,1.0]:
            s = rng.normal(0,sigma,N)-p; e = L+rng.normal(0,sigma,N)+p
            ok = e>s+0.04
            v = iou(np.where(ok,s,0),np.where(ok,e,0.04),0,L)
            m = np.mean([np.mean((v>=t)&ok) for t in taus])
            row.append((p,round(m,3)))
            if best is None or m>best[1]: best=(p,m)
        print(f"L={L}s sigma={sigma}s: ", row, " best pad", best[0])
