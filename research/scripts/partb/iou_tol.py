# Temporal IoU tolerance analysis for the Part A metric
def iou(a,b):
    inter=max(0,min(a[1],b[1])-max(a[0],b[0])); uni=max(a[1],b[1])-min(a[0],b[0]); return inter/uni
print("dur | max pure shift for IoU>=0.3/0.5/0.7 | max symmetric pad each side for 0.3/0.5/0.7 | max late start (end exact) 0.5/0.7")
for D in [1.5,2,3,5,8,10,15,30,60,120]:
    # shift s: (D-s)/(D+s)>=t -> s<=D(1-t)/(1+t)
    sh=[D*(1-t)/(1+t) for t in (0.3,0.5,0.7)]
    # symmetric pad e each side: D/(D+2e)>=t -> e<=D(1-t)/(2t)
    pad=[D*(1-t)/(2*t) for t in (0.3,0.5,0.7)]
    # late start by d (pred shorter): (D-d)/D>=t -> d<=D(1-t)
    late=[D*(1-t) for t in (0.5,0.7)]
    print(f"{D:5.1f}s | "+" / ".join(f"{x:5.2f}" for x in sh)+" | "+" / ".join(f"{x:5.2f}" for x in pad)+" | "+" / ".join(f"{x:5.2f}" for x in late))
# stopped vehicle: rule fires at 10 s confirmation instead of at stop
for D in [15,20,30,60,120]:
    print("stopped_vehicle GT",D,"s; start reported 10 s late -> IoU",round(iou((0,D),(10,D)),3))
# macro F1: effect of predicting an absent class
print()
for k in [4,6,8,10]:
    for S in [0.3,0.5]:
        print(f"|C|={k}, current Score_A={S}: after adding one absent class -> {S*k/(k+1):.3f} (loss {S/(k+1):.3f})")
# break-even probability that a class is present, to enable it
print()
for k in [6,8]:
    for S in [0.3,0.4]:
        for F in [0.2,0.4,0.6]:
            # enable: if present, gain F/(k) approx (class already in C since GT has it: without prediction it contributes 0)
            # if absent and we emit >=1 FP: lose S*k/(k+1) - S = -S/(k+1)
            # break-even p: p*F/k = (1-p)*S/(k+1)
            gain=F/k; loss=S/(k+1); p=loss/(gain+loss)
            print(f"|C|={k} S={S} F1_c_if_present={F}: enable if P(present)>{p:.2f} (assuming it fires at least once when absent)")
