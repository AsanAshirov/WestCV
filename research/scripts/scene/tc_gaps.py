"""Decode Sony NonRealTimeMeta LTC values (BCD FF SS MM HH, bit 0x40 of frames byte = drop-frame) and
infer how much footage was recorded in the clip-number gaps (camera is in Rec-Run TC)."""
import datetime as dt
FPS = 30000 / 1001
clips = {  # name: (frames, ltc_start, ltc_end, creation, lastUpdate)   -- from the 4 file tails
    "C3896": (10200, "58060311", "67460811", "12:18:24", "12:24:05"),
    "C3897": (9525, "68460811", "62041411", "12:24:47", "12:30:05"),
    "C3902": (9525, "43152011", "67322511", "17:58:18", "18:03:36"),
    "C3905": (3825, "58253611", "46333811", "18:22:21", "18:24:29"),
}
def dec(v):
    b = bytes.fromhex(v)
    df = bool(b[0] & 0x40)
    f = int(f"{b[0] & 0x3F:02x}"); s = int(f"{b[1] & 0x7F:02x}"); m = int(f"{b[2] & 0x7F:02x}"); h = int(f"{b[3] & 0x3F:02x}")
    return h, m, s, f, df
def df_to_frames(h, m, s, f):
    tm = 60 * h + m
    return (tm * 60 + s) * 30 + f - 2 * (tm - tm // 10)
prev = None
tot_gap = 0
for name, (n, a, b, c, u) in clips.items():
    ha, hb = dec(a), dec(b)
    fa, fb = df_to_frames(*ha[:4]), df_to_frames(*hb[:4])
    ok = (fb - fa + 1) == n
    print(f"{name}: {n} fr = {n / FPS:6.1f} s | TC {ha[0]:02d}:{ha[1]:02d}:{ha[2]:02d};{ha[3]:02d} -> {hb[0]:02d}:{hb[1]:02d}:{hb[2]:02d};{hb[3]:02d} DF={ha[4]} (TC span matches frame count: {ok}) | wall {c}-{u}")
    if prev:
        pn, pend, pu = prev
        gap = fa - pend - 1
        wall = (dt.datetime.strptime(c, "%H:%M:%S") - dt.datetime.strptime(pu, "%H:%M:%S")).total_seconds()
        print(f"   gap {pn}->{name}: clip numbers skipped = {int(name[1:]) - int(pn[1:]) - 1}, recorded TC in gap = {gap} fr = {gap / FPS:.1f} s ({gap / FPS / 60:.2f} min); wall-clock gap {wall / 60:.1f} min")
    prev = (name, fb, u)
