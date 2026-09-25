"""NOAA solar position (Spencer/NOAA spreadsheet equations) -> sunset, civil dusk, sun elevation at given clock times."""
import math, datetime as dt
def sun(lat, lon, tz, date, hh, mm):
    # returns elevation (deg) at local clock time
    jd = (date - dt.date(2000,1,1)).days + 2451544.5 + (hh + mm/60 - tz)/24
    T = (jd - 2451545)/36525
    L0 = (280.46646 + T*(36000.76983 + 0.0003032*T)) % 360
    M = 357.52911 + T*(35999.05029 - 0.0001537*T)
    e = 0.016708634 - T*(0.000042037 + 0.0000001267*T)
    C = (math.sin(math.radians(M))*(1.914602 - T*(0.004817+0.000014*T)) + math.sin(math.radians(2*M))*(0.019993-0.000101*T) + math.sin(math.radians(3*M))*0.000289)
    lam = L0 + C - 0.00569 - 0.00478*math.sin(math.radians(125.04-1934.136*T))
    eps0 = 23 + (26 + (21.448 - T*(46.815 + T*(0.00059 - T*0.001813)))/60)/60
    eps = eps0 + 0.00256*math.cos(math.radians(125.04-1934.136*T))
    decl = math.degrees(math.asin(math.sin(math.radians(eps))*math.sin(math.radians(lam))))
    y = math.tan(math.radians(eps/2))**2
    eqt = 4*math.degrees(y*math.sin(2*math.radians(L0)) - 2*e*math.sin(math.radians(M)) + 4*e*y*math.sin(math.radians(M))*math.cos(2*math.radians(L0)) - 0.5*y*y*math.sin(4*math.radians(L0)) - 1.25*e*e*math.sin(2*math.radians(M)))
    tst = (hh*60 + mm + eqt + 4*lon - 60*tz) % 1440
    ha = tst/4 - 180
    zen = math.degrees(math.acos(math.sin(math.radians(lat))*math.sin(math.radians(decl)) + math.cos(math.radians(lat))*math.cos(math.radians(decl))*math.cos(math.radians(ha))))
    return 90 - zen
def cross(lat, lon, tz, date, target):
    prev=None
    for m in range(12*60, 22*60):
        el = sun(lat, lon, tz, date, m//60, m%60)
        if prev is not None and prev > target >= el: return f"{m//60:02d}:{m%60:02d}"
        prev = el
d = dt.date(2026,9,18)
for name, lat, lon, tz in [("Tashkent UTC+5", 41.2995, 69.2401, 5), ("Tashkent if clock were UTC+6", 41.2995, 69.2401, 6)]:
    print(name, "sunset(-0.833)", cross(lat,lon,tz,d,-0.833), "civil dusk(-6)", cross(lat,lon,tz,d,-6))
    for hh,mm in [(12,18),(12,30),(17,58),(18,4),(18,22),(18,24)]:
        print(f"   clock {hh:02d}:{mm:02d} sun elevation {sun(lat,lon,tz,d,hh,mm):5.1f} deg")
