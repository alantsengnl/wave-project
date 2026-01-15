#%% Plotting method II.a and II.b

import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = "wave-long-low-5.mp4"

N_RODS = 72
BAR_SPACING = 0.0125
SKIP_LEFT = 0
SKIP_RIGHT = 0

HALF_W = 8
HALF_H = 120

# Load first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

fps = cap.get(cv2.CAP_PROP_FPS)
gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
h, w = gray0.shape

# Detect rod tips
y1 = int(0.05 * h)
y2 = int(0.6 * h)

projection = gray0[y1:y2, :].mean(axis=0)
projection = np.convolve(projection, np.ones(15) / 15, mode="same")

threshold = projection.mean()
xs = np.where(projection > threshold)[0]
if len(xs) == 0:
    raise RuntimeError("Rod region not detected")

x_left = xs[0]
x_right = xs[-1]
x_expected = np.linspace(x_left, x_right, N_RODS)

pts = []
SEARCH = int((x_expected[1] - x_expected[0]) * 0.4)

for x0 in x_expected:
    x0 = int(round(x0))
    x1 = max(0, x0 - SEARCH)
    x2 = min(w, x0 + SEARCH)

    local = projection[x1:x2]
    x_peak = x1 + np.argmax(local)

    col = gray0[:, x_peak]
    ys = np.where(col >= 0.9 * col.max())[0]
    if len(ys) == 0:
        raise RuntimeError("Failed to find rod tip")

    pts.append((x_peak, ys.min()))

pts = np.array(pts)

# Track vertical motion
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
tracks = [[] for _ in range(N_RODS)]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i, (x0, y0) in enumerate(pts):
        x0 = int(x0)
        y0 = int(y0)

        y1 = max(0, y0 - HALF_H)
        y2 = min(gray.shape[0], y0 + HALF_H)
        x1 = max(0, x0 - HALF_W)
        x2 = min(gray.shape[1], x0 + HALF_W)

        roi = gray[y1:y2, x1:x2]
        profile = roi.mean(axis=1)

        rows = np.arange(len(profile))
        yc = np.sum(rows * profile) / np.sum(profile)

        tracks[i].append(y1 + yc)

cap.release()

tracks = [np.array(t) for t in tracks]
time = np.arange(len(tracks[0])) / fps

# Displacements
disp = [t - t.mean() for t in tracks]

# Arrival time detection
arrival_times = []

for y in disp:
    idx_min = np.argmin(y)
    t_cross = np.nan

    for k in range(idx_min, len(y) - 1):
        if y[k] < 0 and y[k + 1] >= 0:
            frac = -y[k] / (y[k + 1] - y[k])
            t_cross = time[k] + frac * (time[k + 1] - time[k])
            break

    arrival_times.append(t_cross)

arrival_times = np.array(arrival_times)

# Δt and speed
dt = np.abs(np.diff(arrival_times))
bar_idx = np.arange(1, len(dt) + 1)

dt_valid = dt[~np.isnan(dt)]
bar_idx_valid = bar_idx[~np.isnan(dt)]

med = np.median(dt_valid)
mad = np.median(np.abs(dt_valid - med))
sigma = 1.4826 * mad
k = 3.0

keep = np.abs(dt_valid - med) < k * sigma
dt_clean = dt_valid[keep]
bar_idx_clean = bar_idx_valid[keep]

dt_mean = np.mean(dt_clean)
v_mean = BAR_SPACING / dt_mean

# Plot Δt
plt.figure(figsize=(6, 4))
plt.scatter(bar_idx_valid, dt_valid, s=40, alpha=0.3, label="All Δt")
plt.scatter(bar_idx_clean, dt_clean, s=55, label="Kept Δt")
plt.axhline(dt_mean, color="red", linestyle="--",
            label=rf"$\langle\Delta t\rangle = {dt_mean:.4f}\,\mathrm{{s}}$")
plt.plot([], [], " ", label=rf"$v = {v_mean:.3f}\,\mathrm{{m/s}}$")
plt.xlabel("Bar number")
plt.ylabel("Δt (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Distance vs arrival time (forward wave)
rod_idx = np.arange(N_RODS)
x_positions = (rod_idx.max() - rod_idx) * BAR_SPACING

valid = ~np.isnan(arrival_times)
tv_all = arrival_times[valid]
xv_all = x_positions[valid]

t_cut = np.percentile(tv_all, 65)
keep = tv_all <= t_cut

tv = tv_all[keep]
xv = xv_all[keep]

order = np.argsort(tv)
tv = tv[order]
xv = xv[order]

coeffs = np.polyfit(tv, xv, 1)
v_fit, b_fit = coeffs
x_fit = np.polyval(coeffs, tv)

plt.figure(figsize=(6, 5))
plt.scatter(tv, xv, s=50, label="Arrival data")
plt.plot(tv, x_fit, color="red", linewidth=2, label="Linear fit")

plt.text(
    tv.min() + 0.05 * (tv.max() - tv.min()),
    xv.min() + 0.85 * (xv.max() - xv.min()),
    rf"$v = {v_fit:.3f}\,\mathrm{{m/s}}$",
    bbox=dict(facecolor="white", edgecolor="black"),
)

plt.xlabel("Arrival time (s)")
plt.ylabel("Distance (m)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print(f"Slope-based wave speed v_fit = {v_fit:.4f} m/s")

# %%
