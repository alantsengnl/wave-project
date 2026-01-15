# %% [Block 1] Load video and get first frame

import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = "wave-long-high-2.mp4"
N_BARS = 40

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("Cannot read first video frame.")

print("Video loaded. Select bar tips in Block 2.")

# %% [Block 2] Select bar tips manually

pts = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < N_BARS:
            pts.append((x, y))
            print(f"Selected bar tip {len(pts)} of {N_BARS}: ({x}, {y})")

cv2.namedWindow("Select bar tips", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Select bar tips", cv2.WND_PROP_TOPMOST, 1)
cv2.imshow("Select bar tips", frame0)
cv2.setMouseCallback("Select bar tips", click_event)

print(f"Click on EXACTLY {N_BARS} bar tips...")

while True:
    disp = frame0.copy()
    for (x, y) in pts:
        cv2.circle(disp, (x, y), 6, (0, 0, 255), -1)
    cv2.imshow("Select bar tips", disp)

    key = cv2.waitKey(10) & 0xFF

    if len(pts) == N_BARS:
        print("All bar tips selected.")
        break

    if key == ord('q'):  
        break

cv2.destroyAllWindows()

print("Selected points:", pts)

# %% [Block 3] Prepare grayscale image and validate bar-tip locations

frame_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

h, w = frame_gray.shape
for (x, y) in pts:
    if x < 0 or x >= w or y < 0 or y >= h:
        raise RuntimeError(f"Selected point ({x}, {y}) is outside the image.")

print("Block 3 complete: grayscale frame ready, points validated.")

# %% [Block 4] Track bar positions over time using vertical brightness centroid

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

tracks = [[] for _ in range(N_BARS)]

HALF_W = 10      
HALF_H = 120    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(N_BARS):
        x0, y0 = pts[i]

        y1 = max(0, y0 - HALF_H)
        y2 = min(gray.shape[0], y0 + HALF_H)
        x1 = max(0, x0 - HALF_W)
        x2 = min(gray.shape[1], x0 + HALF_W)

        col = gray[y1:y2, x1:x2]

        vertical_profile = col.mean(axis=1)

        rows = np.arange(vertical_profile.shape[0], dtype=float)
        y_centroid = np.sum(rows * vertical_profile) / np.sum(vertical_profile)

        y_tracked = y1 + y_centroid

        tracks[i].append(y_tracked)

cap.release()
print("Tracking complete using centroid method.")

# %% [Block 5] Convert tracked y-positions into displacement signals

tracks = [np.array(t, dtype=float) for t in tracks]

T_candidates = [len(t) for t in tracks]
if len(set(T_candidates)) != 1:
    raise RuntimeError("Tracking error: bars have different lengths. Check Block 4.")

T = T_candidates[0]
time = np.arange(T) / fps

disp = [t - np.mean(t) for t in tracks]

print("Displacement signals generated.")

# %% [Block 6] Plot a single bar's displacement vs. time

BAR_ID = 7

# Safety check
if not (0 <= BAR_ID < N_BARS):
    raise ValueError(f"BAR_ID must be between 0 and {N_BARS-1}")

plt.figure(figsize=(10, 4))
plt.plot(time, disp[BAR_ID], linewidth=1.5)

# Zero line for reference
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("Vertical displacement (px)")
plt.title(f"Displacement of Bar {BAR_ID+1}")
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [Block 7] Plot all bar displacements

plt.figure(figsize=(12, 6))

if len(disp) != N_BARS:
    raise RuntimeError("disp array length mismatch — check previous blocks.")

for i in range(N_BARS):
    plt.plot(time, disp[i], linewidth=1.2, label=f"Bar {i+1}")

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("Vertical displacement (px)")
plt.title("Tracked bar motions")
plt.grid(True)
plt.legend(loc="upper left")  
plt.tight_layout()
plt.show()

# %% [Block 8] Plot average displacement of all tracked bars

if any(len(t) != disp[0].shape[0] for t in disp):
    raise RuntimeError("Displacement length mismatch — check Block 5.")

disp_matrix = np.vstack(disp)

avg_disp = disp_matrix.mean(axis=0)

plt.figure(figsize=(10, 4))

plt.plot(time, avg_disp, linewidth=2, label="Mean displacement", color="tab:blue")

plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')

plt.xlabel("Time (s)")
plt.ylabel("Average vertical displacement (px)")
plt.title("Average displacement of all tracked bars")
plt.grid(True)

std_disp = disp_matrix.std(axis=0)
plt.fill_between(time, avg_disp - std_disp, avg_disp + std_disp,
                 color="tab:blue", alpha=0.15, label="±1 std dev")

plt.legend()
plt.tight_layout()
plt.show()

# %% [Block 9] Detect wave arrival times using post-trough zero-crossing

arrival_times = []

for i in range(N_BARS):
    y = disp[i]

    # 1. Find main trough (global minimum)
    idx_min = np.argmin(y)

    # 2. Search forward for first upward zero-crossing
    t_cross = None
    for k in range(idx_min, len(y) - 1):
        if y[k] < 0 and y[k + 1] >= 0:
            # Linear interpolation for sub-frame timing
            frac = -y[k] / (y[k + 1] - y[k])
            t_cross = time[k] + frac * (time[k + 1] - time[k])
            break

    if t_cross is None:
        arrival_times.append(np.nan)
    else:
        arrival_times.append(t_cross)

arrival_times = np.array(arrival_times)

# %% [Block 10] Bar positions and data selection

BAR_SPACING = 0.0127  # meters (manual)

bar_indices = np.arange(N_BARS)
x_positions = bar_indices * BAR_SPACING  # meters

# Optional: skip early bars to avoid release artefacts
SKIP_FIRST =  0  # adjust if needed

valid = (~np.isnan(arrival_times)) & (bar_indices >= SKIP_FIRST)

x_valid = x_positions[valid]
t_valid = arrival_times[valid]

# %% [Block 11] Wave speed estimation via linear regression

# Fit x = v * t + b
coeffs = np.polyfit(t_valid, x_valid, 1)
v_est = coeffs[0]      # slope = wave speed
x_intercept = coeffs[1]

# Residuals for uncertainty
x_fit = np.polyval(coeffs, t_valid)
residuals = x_valid - x_fit

v_uncertainty = np.std(residuals) / np.std(t_valid)

print(f"Estimated wave speed: {v_est:.3f} m/s")
print(f"Estimated uncertainty: ±{v_uncertainty:.3f} m/s")

# %% [Block 12] Distance vs arrival time plot

plt.figure(figsize=(6, 5))

plt.scatter(t_valid, x_valid, s=50, label="Measured arrival times")
plt.plot(t_valid, x_fit, color="red", label=f"Fit: v = {v_est:.3f} m/s")

plt.xlabel("Arrival time (s)")
plt.ylabel("Distance along machine (m)")
plt.title("Wave propagation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()




