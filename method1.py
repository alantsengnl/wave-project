# %% [Block 1] Load video and get first frame

import cv2
import numpy as np

VIDEO_PATH = "wave-long-high-2.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise ValueError(f"Cannot open video: {VIDEO_PATH}")

ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Could not read first frame.")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Video loaded successfully.")
# %% [Block 2] Select two ROIs on the first frame

print("Select ROI 1, Press ENTER, Select ROI 2, Press ENTER")

h, w = first_frame.shape[:2]
max_w, max_h = 1382, 777
scale = min(max_w / w, max_h / h, 1.0)
disp_frame = cv2.resize(first_frame, (int(w * scale), int(h * scale)))

cv2.namedWindow("ROIs", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("ROIs", cv2.WND_PROP_TOPMOST, 1)

roi1_small = cv2.selectROI("ROIs", disp_frame, showCrosshair=True)
roi2_small = cv2.selectROI("ROIs", disp_frame, showCrosshair=True)
cv2.destroyAllWindows()

def unscale_roi(roi_small):
    x_s, y_s, w_s, h_s = roi_small
    return (int(x_s / scale), int(y_s / scale),
            int(w_s / scale), int(h_s / scale))

roi1 = unscale_roi(roi1_small)
roi2 = unscale_roi(roi2_small)

def roi_center(roi):
    x, y, w, h = roi
    return (x + w / 2, y + h / 2)

roi1_center = roi_center(roi1)
roi2_center = roi_center(roi2)
# %% [Block 3] Extract brightness signals and detect arrival times

def smooth(x, k=7):
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")

def first_rise(signal, baseline_frames=10, frac=0.15):
    baseline = float(np.median(signal[:baseline_frames]))
    peak = float(signal.max())
    thresh = baseline + frac * (peak - baseline)

    above = np.where(signal >= thresh)[0]
    if len(above) == 0:
        return int(np.argmax(signal))   # fallback: take global max
    return int(above[0])

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise RuntimeError("Invalid FPS in video.")

int1, int2 = [], []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2

    int1.append(gray[y1:y1+h1, x1:x1+w1].mean())
    int2.append(gray[y2:y2+h2, x2:x2+w2].mean())

cap.release()

int1 = smooth(np.array(int1), k=7)
int2 = smooth(np.array(int2), k=7)

idx0 = first_rise(int1)
idx1 = first_rise(int2)

t0 = idx0 / fps
t1 = idx1 / fps
dt = abs(t1 - t0)

cx1, cy1 = roi1_center
cx2, cy2 = roi2_center
dx_pixels = abs(cx2 - cx1)

pixel_speed = dx_pixels / dt

print(f"ROI1 arrival frame: {idx0}, t0 = {t0:.6f} s")
print(f"ROI2 arrival frame: {idx1}, t1 = {t1:.6f} s")
print(f"Δt = {dt:.6f} s")
print(f"Horizontal distance between ROIs: {dx_pixels:.3f} px")
print(f"Wave speed (pixels/s): {pixel_speed:.3f}")

# Save for Block 5
dt_global = dt
dx_pixels_global = dx_pixels
pixel_speed_global = pixel_speed

# %% [Block 4] Measure scale (meters → pixels)

KNOWN_LENGTH_M = 0.92   

h, w = first_frame.shape[:2]

max_w, max_h = 1382, 777
scale = min(max_w / w, max_h / h, 1.0)
disp_frame = cv2.resize(first_frame, (int(w * scale), int(h * scale)))

print("Select a region spanning the known physical length (0.92 m).")

cv2.namedWindow("Scale measurement", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Scale measurement", cv2.WND_PROP_TOPMOST, 1)
scale_roi_small = cv2.selectROI("Scale measurement", disp_frame, showCrosshair=True)
cv2.destroyAllWindows()

x_s, y_s, w_s, h_s = scale_roi_small

L_pixels = w_s / scale
m_per_pixel = KNOWN_LENGTH_M / L_pixels

print(f"Machine span in pixels: {L_pixels:.2f}")
print(f"Scale: {m_per_pixel:.6e} m/pixel")

m_per_pixel_global = m_per_pixel

# %% [Block 5] Convert pixel speed to meters per second

if 'pixel_speed_global' not in globals() or 'm_per_pixel_global' not in globals():
    raise RuntimeError("Missing inputs: run Blocks 3 and 4 first.")

wave_speed = pixel_speed_global * m_per_pixel_global

print(f"Wave speed: {wave_speed:.4f} m/s")






# %%
