
import json
import argparse

import numpy as np
import cv2
import onnxruntime as ort
import yaml

PATH = "C:\\Users\\kaiyi\\Desktop\\vision\\vision-assignment\\detection\\assets\\large.mp4"
MODEL = "C:\\Users\\kaiyi\\Desktop\\vision\\vision-assignment\\detection\\assets\\0526.onnx"

IMPUT_SIZE = 640
FPS = 30.0

COLORS   = ["blue", "red", "neutral", "purple"]
PATTERNS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"]


def parse_args():
    p = argparse.ArgumentParser(description="Armor detection + PnP -> JSON Lines")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--video", required=True, help="input video path")
    p.add_argument("--calibration", required=True, help="calibration YAML path")


def main():
    args = parse_args()
#-------------------------------Open video and read a frame--------------------------------
    cap = cv2.VideoCapture(PATH)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    # cv2.imshow("Original Frame", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cap.release()
    if not ret:
        print("Error: Could not read the frame.")
        return

    ##-------------------------------Letterbox the frame--------------------------------
    img, scale, pad_x, pad_y = letterbox(frame)
    print(f"scale={scale:.4f} pad_x={pad_x} pad_y={pad_y}")
    # cv2.imshow("Letterboxed Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    K, dist = load_calibration(args.calibration)       # once, next to the ONNX session
    session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor = to_tensor(img)
    # print("input:", tensor.shape, tensor.dtype)
    raw = session.run(None, {input_name: tensor})[0][0]
    # print("output:", raw.shape, raw.dtype)

    # dets = nms(decode(raw, 0.5), 0.5)
    # for d in dets:
    #     d["corners_px"] = unletterbox(d["corners"], scale, pad_x, pad_y)
    # # print("detections:", dets)


    # for d in dets:
    #     d["corners_px"] = unletterbox(d["corners"], scale, pad_x, pad_y)
    #     d["corners_camera_m"], d["reprojection_error_px"] = solve_pose(d["corners_px"], d["pattern"], K, dist)


    frame_index = 0
    with open(args.output, "w", encoding="utf-8", newline="\n") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = process_frame(frame, session, input_name, K, dist,
                                       args.conf, args.iou)
            record = {
                "frame_index": frame_index,
                "timestamp_s": frame_index / FPS,
                "detections": detections,
            }
            ##-------------------------------Output results--------------------------------
            out.write(json.dumps(record, allow_nan=False) + "\n")
            frame_index += 1
    cap.release()

    if frame_index == 0:
        print("Error: video contains no frames")
    
def process_frame(frame, session, input_name, K, dist, conf_thr=0.5, iou_thr=0.5):
    img, scale,pad_x, pad_y = letterbox(frame)
    raw = session.run(None, {input_name: to_tensor(img)})[0][0] #get the raw output from the model
    dets = nms(decode(raw, conf_thr), iou_thr) #run model

    results = []
    for d in dets:
         corners_px = unletterbox(d["corners"], scale, pad_x, pad_y)
         corners_camera_m, reprojection_error_px = solve_pose(d["corners_px"], d["pattern"], K, dist)
         results.append({
            "color": d["color"],
            "pattern": d["pattern"],
            "confidence": d["confidence"],
            "corners_px": corners_px.astype(np.float32).tolist(), #raw coners coordinates
            "corners_camera_m": corners_camera_m.tolist() if corners_camera_m is not None else None, # coners coordinates in camera frame
            "reprojection_error_px": reprojection_error_px,
         })
    return results

#-------------------------------Helper functions--------------------------------
#-------------------------------Preprocess -------------------------------------
def letterbox(frame, size=640):
    h, w = frame.shape[:2]
    scale = min(size / w, size / h)                  
    nw, nh = round(w * scale), round(h * scale)      
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2  
    canvas = np.full((size, size, 3), 114, dtype=np.uint8) 
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    # cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + nw, pad_y + nh), (0, 255, 0), 2)
    # cv2.imshow("Letterboxed Image", canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return canvas, scale, pad_x, pad_y

def to_tensor(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    x = rgb.astype(np.float32) / 255.0           
    x = x.transpose(2, 0, 1)[None]               # HWC → CHW, add batch: (1, 3, 640, 640)
    return np.ascontiguousarray(x, dtype=np.float16)  
#-------------------------------decoding -------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def decode(raw, conf_thr=0.5):
    conf = sigmoid(raw[:, 8])            # objectness logit -> probability
    keep = conf > conf_thr               # filter first: 25200 rows -> a handful
    dets = []
    for row, c in zip(raw[keep], conf[keep]):
        dets.append({
            "corners":    row[0:8].reshape(4, 2).copy(),   # keep UL, LL, LR, UR order!
            "confidence": float(c),                         # sigmoid(obj) only
            "color":      COLORS[int(np.argmax(row[9:13]))],
            "pattern":    PATTERNS[int(np.argmax(row[13:22]))],
        })
    return dets

def box_of(corners):
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return x1, y1, x2, y2

def iou(a, b): # intersection over union
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def nms(dets, iou_thr=0.5): # non-maximum suppression
    dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in dets:
        b = box_of(d["corners"])
        if all(iou(b, box_of(k["corners"])) <= iou_thr for k in kept):
            kept.append(d)
    return kept

def unletterbox(corners, scale, pad_x, pad_y):
    """corners: (4, 2) in 640×640 model space → (4, 2) in original image pixels"""
    return (corners - np.array([pad_x, pad_y], dtype=np.float32)) / scale

#-------------------------------PnP -------------------------------------
SMALL = (0.135, 0.056)   # width, height in metres
LARGE = (0.230, 0.056)
LARGE_PATTERNS = {"1", "base_big"}

def board_points(pattern):
    w, h = LARGE if pattern in LARGE_PATTERNS else SMALL
    return np.array([[-w/2, -h/2, 0],    # upper-left
                     [-w/2,  h/2, 0],    # lower-left
                     [ w/2,  h/2, 0],    # lower-right
                     [ w/2, -h/2, 0]],   # upper-right   ← same order as the model's corners
                    dtype=np.float64)

def load_calibration(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)      # 3×3 camera matrix
    dist = np.array(data["distort_coeffs"], dtype=np.float64)  # k1, k2, p1, p2, k3
    return K, dist
def solve_pose(corners_px, pattern, K, dist):
    obj = board_points(pattern)
    img = np.asarray(corners_px, dtype=np.float64).reshape(4, 1, 2)
    try:
        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None, None

    best = None
    for rvec, tvec in zip(rvecs, tvecs):          # IPPE gives up to 2 candidates
        R, _ = cv2.Rodrigues(rvec)                # rotation vector -> 3×3 matrix
        cam = (R @ obj.T + tvec.reshape(3, 1)).T  # each physical corner -> camera frame
        if np.any(cam[:, 2] <= 0):                # must be in front of the camera
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = float(np.sqrt(np.sum((proj.reshape(4, 2) - img.reshape(4, 2)) ** 2) / 4))
        if not np.isfinite(err) or not np.all(np.isfinite(cam)):
            continue
        if best is None or err < best[1]:         # keep the lowest reprojection error
            best = (cam, err)
    return best if best else (None, None)


if __name__ == "__main__":
    main()
