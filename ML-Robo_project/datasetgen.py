

import sys
import os
import random
import time
from pathlib import Path

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    sys.exit("[ERROR] pybullet not found.  Run:  pip install pybullet")

try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy not found.    Run:  pip install numpy")

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] opencv not found.   Run:  pip install opencv-python")

try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] pandas not found.   Run:  pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")          # ← non-interactive backend, safe for VS Code
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("[ERROR] matplotlib not found. Run:  pip install matplotlib")




IMG_W, IMG_H   = 224, 224      # final image size for CNN input
CAM_WIDTH      = 320          
CAM_HEIGHT     = 240           #
FOV            = 60            # camera field-of-view   (degrees)
STEPS_PER_RUN  = 40          # max simulation steps per run
N_RUNS         = 10            # number of traversal runs

# Dataset folder — created next to this .py file
SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "robot_dataset"

NOISE_TYPES = ["clean", "gaussian_noise", "gaussian_blur", "both"]

ACTIONS = {
    "forward" : 0,
    "left"    : 1,
    "right"   : 2,
    "stop"    : 3,
}

# Waypoints (x, y) the virtual camera follows through the maze
WAYPOINTS = [
    (-4.0, -4.0),
    (-4.0,  0.0),
    (-4.0,  4.0),
    ( 0.0,  4.0),
    ( 0.0,  0.0),
    ( 0.0, -4.0),
    ( 4.0, -4.0),
    ( 4.0,  0.0),
    ( 4.0,  4.0),
    ( 0.0,  4.5),
]




def create_folders():
    for noise in NOISE_TYPES:
        (DATASET_DIR / noise / "images").mkdir(parents=True, exist_ok=True)
    print("[OK] Output folders created under:", DATASET_DIR)




def start_pybullet():
    """Connect to PyBullet in headless (DIRECT) mode."""
    client = p.connect(p.DIRECT)         
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    print("[OK] PyBullet started (DIRECT / headless mode)")
    return client


def load_ground():
    p.loadURDF("plane.urdf")


def build_wall(pos, size, color=(0.4, 0.4, 0.4, 1)):
    """Spawn a static box wall in the simulation."""
    col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )


def build_maze():
    """Build the 3-D maze from wall segments."""
    WALL_H = 1.0    # height 
    WALL_T = 0.15   # thickness

    # Outer boundary walls
    boundary = [
        ([0,   5,  WALL_H], [5,      WALL_T, WALL_H]),   # North
        ([0,  -5,  WALL_H], [5,      WALL_T, WALL_H]),   # South
        ([5,   0,  WALL_H], [WALL_T, 5,      WALL_H]),   # East
        ([-5,  0,  WALL_H], [WALL_T, 5,      WALL_H]),   # West
    ]
    for pos, size in boundary:
        build_wall(pos, size, color=(0.2, 0.2, 0.8, 1))   # blue border

    # Inner maze walls
    inner = [
        ([ 0,    2,  WALL_H], [2.5,    WALL_T, WALL_H]),
        ([-2,    0,  WALL_H], [1.5,    WALL_T, WALL_H]),
        ([ 2,   -2,  WALL_H], [1.5,    WALL_T, WALL_H]),
        ([ 0,   -3,  WALL_H], [2.0,    WALL_T, WALL_H]),
        ([ 2,    3,  WALL_H], [WALL_T, 2.0,    WALL_H]),
        ([-3,    1,  WALL_H], [WALL_T, 1.5,    WALL_H]),
        ([ 0,   -1,  WALL_H], [WALL_T, 1.0,    WALL_H]),
        ([ 3,    0,  WALL_H], [WALL_T, 2.5,    WALL_H]),
        ([-1,   -3,  WALL_H], [WALL_T, 1.5,    WALL_H]),
        ([-3,   -2,  WALL_H], [1.0,    WALL_T, WALL_H]),
        ([ 1,    4,  WALL_H], [1.0,    WALL_T, WALL_H]),
        ([-4,    3,  WALL_H], [WALL_T, 1.0,    WALL_H]),
        ([ 4,   -3,  WALL_H], [WALL_T, 1.5,    WALL_H]),
    ]
    colors = [
        (0.7, 0.3, 0.2, 1),   # reddish
        (0.2, 0.6, 0.3, 1),   # green
        (0.6, 0.5, 0.1, 1),   # yellow-brown
    ]
    for i, (pos, size) in enumerate(inner):
        build_wall(pos, size, color=colors[i % len(colors)])

    print("[OK] Maze built  (4 boundary walls + 13 inner walls)")




def get_camera_image(cam_pos, yaw_deg):
    
    yaw_rad = np.deg2rad(yaw_deg)
    target  = [
        cam_pos[0] + np.cos(yaw_rad),
        cam_pos[1] + np.sin(yaw_rad),
        cam_pos[2],
    ]
    view_mat = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
    proj_mat = p.computeProjectionMatrixFOV(
        fov=FOV, aspect=CAM_WIDTH / CAM_HEIGHT, nearVal=0.1, farVal=100.0
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=CAM_WIDTH, height=CAM_HEIGHT,
        viewMatrix=view_mat, projectionMatrix=proj_mat,
        renderer=p.ER_TINY_RENDERER,
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH, 4)[:, :, :3]
    return rgb


def preprocess(img):
   
    return cv2.resize(img, (IMG_W, IMG_H))


def add_gaussian_noise(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_gaussian_blur(img, ksize=15):
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_degradation(img, noise_type):
    if noise_type == "clean":
        return img.copy()
    elif noise_type == "gaussian_noise":
        return add_gaussian_noise(img)
    elif noise_type == "gaussian_blur":
        return add_gaussian_blur(img)
    elif noise_type == "both":
        return add_gaussian_blur(add_gaussian_noise(img))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")




def get_action(cam_pos, cam_yaw, target_wp):
    """
    Decide action based on angle to next waypoint.
    Returns (action_str, new_cam_pos, new_cam_yaw).
    """
    dx   = target_wp[0] - cam_pos[0]
    dy   = target_wp[1] - cam_pos[1]
    dist = np.hypot(dx, dy)

    if dist < 0.3:
        return "stop", cam_pos, cam_yaw

    desired_yaw = np.degrees(np.arctan2(dy, dx))
    diff        = (desired_yaw - cam_yaw + 180) % 360 - 180   # [-180, 180]

    STEP = 0.12   # metres per simulation step

    if abs(diff) > 20:
        action  = "left" if diff > 0 else "right"
        new_yaw = cam_yaw + np.sign(diff) * 10
        new_pos = cam_pos
    else:
        action  = "forward"
        new_yaw = cam_yaw
        yaw_rad = np.deg2rad(cam_yaw)
        new_pos = [
            cam_pos[0] + STEP * np.cos(yaw_rad),
            cam_pos[1] + STEP * np.sin(yaw_rad),
            cam_pos[2],
        ]

    return action, new_pos, new_yaw




def collect_dataset():
    records     = []
    img_counter = 0
    t_start     = time.time()

    client = start_pybullet()
    load_ground()
    build_maze()

    print(f"\n[INFO] Starting {N_RUNS} collection runs …\n")

    for run in range(N_RUNS):
        # Slight random start for dataset diversity
        cam_pos = [-4.0 + random.uniform(-0.3, 0.3),
                   -4.0 + random.uniform(-0.3, 0.3),
                   0.6]                               # camera height 0.6 m
        cam_yaw = random.uniform(-10, 10)
        wp_idx  = 0
        step    = 0

        while step < STEPS_PER_RUN and wp_idx < len(WAYPOINTS):
            target_wp                    = WAYPOINTS[wp_idx]
            action, new_pos, new_yaw     = get_action(cam_pos, cam_yaw, target_wp)

            if action == "stop":
                wp_idx += 1
                continue

            # Capture + preprocess one clean frame
            clean_img = preprocess(get_camera_image(cam_pos, cam_yaw))

            # Save all 4 degradation variants
            for noise_type in NOISE_TYPES:
                degraded = apply_degradation(clean_img, noise_type)
                fname    = f"img_{img_counter:06d}.png"
                fpath    = DATASET_DIR / noise_type / "images" / fname
                cv2.imwrite(str(fpath), cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR))

                records.append({
                    "filename"   : fname,
                    "noise_type" : noise_type,
                    "action"     : action,
                    "label"      : ACTIONS[action],
                    "run"        : run,
                    "step"       : step,
                    "cam_x"      : round(cam_pos[0], 3),
                    "cam_y"      : round(cam_pos[1], 3),
                    "cam_yaw"    : round(cam_yaw, 2),
                })

            cam_pos      = new_pos
            cam_yaw      = new_yaw
            img_counter += 1
            step        += 1

            p.stepSimulation()

        elapsed = time.time() - t_start
        print(f"  Run {run+1:>2}/{N_RUNS}  |  frames this run: {step:>4}"
              f"  |  total frames: {img_counter:>5}  |  elapsed: {elapsed:.1f}s")

    p.disconnect()
    total_imgs = img_counter * len(NOISE_TYPES)
    print(f"\n[OK] Collection complete.")
    print(f"     Unique frames : {img_counter}")
    print(f"     Total images  : {total_imgs}  (×{len(NOISE_TYPES)} variants)")
    return records




def save_labels(records):
    df = pd.DataFrame(records)

    for noise_type in NOISE_TYPES:
        subset   = df[df["noise_type"] == noise_type][["filename", "action", "label"]]
        csv_path = DATASET_DIR / noise_type / "labels.csv"
        subset.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}  ({len(subset)} rows)")

    master = DATASET_DIR / "all_labels.csv"
    df.to_csv(master, index=False)
    print(f"  Master CSV: {master}  ({len(df)} rows)")

    print("\nSample rows:")
    print(df.sample(min(6, len(df))).to_string(index=False))
    return df




def save_stats_plot(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    df_clean      = df[df["noise_type"] == "clean"]
    action_counts = df_clean["action"].value_counts()
    axes[0].bar(action_counts.index, action_counts.values,
                color=["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"])
    axes[0].set_title("Action Label Distribution")
    axes[0].set_xlabel("Action")
    axes[0].set_ylabel("Count")

    noise_counts = df.groupby("noise_type").size()
    axes[1].bar(noise_counts.index, noise_counts.values,
                color=["#264653", "#2A9D8F", "#E9C46A", "#E76F51"])
    axes[1].set_title("Images per Degradation Type")
    axes[1].set_xlabel("Noise Type")
    axes[1].set_ylabel("Count")

    plt.suptitle("Dataset Statistics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = DATASET_DIR / "dataset_stats.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def save_degradation_comparison():
    img_folder = DATASET_DIR / "clean" / "images"
    sample_path = next(img_folder.iterdir(), None)
    if sample_path is None:
        print("[WARN] No clean images found — skipping comparison plot.")
        return

    base = cv2.cvtColor(cv2.imread(str(sample_path)), cv2.COLOR_BGR2RGB)
    variants = {
        "Clean"          : base,
        "Gaussian Noise" : add_gaussian_noise(base),
        "Gaussian Blur"  : add_gaussian_blur(base),
        "Noise + Blur"   : add_gaussian_blur(add_gaussian_noise(base)),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img) in zip(axes, variants.items()):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("Vision Degradation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = DATASET_DIR / "degradation_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def save_intensity_histograms():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    colors    = ["#4C9BE8", "#F4A261", "#2A9D8F", "#E76F51"]

    for ax, noise_type, clr in zip(axes, NOISE_TYPES, colors):
        img_folder = DATASET_DIR / noise_type / "images"
        img_files  = sorted(img_folder.iterdir())[:50]
        pixels     = []
        for fp in img_files:
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                pixels.extend(img.flatten().tolist())
        ax.hist(pixels, bins=64, color=clr, alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.set_title(noise_type.replace("_", " ").title(), fontsize=9)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")

    fig.suptitle("Pixel Intensity Distribution per Noise Type",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = DATASET_DIR / "intensity_histograms.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")



def main():
    print("=" * 60)
    print("  VISION IMPAIRED ROBOT — Dataset Generator")
    print("  Authors: Abeer Azeem (22L-7781) | Mehreen Imran (22L-7668)")
    print("=" * 60)

    # 1. Folders
    create_folders()

    # 2. Simulate  collect images
    records = collect_dataset()

    # 3. Labels CSV
    print("\n[INFO] Saving label CSV files …")
    df = save_labels(records)

    # 4. Plots
    print("\n[INFO] Generating analysis plots …")
    save_stats_plot(df)
    save_degradation_comparison()
    save_intensity_histograms()

    # 5. Final summary
    print("\n" + "=" * 60)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Output folder  : {DATASET_DIR}")
    print(f"  Total images   : {len(df)}")
    print(f"  Image size     : {IMG_W} x {IMG_H} px")
    print(f"  Noise variants : {NOISE_TYPES}")
    print(f"  Action classes : {list(ACTIONS.keys())}")
    print()
    print("  Files created:")
    print("    robot_dataset/")
    print("    ├── all_labels.csv")
    print("    ├── dataset_stats.png")
    print("    ├── degradation_comparison.png")
    print("    ├── intensity_histograms.png")
    for n in NOISE_TYPES:
        print(f"    ├── {n}/")
        print(f"    │   ├── labels.csv")
        print(f"    │   └── images/   ← PNG frames")
    print()
    print("  Next: train your CNN on these images + label CSVs!")
    print("=" * 60)


if __name__ == "__main__":
    main()
