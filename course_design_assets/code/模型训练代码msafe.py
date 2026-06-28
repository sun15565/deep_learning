"""
YOLO11s + EMCAD 训练脚本 - 水稻+玉米 19 类
EMCAD: Efficient Multi-scale Convolutional Attention Decoding（多尺度 DW + 通道/空间门控注意力）
支持本地与服务器：通过 PROJECT_ROOT 或脚本所在目录自动解析路径；服务器可加大 batch/workers。
"""
import os

# 必须在 import torch/ultralytics 之前设置，避免 Windows 下 OpenMP 重复初始化报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import multiprocessing

from ultralytics import YOLO

# ============== 路径配置（服务器上传项目后无需改 E 盘路径，用脚本所在目录） ==============
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
# 与 Baseline 公平对比：使用同一替换后数据集
DATA_YAML = PROJECT_ROOT / "merged_yolo_balanced_ricemaize_replaced_rice" / "data.yaml"
MODEL_YAML = PROJECT_ROOT / "yolo11s_emcad_19cls.yaml"

# ============== 训练配置：EMCAD 比 Baseline 更耗显存，默认小 batch ==============
# 本地 (约 8GB 显存): batch=4~8, workers=4~6；仍 OOM 可设 BATCH=2 或 IMGSZ=640
# 服务器 (显存 12GB+): batch=16, workers=12
BATCH = int(os.environ.get("BATCH", 20))         # 默认 8 避免 OOM，显存充足再加大
WORKERS = int(os.environ.get("WORKERS", 12))     # 默认 0 避免 Windows DataLoader 多进程崩溃；Linux 可设 6~12
DEVICE = os.environ.get("DEVICE", "0")         # 多卡可设 "0,1"
IMGSZ = 960
EPOCHS = 150
PATIENCE = 50

RUN_NAME = "ricemaize_19cls_y11s_emcad"
RUN_DIR = PROJECT_ROOT / "runs" / "detect" / RUN_NAME
LAST_PT = RUN_DIR / "weights" / "last.pt"

# 从上次中断的 last.pt 继续训练：设为 True 或环境变量 RESUME=1
RESUME = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not DATA_YAML.exists():
        print(f"[ERROR] 数据配置不存在: {DATA_YAML}")
        print("请先运行: python replace_rice_with_new_dataset.py 生成替换后数据集")
        exit(1)

    # 恢复训练：必须存在 last.pt
    if RESUME:
        if not LAST_PT.exists():
            print(f"[ERROR] 恢复训练需要存在: {LAST_PT}")
            exit(1)
        model = YOLO(str(LAST_PT))
        print("=" * 70)
        print("从 last.pt 恢复训练（EMCAD）")
        print("=" * 70)
        print(f" checkpoint: {LAST_PT}")
    else:
        if not MODEL_YAML.exists():
            print(f"[ERROR] 模型配置不存在: {MODEL_YAML}")
            exit(1)
        model = YOLO(str(MODEL_YAML))
        print("=" * 70)
        print("训练 YOLO11s + EMCAD（水稻+玉米 19 类）")
        print("EMCAD: 多尺度深度卷积 + 通道/空间门控注意力")
        print("=" * 70)

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据: {DATA_YAML}")
    print(f"batch={BATCH}, workers={WORKERS}, device={DEVICE}, imgsz={IMGSZ}")
    print("=" * 70)
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        resume=RESUME,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        optimizer="AdamW",
        lr0=0.0005,  # EMCAD 参数更多，降低初始学习率避免训练不稳定
        lrf=0.1,
        warmup_epochs=5.0,  # 增加 warmup 让 EMCAD 模块逐步激活
        weight_decay=0.0005,
        mosaic=0.3,
        mixup=0.0,
        scale=0.3,
        patience=PATIENCE,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("训练完成！最佳模型:", RUN_DIR / "weights" / "best.pt")
    print("=" * 70)
