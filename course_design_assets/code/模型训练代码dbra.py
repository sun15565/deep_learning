"""
YOLO11s + C2PSA_DBRA 训练脚本 - 水稻+玉米 19 类

在 baseline 上仅将 backbone 末端 C2PSA 替换为 C2PSA_DBRA（可变形双级路由注意力）。
使用同一数据集：merged_yolo_balanced_ricemaize_replaced_rice。
支持从 last.pt 恢复训练（RESUME=1）。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import multiprocessing

from ultralytics import YOLO

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_YAML = PROJECT_ROOT / "merged_yolo_balanced_ricemaize_replaced_rice" / "data.yaml"
MODEL_YAML = PROJECT_ROOT / "yolo11s_dbra_19cls.yaml"

BATCH = int(os.environ.get("BATCH", 8))
WORKERS = int(os.environ.get("WORKERS", 0))
DEVICE = os.environ.get("DEVICE", "0")
IMGSZ = 960
EPOCHS = 150
PATIENCE = 50

RUN_NAME = "ricemaize_19cls_y11s_dbra"
RUN_DIR = PROJECT_ROOT / "runs" / "detect" / RUN_NAME
LAST_PT = RUN_DIR / "weights" / "last.pt"

# 从上次中断的 last.pt 继续训练：设为 True 或环境变量 RESUME=1
RESUME = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")

if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not DATA_YAML.exists():
        print(f"[ERROR] 数据配置不存在: {DATA_YAML}")
        print("请先运行: python replace_rice_with_new_dataset.py 生成新数据集")
        exit(1)

    if RESUME:
        if not LAST_PT.exists():
            print(f"[ERROR] 恢复训练需要存在: {LAST_PT}")
            exit(1)
        model = YOLO(str(LAST_PT))
        print("=" * 70)
        print("从 last.pt 恢复训练（C2PSA_DBRA）")
        print("=" * 70)
        print(f" checkpoint: {LAST_PT}")
    else:
        if not MODEL_YAML.exists():
            print(f"[ERROR] 模型配置不存在: {MODEL_YAML}")
            exit(1)
        model = YOLO(str(MODEL_YAML))
        print("=" * 70)
        print("训练 YOLO11s + C2PSA_DBRA（可变形双级路由注意力，baseline 改进）")
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
        lr0=0.001,
        lrf=0.1,
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
