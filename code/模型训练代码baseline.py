"""
YOLO11s Baseline 训练脚本 - 水稻+玉米 19 类
使用 merged_yolo_balanced_ricemaize 数据集
"""

from ultralytics import YOLO
from pathlib import Path
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    DATA_YAML = Path(r"E:\study\病虫害\merged_yolo_balanced_ricemaize\data.yaml")

    if not DATA_YAML.exists():
        print(f"[ERROR] 数据配置不存在: {DATA_YAML}")
        print("请先运行: 1) merge_2crops_yolo.py  2) balance_and_patch_ricemaize.py")
        exit(1)

    print("=" * 70)
    print("训练 YOLO11s Baseline（水稻+玉米 19 类，不含小麦）")
    print("=" * 70)
    print(f"数据: {DATA_YAML}")
    print("=" * 70)

    model = YOLO("yolo11s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=960,
        batch=8,
        device=0,
        workers=6,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.0005,
        mosaic=0.3,
        mixup=0.0,
        scale=0.3,
        patience=50,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        project="runs/detect",
        name="ricemaize_19cls_y11s_baseline",
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("训练完成！最佳模型: runs/detect/ricemaize_19cls_y11s_baseline/weights/best.pt")
    print("=" * 70)
