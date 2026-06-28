"""
YOLO11s 改进版训练 - 参考 6 条亲测有效思路，适合小目标/遥感/密集场景。

1. 多尺度特征金字塔：增加 P2 小目标分支，4 尺度检测头（P2/P3/P4/P5）
2. 注意力：P2/P3/P4/P5 均用 GatedAttnBlock（门控+CBAM+残差）
3. 数据增强：Mosaic + Copy-Paste，scale 加强小目标
4. 损失/标签分配：沿用默认（可后续接 Focal、SimOTA 调参）

直接放服务器跑：改 BATCH/WORKERS 即可。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import multiprocessing

from ultralytics import YOLO

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent))
DATA_YAML = PROJECT_ROOT / "merged_yolo_balanced_ricemaize_replaced_rice" / "data.yaml"
MODEL_YAML = PROJECT_ROOT / "yolo11s_improved_19cls.yaml"

# 服务器可加大；本地 8GB 显存建议 8
BATCH = int(os.environ.get("BATCH", 20))
WORKERS = int(os.environ.get("WORKERS", 12))
DEVICE = os.environ.get("DEVICE", "0")
IMGSZ = 960
EPOCHS = 150
PATIENCE = 50

# 数据增强：小目标友好（参考思路 4）
MOSAIC = float(os.environ.get("MOSAIC", 0.5))       # 提高 Mosaic 概率
COPY_PASTE = float(os.environ.get("COPY_PASTE", 0.2))  # Copy-Paste
SCALE = float(os.environ.get("SCALE", 0.5))         # 尺度增强范围，小目标多缩放

# 从 last.pt 继续训练：RESUME=1 或环境变量 RESUME=1
RESUME = os.environ.get("RESUME", "").lower() in ("1", "true", "yes")
RUN_DIR = PROJECT_ROOT / "runs" / "detect" / "ricemaize_19cls_y11s_improved"
LAST_PT = RUN_DIR / "weights" / "last.pt"

if __name__ == "__main__":
    multiprocessing.freeze_support()

    if not DATA_YAML.exists():
        print(f"[ERROR] 数据不存在: {DATA_YAML}")
        exit(1)

    if RESUME:
        if not LAST_PT.exists():
            print(f"[ERROR] 恢复训练需要存在: {LAST_PT}")
            exit(1)
        model = YOLO(str(LAST_PT))
        print("=" * 70)
        print("从 last.pt 恢复训练（YOLO11s 改进版）")
        print("=" * 70)
        print(f" checkpoint: {LAST_PT}")
    else:
        if not MODEL_YAML.exists():
            print(f"[ERROR] 模型配置不存在: {MODEL_YAML}")
            exit(1)
        model = YOLO(str(MODEL_YAML))
        print("=" * 70)
        print("YOLO11s 改进版（P2 四尺度 + 门控注意力 + Mosaic/CopyPaste）")
        print("=" * 70)

    print(f"数据: {DATA_YAML}")
    print(f"batch={BATCH}, workers={WORKERS}, mosaic={MOSAIC}, copy_paste={COPY_PASTE}, scale={SCALE}")
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
        mosaic=MOSAIC,
        copy_paste=COPY_PASTE,
        scale=SCALE,
        mixup=0.1,
        patience=PATIENCE,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        amp=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="ricemaize_19cls_y11s_improved",
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("训练完成！最佳模型:", PROJECT_ROOT / "runs" / "detect" / "ricemaize_19cls_y11s_improved" / "weights" / "best.pt")
    print("=" * 70)
