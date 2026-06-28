import json
import csv
import math
import sqlite3
import sys
import time
import uuid
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, session, url_for
try:
    import cv2
except ImportError:
    cv2 = None
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
from werkzeug.security import check_password_hash, generate_password_hash

# 优先使用项目内的 ultralytics（包含自定义模块，如 emcad）
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


BASE_DIR = CURRENT_DIR
MODEL_DIR = BASE_DIR / "model"
IMAGE_DIR = BASE_DIR / "image"
UPLOAD_DIR = BASE_DIR / "uploads"
AVATAR_DIR = UPLOAD_DIR / "avatars"
RESULT_DIR = BASE_DIR / "results"
DB_PATH = BASE_DIR / "app.db"
UAV_SOURCE_DIR = PROJECT_ROOT / "uav_farmland_output_v3"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAME_ZH = {
    "Crack-Detection": "裂缝",
    "Exposed Rebar": "钢筋外露",
    "Spalling": "剥落",
    "Break": "破损",
    "Efflorescence": "白华",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = "agriscan-dev-secret-key"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
if CORS is not None:
    CORS(app)

_MODEL_CACHE: Dict[str, YOLO] = {}


def _init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                avatar_url TEXT,
                failed_login_attempts INTEGER NOT NULL DEFAULT 0,
                lock_until INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ts INTEGER NOT NULL,
                model_id TEXT,
                model_label TEXT,
                total INTEGER,
                inference_ms REAL,
                stats_json TEXT,
                original_url TEXT,
                result_url TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drone_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ts INTEGER NOT NULL,
                plan_name TEXT,
                crop_type TEXT,
                disease_type TEXT,
                risk_score REAL,
                risk_level TEXT,
                field_area_mu REAL,
                flight_distance_m REAL,
                spray_volume_l REAL,
                plan_json TEXT
            )
            """
        )
        cols = conn.execute("PRAGMA table_info(history)").fetchall()
        col_names = {c[1] for c in cols}
        if "user_id" not in col_names:
            conn.execute("ALTER TABLE history ADD COLUMN user_id INTEGER")
        user_cols = conn.execute("PRAGMA table_info(users)").fetchall()
        user_col_names = {c[1] for c in user_cols}
        if "avatar_url" not in user_col_names:
            conn.execute("ALTER TABLE users ADD COLUMN avatar_url TEXT")
        if "failed_login_attempts" not in user_col_names:
            conn.execute("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER NOT NULL DEFAULT 0")
        if "lock_until" not in user_col_names:
            conn.execute("ALTER TABLE users ADD COLUMN lock_until INTEGER NOT NULL DEFAULT 0")
        conn.commit()


def _scan_models() -> Dict[str, Dict[str, str]]:
    models: Dict[str, Dict[str, str]] = {}
    if not MODEL_DIR.exists():
        return models

    for model_root in MODEL_DIR.iterdir():
        if not model_root.is_dir():
            continue
        weight = model_root / "weights" / "best.pt"
        if weight.exists():
            model_id = model_root.name
            models[model_id] = {
                "model_id": model_id,
                "model_label": model_id.replace("_", " "),
                "weight_path": str(weight),
            }
    return models


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_model_results_csv(model_id: str) -> Dict[str, Any]:
    model_root = MODEL_DIR / model_id
    csv_path = model_root / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到训练结果文件: {csv_path}")

    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {k.strip(): v for k, v in row.items() if k}
            rows.append(normalized)

    if not rows:
        raise ValueError("results.csv 内容为空")

    def getv(row: Dict[str, Any], key: str) -> float:
        return _to_float(row.get(key, "0"))

    last = rows[-1]
    epochs = list(range(1, len(rows) + 1))
    map50_curve = [getv(r, "metrics/mAP50(B)") for r in rows]
    map50_mask_curve = [getv(r, "metrics/mAP50(M)") for r in rows]
    train_box_loss = [getv(r, "train/box_loss") for r in rows]
    train_seg_loss = [getv(r, "train/seg_loss") for r in rows]
    val_box_loss = [getv(r, "val/box_loss") for r in rows]
    val_seg_loss = [getv(r, "val/seg_loss") for r in rows]

    return {
        "model_id": model_id,
        "epochs": len(rows),
        "best": {
            "box_p": round(getv(last, "metrics/precision(B)"), 4),
            "box_r": round(getv(last, "metrics/recall(B)"), 4),
            "box_map50": round(getv(last, "metrics/mAP50(B)"), 4),
            "box_map50_95": round(getv(last, "metrics/mAP50-95(B)"), 4),
            "mask_p": round(getv(last, "metrics/precision(M)"), 4),
            "mask_r": round(getv(last, "metrics/recall(M)"), 4),
            "mask_map50": round(getv(last, "metrics/mAP50(M)"), 4),
            "mask_map50_95": round(getv(last, "metrics/mAP50-95(M)"), 4),
        },
        "curves": {
            "epochs": epochs,
            "box_map50": map50_curve,
            "mask_map50": map50_mask_curve,
            "train_box_loss": train_box_loss,
            "train_seg_loss": train_seg_loss,
            "val_box_loss": val_box_loss,
            "val_seg_loss": val_seg_loss,
        },
    }


def _current_user_id() -> Optional[int]:
    uid = session.get("user_id")
    return int(uid) if uid is not None else None


def _require_login_api() -> Optional[int]:
    uid = _current_user_id()
    return uid


def _get_model(model_id: str) -> YOLO:
    if YOLO is None:
        raise RuntimeError("当前 Python 环境未安装 ultralytics，无法执行病虫害识别。请先安装 requirements.txt。")
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]
    models = _scan_models()
    if model_id not in models:
        raise ValueError(f"模型不存在: {model_id}")
    yolo_model = YOLO(models[model_id]["weight_path"])
    _MODEL_CACHE[model_id] = yolo_model
    return yolo_model


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_SUFFIX


def _run_detection_on_image(
    image_path: Path,
    model_id: str,
    conf: float,
    iou: float,
) -> Dict[str, Any]:
    if cv2 is None:
        raise RuntimeError("当前 Python 环境未安装 opencv-python，无法保存识别结果图。请先安装 requirements.txt。")
    model = _get_model(model_id)
    start = time.perf_counter()
    result = model.predict(str(image_path), conf=conf, iou=iou, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    detections: List[Dict[str, Any]] = []
    stats: Dict[str, int] = {}
    names = model.names

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            conf_score = float(box.conf.item())
            xyxy = box.xyxy[0].tolist()
            cls_name_en = names.get(cls_idx, str(cls_idx))
            cls_name_zh = CLASS_NAME_ZH.get(cls_name_en, cls_name_en)
            stats[cls_name_zh] = stats.get(cls_name_zh, 0) + 1
            detections.append(
                {
                    "class_id": cls_idx,
                    "class_name_en": cls_name_en,
                    "class_name_zh": cls_name_zh,
                    "confidence": round(conf_score, 4),
                    "bbox_xyxy": [round(v, 2) for v in xyxy],
                }
            )

    plotted = result.plot()
    result_name = f"{image_path.stem}_result.jpg"
    result_path = RESULT_DIR / result_name
    cv2.imwrite(str(result_path), plotted)

    return {
        "detections": detections,
        "stats": stats,
        "total": len(detections),
        "inference_ms": round(inference_ms, 2),
        "result_name": result_name,
    }


def _num(data: Dict[str, Any], key: str, default: float) -> float:
    try:
        return float(data.get(key, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _excel_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch.upper()) - ord("A") + 1
    return idx - 1


def _xlsx_text(cell: ET.Element, shared_strings: List[str], ns: Dict[str, str]) -> Any:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("x:v", ns)
    inline_node = cell.find("x:is/x:t", ns)
    if inline_node is not None:
        return inline_node.text or ""
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return raw == "1"
    try:
        num = float(raw)
        return int(num) if num.is_integer() else num
    except ValueError:
        return raw


def _read_xlsx_sheets(xlsx_path: Path) -> Dict[str, List[List[Any]]]:
    ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//x:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"].lstrip("/")
            for rel in rels.findall("rel:Relationship", ns)
        }

        sheets: Dict[str, List[List[Any]]] = {}
        for sheet in workbook.findall("x:sheets/x:sheet", ns):
            title = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{ns['r']}}}id"]
            target = rel_map[rel_id]
            sheet_path = f"xl/{target}" if not target.startswith("xl/") else target
            root = ET.fromstring(zf.read(sheet_path))
            rows: List[List[Any]] = []
            for row in root.findall(".//x:sheetData/x:row", ns):
                values: List[Any] = []
                for cell in row.findall("x:c", ns):
                    col_idx = _excel_col_index(cell.attrib.get("r", "A1"))
                    while len(values) < col_idx:
                        values.append(None)
                    values.append(_xlsx_text(cell, shared_strings, ns))
                rows.append(values)
            sheets[title] = rows
        return sheets


def _records(rows: List[List[Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    headers = [str(h) for h in rows[0]]
    records = []
    for row in rows[1:]:
        item = {headers[i]: row[i] if i < len(row) else None for i in range(len(headers))}
        if any(v is not None for v in item.values()):
            records.append(item)
    return records


def _load_uav_reference_plan() -> Dict[str, Any]:
    xlsx_files = list(UAV_SOURCE_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"未找到无人机决策结果表: {UAV_SOURCE_DIR}")
    sheets = _read_xlsx_sheets(xlsx_files[0])
    summary = {str(row[0]): row[1] for row in sheets.get("总体指标", [])[1:] if row}
    parcels = _records(sheets.get("地块风险", []))
    tasks = _records(sheets.get("病害任务点", []))
    routes = _records(sheets.get("无人机路径", []))
    spray = _records(sheets.get("喷洒路径", []))
    image_files = [p for p in UAV_SOURCE_DIR.glob("*.png")]

    return {
        "plan_name": "不规则农田无人机变量喷洒决策方案",
        "source_file": xlsx_files[0].name,
        "method": {
            "farmland_model": "不规则农田地块空间建模",
            "risk_model": "R_total = f(R_meteo, R_history, R_vision)，融合气象风险、历史风险与视觉识别风险",
            "task_model": "按病害核心面积、地块风险与识别置信度形成 priority，并生成 dose_L_per_ha 与 core_dose_L",
            "route_model": "将 24 个病害任务点分配给 3 架无人机，输出 task_order、route_length_m 和触发式变量喷洒路径",
        },
        "summary": summary,
        "parcels": parcels,
        "tasks": tasks,
        "routes": routes,
        "spray_paths": spray,
        "images": [
            {"name": p.name, "url": url_for("uav_assets_file", path=p.name)}
            for p in sorted(image_files, key=lambda item: item.name)
        ],
    }


def _risk_level(score: float) -> str:
    if score >= 75:
        return "重度防治"
    if score >= 55:
        return "中度防治"
    if score >= 35:
        return "轻度干预"
    return "巡检观察"


def _compute_drone_plan(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Field grid + drone operation decision model for pest-control flights."""
    field_length = _clamp(_num(payload, "field_length_m", 120), 20, 2000)
    field_width = _clamp(_num(payload, "field_width_m", 80), 20, 2000)
    grid_rows = int(_clamp(_num(payload, "grid_rows", 8), 2, 30))
    grid_cols = int(_clamp(_num(payload, "grid_cols", 10), 2, 30))
    detection_count = _clamp(_num(payload, "detection_count", 12), 0, 5000)
    infected_ratio = _clamp(_num(payload, "infected_ratio", 18), 0, 100)
    avg_confidence = _clamp(_num(payload, "avg_confidence", 82), 0, 100)
    humidity = _clamp(_num(payload, "humidity", 72), 0, 100)
    temperature = _clamp(_num(payload, "temperature", 26), -10, 50)
    wind_speed = _clamp(_num(payload, "wind_speed", 2.5), 0, 20)
    swath_width = _clamp(_num(payload, "swath_width_m", 5), 1, 20)
    tank_capacity = _clamp(_num(payload, "tank_capacity_l", 16), 1, 80)
    battery_minutes = _clamp(_num(payload, "battery_minutes", 22), 5, 90)

    crop_type = str(payload.get("crop_type") or "水稻").strip()
    disease_type = str(payload.get("disease_type") or "病虫害综合").strip()
    growth_stage = str(payload.get("growth_stage") or "拔节/孕穗期").strip()
    plan_name = str(payload.get("plan_name") or f"{crop_type}{disease_type}无人机防治方案").strip()

    stage_weight = {
        "苗期": 0.88,
        "分蘖期": 0.96,
        "拔节/孕穗期": 1.12,
        "抽穗/灌浆期": 1.18,
        "成熟期": 0.82,
    }.get(growth_stage, 1.0)
    temp_risk = 100 - min(abs(temperature - 27) * 7, 55)
    weather_score = humidity * 0.55 + temp_risk * 0.35 + max(0, 10 - wind_speed) * 1.0
    vision_score = infected_ratio * 0.72 + min(detection_count / (grid_rows * grid_cols) * 24, 24) + avg_confidence * 0.18
    wind_penalty = 18 if wind_speed >= 6 else 8 if wind_speed >= 4 else 0
    risk_score = _clamp((vision_score * 0.68 + weather_score * 0.32) * stage_weight - wind_penalty, 0, 100)
    risk_level = _risk_level(risk_score)

    if risk_score >= 75:
        spray_rate_l_ha = 30
        flight_speed_m_s = 3.2
        altitude_m = 2.4
    elif risk_score >= 55:
        spray_rate_l_ha = 24
        flight_speed_m_s = 3.8
        altitude_m = 2.8
    elif risk_score >= 35:
        spray_rate_l_ha = 18
        flight_speed_m_s = 4.5
        altitude_m = 3.2
    else:
        spray_rate_l_ha = 10
        flight_speed_m_s = 5.2
        altitude_m = 3.6
    if wind_speed >= 4:
        altitude_m = max(2.0, altitude_m - 0.3)
        flight_speed_m_s = max(2.6, flight_speed_m_s - 0.4)

    field_area_m2 = field_length * field_width
    field_area_mu = field_area_m2 / 666.6667
    field_area_ha = field_area_m2 / 10000
    spray_volume_l = field_area_ha * spray_rate_l_ha
    refill_count = max(0, math.ceil(spray_volume_l / tank_capacity) - 1)

    lane_count = max(1, math.ceil(field_width / swath_width))
    waypoints = []
    for i in range(lane_count):
        y = min(field_width, i * swath_width + swath_width / 2)
        if i % 2 == 0:
            waypoints.extend([{"x": 0, "y": round(y, 2)}, {"x": round(field_length, 2), "y": round(y, 2)}])
        else:
            waypoints.extend([{"x": round(field_length, 2), "y": round(y, 2)}, {"x": 0, "y": round(y, 2)}])
    flight_distance_m = lane_count * field_length + max(0, lane_count - 1) * swath_width
    flight_minutes = flight_distance_m / flight_speed_m_s / 60
    battery_sorties = max(1, math.ceil(flight_minutes / (battery_minutes * 0.82)))

    cells = []
    center_r = (grid_rows - 1) / 2
    center_c = (grid_cols - 1) / 2
    for r in range(grid_rows):
        row = []
        for c in range(grid_cols):
            dist = abs(r - center_r) / max(center_r, 1) * 0.55 + abs(c - center_c) / max(center_c, 1) * 0.45
            local = _clamp(risk_score + (0.52 - dist) * 28 + ((r * 7 + c * 11) % 9 - 4), 0, 100)
            row.append({"row": r + 1, "col": c + 1, "risk": round(local, 1), "level": _risk_level(local)})
        cells.append(row)

    if risk_score >= 55 and wind_speed < 6:
        action = "执行变量喷洒作业，优先覆盖高风险网格，完成后 24-48 小时复查。"
    elif risk_score >= 35:
        action = "执行低剂量定点干预，保留边界缓冲区并加强复查。"
    elif wind_speed >= 6:
        action = "风速偏高，暂停喷洒，仅生成航线并等待适航窗口。"
    else:
        action = "暂不喷洒，执行低空巡田建模和病虫害复核。"

    return {
        "plan_name": plan_name,
        "crop_type": crop_type,
        "disease_type": disease_type,
        "growth_stage": growth_stage,
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "action": action,
        "field": {
            "length_m": round(field_length, 2),
            "width_m": round(field_width, 2),
            "area_mu": round(field_area_mu, 2),
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "cells": cells,
        },
        "operation": {
            "altitude_m": round(altitude_m, 1),
            "flight_speed_m_s": round(flight_speed_m_s, 1),
            "swath_width_m": round(swath_width, 1),
            "spray_rate_l_ha": spray_rate_l_ha,
            "spray_volume_l": round(spray_volume_l, 2),
            "refill_count": refill_count,
            "battery_sorties": battery_sorties,
            "flight_distance_m": round(flight_distance_m, 1),
            "flight_minutes": round(flight_minutes, 1),
            "waypoints": waypoints,
        },
        "inputs": payload,
    }


@app.route("/")
def index():
    if not _current_user_id():
        return redirect(url_for("login_page"))
    return render_template("index.html")


@app.get("/login")
def login_page():
    if _current_user_id():
        return redirect(url_for("index"))
    return render_template("login.html")


@app.get("/register")
def register_page():
    if _current_user_id():
        return redirect(url_for("index"))
    return render_template("register.html")


@app.get("/profile")
def profile_page():
    if not _current_user_id():
        return redirect(url_for("login_page"))
    return render_template("profile.html")


@app.get("/model-info")
def model_info_page():
    if not _current_user_id():
        return redirect(url_for("login_page"))
    return render_template("model_info.html")


@app.get("/drone-decision")
def drone_decision_page():
    if not _current_user_id():
        return redirect(url_for("login_page"))
    return render_template("drone_decision.html")


@app.get("/reset-password")
def reset_password_page():
    if _current_user_id():
        return redirect(url_for("index"))
    return render_template("reset_password.html")


@app.get("/api/me")
def api_me():
    uid = _current_user_id()
    if not uid:
        return jsonify({"ok": False, "message": "未登录"}), 401
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, created_at, avatar_url FROM users WHERE id = ?",
            (uid,),
        ).fetchone()
    if not row:
        session.clear()
        return jsonify({"ok": False, "message": "用户不存在"}), 401
    return jsonify({"ok": True, "user": dict(row)})


@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    if len(username) < 3:
        return jsonify({"ok": False, "message": "用户名至少 3 位"}), 400
    if len(password) < 6:
        return jsonify({"ok": False, "message": "密码至少 6 位"}), 400

    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if exists:
            return jsonify({"ok": False, "message": "用户名已存在"}), 400
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, generate_password_hash(password), int(time.time())),
        )
        conn.commit()
        uid = cur.lastrowid

    session["user_id"] = uid
    session.permanent = True
    return jsonify({"ok": True, "message": "注册成功"})


@app.post("/api/login")
def api_login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    if not username or not password:
        return jsonify({"ok": False, "message": "请输入用户名和密码"}), 400

    remember = bool(data.get("remember", False))
    now = int(time.time())

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT id, username, password_hash, failed_login_attempts, lock_until FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    if not row:
        return jsonify({"ok": False, "message": "用户名或密码错误"}), 401
    if int(row["lock_until"] or 0) > now:
        remain = int(row["lock_until"]) - now
        return jsonify({"ok": False, "message": f"登录已锁定，请 {remain} 秒后重试"}), 429
    if not check_password_hash(row["password_hash"], password):
        failed = int(row["failed_login_attempts"] or 0) + 1
        lock_until = 0
        if failed >= 5:
            lock_until = now + 15 * 60
            failed = 0
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "UPDATE users SET failed_login_attempts = ?, lock_until = ? WHERE id = ?",
                (failed, lock_until, row["id"]),
            )
            conn.commit()
        if lock_until > now:
            return jsonify({"ok": False, "message": "连续失败过多，账号已锁定 15 分钟"}), 429
        return jsonify({"ok": False, "message": "用户名或密码错误"}), 401

    session["user_id"] = row["id"]
    session.permanent = remember
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE users SET failed_login_attempts = 0, lock_until = 0 WHERE id = ?",
            (row["id"],),
        )
        conn.commit()
    return jsonify({"ok": True, "message": "登录成功"})


@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True, "message": "已退出登录"})


@app.post("/api/forgot-password")
def api_forgot_password():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    new_password = str(data.get("new_password", "")).strip()
    if len(username) < 3 or len(new_password) < 6:
        return jsonify({"ok": False, "message": "用户名或新密码格式不正确"}), 400
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return jsonify({"ok": False, "message": "用户不存在"}), 404
        conn.execute(
            "UPDATE users SET password_hash = ?, failed_login_attempts = 0, lock_until = 0 WHERE id = ?",
            (generate_password_hash(new_password), row[0]),
        )
        conn.commit()
    return jsonify({"ok": True, "message": "密码已重置，请重新登录"})


@app.post("/api/me/avatar")
def api_update_avatar():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    if "avatar" not in request.files:
        return jsonify({"ok": False, "message": "缺少头像文件"}), 400
    avatar = request.files["avatar"]
    if not avatar.filename or not _allowed(avatar.filename):
        return jsonify({"ok": False, "message": "头像格式不支持"}), 400
    ext = Path(avatar.filename).suffix.lower()
    avatar_name = f"avatar_{uid}_{int(time.time() * 1000)}{ext}"
    avatar_path = AVATAR_DIR / avatar_name
    avatar.save(str(avatar_path))
    avatar_url = url_for("uploads_file", path=f"avatars/{avatar_name}")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE users SET avatar_url = ? WHERE id = ?", (avatar_url, uid))
        conn.commit()
    return jsonify({"ok": True, "avatar_url": avatar_url})


@app.post("/api/me/password")
def api_change_password():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    data = request.get_json(silent=True) or {}
    old_password = str(data.get("old_password", "")).strip()
    new_password = str(data.get("new_password", "")).strip()
    if len(new_password) < 6:
        return jsonify({"ok": False, "message": "新密码至少 6 位"}), 400
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (uid,)).fetchone()
        if not row or not check_password_hash(row["password_hash"], old_password):
            return jsonify({"ok": False, "message": "旧密码错误"}), 400
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (generate_password_hash(new_password), uid),
        )
        conn.commit()
    return jsonify({"ok": True, "message": "密码修改成功"})


@app.get("/api/models")
def api_models():
    models = _scan_models()
    default_model_id = next(iter(models.keys()), None)
    return jsonify(
        {
            "ok": True,
            "default": default_model_id,
            "models": list(models.values()),
        }
    )


@app.get("/api/model-info")
def api_model_info():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    model_id = (request.args.get("model_id") or "").strip()
    models = _scan_models()
    if not model_id:
        model_id = next(iter(models.keys()), "")
    if not model_id or model_id not in models:
        return jsonify({"ok": False, "message": "模型不存在"}), 404
    try:
        info = _read_model_results_csv(model_id)
    except Exception as exc:
        return jsonify({"ok": False, "message": f"读取模型信息失败: {exc}"}), 500
    return jsonify({"ok": True, "info": info})


@app.post("/api/drone/plan")
def api_drone_plan():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    payload = request.get_json(silent=True) or {}
    try:
        if payload.get("use_reference_plan", True):
            plan = _load_uav_reference_plan()
        else:
            plan = _compute_drone_plan(payload)
    except Exception as exc:
        return jsonify({"ok": False, "message": f"无人机决策计算失败: {exc}"}), 400

    summary = plan.get("summary", {})
    operation = plan.get("operation", {})
    risk_score = plan.get("risk_score")
    if risk_score is None:
        parcels = plan.get("parcels", [])
        risk_score = round(sum(float(p.get("R_total") or 0) for p in parcels) / max(len(parcels), 1), 4)
    risk_level = plan.get("risk_level") or "变量喷洒"
    field_area_mu = float(summary.get("农田面积(m²)", 0) or 0) / 666.6667 if summary else plan.get("field", {}).get("area_mu", 0)
    flight_distance_m = float(summary.get("实际喷洒航线长度(m)", 0) or operation.get("flight_distance_m", 0) or 0)
    spray_volume_l = float(summary.get("触发式变量喷洒药量(L)", 0) or operation.get("spray_volume_l", 0) or 0)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO drone_plans (
                user_id, ts, plan_name, crop_type, disease_type, risk_score, risk_level,
                field_area_mu, flight_distance_m, spray_volume_l, plan_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                int(time.time() * 1000),
                plan["plan_name"],
                plan.get("crop_type", "多作物农田"),
                plan.get("disease_type", "病害任务区"),
                risk_score,
                risk_level,
                field_area_mu,
                flight_distance_m,
                spray_volume_l,
                json.dumps(plan, ensure_ascii=False),
            ),
        )
        conn.commit()
    return jsonify({"ok": True, "plan": plan})


@app.get("/api/drone/reference")
def api_drone_reference():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    try:
        return jsonify({"ok": True, "plan": _load_uav_reference_plan()})
    except Exception as exc:
        return jsonify({"ok": False, "message": f"读取无人机方法文件失败: {exc}"}), 500


@app.get("/api/drone/plans")
def api_drone_plans():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    limit = int(request.args.get("limit", 12))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, ts, plan_name, crop_type, disease_type, risk_score, risk_level,
                   field_area_mu, flight_distance_m, spray_volume_l, plan_json
            FROM drone_plans WHERE user_id = ? ORDER BY id DESC LIMIT ?
            """,
            (uid, limit),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["plan"] = json.loads(item.pop("plan_json") or "{}")
        items.append(item)
    return jsonify({"ok": True, "items": items})


@app.post("/api/detect")
def api_detect():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    if "image" not in request.files:
        return jsonify({"ok": False, "message": "缺少 image 文件"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"ok": False, "message": "文件名为空"}), 400
    if not _allowed(file.filename):
        return jsonify({"ok": False, "message": "不支持的图片格式"}), 400

    models = _scan_models()
    model_id = request.form.get("model_id") or next(iter(models.keys()), None)
    if not model_id:
        return jsonify({"ok": False, "message": "未发现可用模型"}), 400
    if model_id not in models:
        return jsonify({"ok": False, "message": f"模型不存在: {model_id}"}), 400

    conf = float(request.form.get("conf", 0.25))
    iou = float(request.form.get("iou", 0.45))

    ext = Path(file.filename).suffix.lower()
    image_name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}{ext}"
    image_path = UPLOAD_DIR / image_name
    file.save(str(image_path))

    try:
        detect_data = _run_detection_on_image(image_path, model_id, conf, iou)
    except Exception as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "message": f"模型推理失败：{exc}",
                    "hint": "请检查模型是否依赖自定义网络模块（如 emcad），并确认已从项目内 ultralytics 加载。",
                }
            ),
            500,
        )
    original_url = url_for("uploads_file", path=image_name)
    result_url = url_for("results_file", path=detect_data["result_name"])

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO history (user_id, ts, model_id, model_label, total, inference_ms, stats_json, original_url, result_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uid,
                int(time.time() * 1000),
                model_id,
                models[model_id]["model_label"],
                detect_data["total"],
                detect_data["inference_ms"],
                json.dumps(detect_data["stats"], ensure_ascii=False),
                original_url,
                result_url,
            ),
        )
        conn.commit()

    return jsonify(
        {
            "ok": True,
            "model_id": model_id,
            "model_label": models[model_id]["model_label"],
            "original_url": original_url,
            "result_url": result_url,
            "detections": detect_data["detections"],
            "stats": detect_data["stats"],
            "total": detect_data["total"],
            "inference_ms": detect_data["inference_ms"],
        }
    )


@app.get("/api/history")
def api_history():
    uid = _require_login_api()
    if not uid:
        return jsonify({"ok": False, "message": "请先登录"}), 401
    limit = int(request.args.get("limit", 20))
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, ts, model_id, model_label, total, inference_ms, stats_json, original_url, result_url
            FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?
            """,
            (uid, limit),
        ).fetchall()

    data = []
    for row in rows:
        stats = json.loads(row["stats_json"]) if row["stats_json"] else {}
        data.append({**dict(row), "stats": stats})
    return jsonify({"ok": True, "items": data})


@app.route("/uploads/<path:path>")
def uploads_file(path: str):
    return send_from_directory(UPLOAD_DIR, path)


@app.route("/results/<path:path>")
def results_file(path: str):
    return send_from_directory(RESULT_DIR, path)


@app.route("/image/<path:path>")
def image_file(path: str):
    return send_from_directory(IMAGE_DIR, path)


@app.route("/uav-assets/<path:path>")
def uav_assets_file(path: str):
    return send_from_directory(UAV_SOURCE_DIR, path)


if __name__ == "__main__":
    _init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
