# -*- coding: utf-8 -*-
"""
最新版本：一比一农田空间图（道路优化 + 农田颜色调整版）
功能：
1. 农田整体比例为 1:1
2. 道路网络更合理，包含交汇、跨河
3. 道路避开湖泊
4. 除河流/道路/建筑物/湖泊外，其余区域全部划为农田
5. 农田颜色调整为更接近参考图的浅黄绿色风格
"""

import warnings
from pathlib import Path
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

# ============================
# 全局参数
# ============================
SEED = 42
np.random.seed(SEED)

OUT_DIR = Path("uav_farmland_output_square")
OUT_DIR.mkdir(exist_ok=True)

# 1:1 画布
W, H = 1000, 1000

# 原始参考尺寸
OLD_W, OLD_H = 1000, 720
SX = W / OLD_W
SY = H / OLD_H
S_LEN = (SX + SY) / 2.0

ROAD_W = S_LEN * 4.2


# ============================
# 坐标缩放工具
# ============================
def S(x, y):
    return (x * SX, y * SY)


def SL(v):
    return v * S_LEN


# ============================
# 绘图工具
# ============================
def add_geom(ax, geom, fc, ec="white", lw=0.8, alpha=1.0, z=1):
    if geom.is_empty:
        return

    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            add_geom(ax, g, fc=fc, ec=ec, lw=lw, alpha=alpha, z=z)
        return

    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.fill(
            x, y,
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            alpha=alpha,
            zorder=z
        )


def setup_map_ax(ax, title):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, pad=12)

    for s in ax.spines.values():
        s.set_visible(False)


# ============================
# Voronoi 有限区域转换
# ============================
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("输入必须是二维点")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)

            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


# ============================
# 更均匀的种子点采样
# ============================
def poisson_like_points(poly, n_points, min_dist=74, max_try=90000):
    points = []
    minx, miny, maxx, maxy = poly.bounds
    tries = 0

    while len(points) < n_points and tries < max_try:
        tries += 1
        p = np.array([
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        ])

        if not poly.contains(Point(*p)):
            continue

        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)

    while len(points) < n_points and tries < max_try * 2:
        tries += 1
        p = np.array([
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        ])

        if not poly.contains(Point(*p)):
            continue

        if all(np.linalg.norm(p - q) >= min_dist * 0.78 for q in points):
            points.append(p)

    return np.asarray(points)


# ============================
# 几何拆分工具
# ============================
def explode_polygons(geom):
    if geom.is_empty:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            out.extend(explode_polygons(g))
        return out

    return []


def polar_point(center, radius, angle_deg):
    a = math.radians(angle_deg)
    return (
        center[0] + radius * math.cos(a),
        center[1] + radius * math.sin(a)
    )


# ============================
# 构建农田空间
# ============================
def build_farmland_space():
    boundary = Polygon([(0, 0), (W, 0), (W, H), (0, H)])

    # ------------------------
    # 水体与建筑物
    # ------------------------
    river_main = LineString([
        S(-20, 500), S(90, 490), S(210, 525), S(330, 500), S(455, 535),
        S(590, 495), S(710, 525), S(850, 500), S(1020, 490)
    ]).buffer(SL(19), cap_style=2, join_style=2)

    river_left = LineString([
        S(160, 720), S(155, 600), S(180, 455), S(145, 310), S(170, 150), S(150, -10)
    ]).buffer(SL(11), cap_style=2, join_style=2)

    pond_left = Point(*S(320, 230)).buffer(SL(28), resolution=64)
    pond_right = Point(*S(665, 315)).buffer(SL(36), resolution=64)
    lakes = unary_union([pond_left, pond_right])

    village_center = S(500, 335)
    village_radius = SL(63)
    village = Point(*village_center).buffer(village_radius, resolution=96)

    # ------------------------
    # 更合理的道路网络
    # ------------------------
    # 顶部横向道路（已上移一点）
    n_junc = S(365, 568)

    road_north = LineString([
        S(-20, 623), S(150, 598), S(270, 578), n_junc,
        S(540, 563), S(770, 575), S(1020, 593)
    ])

    # 连接北路与中部主路
    w_junc = S(320, 270)
    road_connector = LineString([
        n_junc, S(350, 505), S(338, 425), w_junc
    ])

    # 西侧主路
    ring_radius = village_radius + ROAD_W * 1.8
    ring_w = polar_point(village_center, ring_radius, 205)
    ring_sw = polar_point(village_center, ring_radius, 250)
    ring_se = polar_point(village_center, ring_radius, 305)
    ring_e = polar_point(village_center, ring_radius, 15)

    road_main = LineString([
        S(-20, 220), S(120, 215), S(230, 220), w_junc, ring_w
    ])

    # 村庄外环道路
    road_ring = LineString([ring_w, ring_sw, ring_se, ring_e])

    # 东向道路（绕开右侧湖泊）
    road_east = LineString([
        ring_e, S(615, 370), S(720, 372), S(860, 360), S(1020, 320)
    ])

    # 东南道路
    road_southeast = LineString([
        ring_se, S(590, 175), S(720, 140), S(880, 110), S(1020, 85)
    ])

    roads_center = [
        road_north,
        road_connector,
        road_main,
        road_ring,
        road_east,
        road_southeast
    ]

    # 检查道路不得与湖泊相交（但允许与河流交汇）
    for i, r in enumerate(roads_center, start=1):
        if r.intersects(lakes.buffer(ROAD_W * 0.6)):
            raise RuntimeError(f"道路 {i} 与湖泊相交或过近，请调整")

    road_buffer = unary_union([
        r.buffer(ROAD_W, cap_style=2, join_style=2)
        for r in roads_center
    ])

    # 障碍物
    obstacles = unary_union([
        river_main,
        river_left,
        pond_left,
        pond_right,
        village,
        road_buffer
    ])

    # 剩余区域全部划为农田
    field_area = boundary.difference(obstacles)

    # ------------------------
    # 农田分块
    # ------------------------
    seed_points = poisson_like_points(field_area, n_points=88, min_dist=76)
    vor = Voronoi(seed_points)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=3000)

    parcels = []
    for reg in regions:
        poly = Polygon(vertices[reg]).intersection(field_area)
        for g in explode_polygons(poly):
            if g.area > 10:
                parcels.append(g)

    # 补齐残余，确保没有额外空白
    union_parcels = unary_union(parcels) if parcels else Polygon()
    residual = field_area.difference(union_parcels)

    for g in explode_polygons(residual):
        if g.area > 1:
            parcels.append(g)

    return {
        "boundary": boundary,
        "field_area": field_area,
        "parcels": parcels,
        "river_main": river_main,
        "river_left": river_left,
        "pond_left": pond_left,
        "pond_right": pond_right,
        "village": village,
        "roads_center": roads_center,
        "road_buffer": road_buffer
    }


# ============================
# 画桥的标识
# ============================
def draw_bridge_mark(ax, road, river_poly, length=22):
    inter = road.intersection(river_poly)

    if inter.is_empty:
        return

    geoms = []
    if inter.geom_type == "LineString":
        geoms = [inter]
    elif inter.geom_type == "MultiLineString":
        geoms = list(inter.geoms)

    for g in geoms:
        if g.length < 1:
            continue

        mid = g.interpolate(0.5, normalized=True)
        t1 = g.interpolate(max(0.0, 0.3), normalized=True)
        t2 = g.interpolate(min(1.0, 0.7), normalized=True)

        dx, dy = t2.x - t1.x, t2.y - t1.y
        norm = (dx**2 + dy**2) ** 0.5
        if norm == 0:
            continue

        dx, dy = dx / norm, dy / norm
        px, py = -dy, dx

        half = length / 2
        x1, y1 = mid.x - px * half, mid.y - py * half
        x2, y2 = mid.x + px * half, mid.y + py * half

        ax.plot([x1, x2], [y1, y2], color="#8a8a8a", linewidth=1.2, zorder=10)
        ax.plot([x1, x2], [y1, y2], color="#f7f7f2", linewidth=0.6, zorder=11)


# ============================
# 农田颜色：改成更接近参考图
# ============================
def get_farmland_palette():
    """
    更接近你给的参考图：
    偏浅黄绿、淡绿、少量偏深橄榄绿
    """
    return [
        "#dfe7a6",  # 很浅黄绿
        "#d7e096",
        "#cfdb88",
        "#c7d67b",
        "#bdd06f",
        "#b4ca67",
        "#a9bf63",
        "#96ad58"   # 少量稍深色
    ]


# ============================
# 绘制底图
# ============================
def draw_base_map(layers, save_path):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # 水体、建筑物、道路配色
    water_color = "#90cadd"
    water_edge = "#5eb3c7"
    village_color = "#dacfb9"
    road_edge = "#d8d9ce"
    road_color = "#f3efe6"

    # 农田颜色
    farmland_palette = get_farmland_palette()

    fig, ax = plt.subplots(figsize=(10, 10), dpi=240)
    setup_map_ax(ax, "Square farmland spatial model (road-network refined)")

    # 农田块颜色更接近参考图
    rng = np.random.default_rng(SEED)
    palette_idx = rng.integers(0, len(farmland_palette), size=len(layers["parcels"]))

    # 少量块更深一点，模拟参考图中的深绿色地块
    deep_idx = rng.choice(
        np.arange(len(layers["parcels"])),
        size=max(1, len(layers["parcels"]) // 10),
        replace=False
    )

    for i, p in enumerate(layers["parcels"]):
        fc = farmland_palette[palette_idx[i]]
        if i in deep_idx:
            fc = "#8ea659"
        add_geom(ax, p, fc=fc, ec="#f3f6ea", lw=0.8, alpha=0.99, z=1)

    # 水体
    for water in [
        layers["river_main"],
        layers["river_left"],
        layers["pond_left"],
        layers["pond_right"]
    ]:
        add_geom(ax, water, water_color, ec=water_edge, lw=1.4, alpha=0.98, z=4)

    # 建筑物
    add_geom(ax, layers["village"], village_color, ec="white", lw=2.0, alpha=0.99, z=5)

    # 道路
    for r in layers["roads_center"]:
        x, y = r.xy
        ax.plot(x, y, color=road_edge, linewidth=5.4, solid_capstyle="round", zorder=8)
        ax.plot(x, y, color=road_color, linewidth=3.25, solid_capstyle="round", zorder=9)

        draw_bridge_mark(ax, r, layers["river_main"], length=24)
        draw_bridge_mark(ax, r, layers["river_left"], length=22)

    # 图例
    legend = [
        Patch(facecolor="#c7d67b", edgecolor="white", label="Farmland parcel"),
        Patch(facecolor=water_color, edgecolor=water_edge, label="River / lake"),
        Patch(facecolor=village_color, edgecolor="white", label="Building cluster"),
        Line2D([0], [0], color=road_color, linewidth=3.25, label="Road")
    ]

    ax.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10.5
    )

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ============================
# 主程序
# ============================
def main():
    layers = build_farmland_space()

    out_path = OUT_DIR / "图1_一比一农田空间模型_颜色调整版.png"
    draw_base_map(layers, out_path)

    farmland_union = unary_union(layers["parcels"]) if layers["parcels"] else Polygon()
    residual = layers["field_area"].difference(farmland_union)
    residual_area = residual.area if not residual.is_empty else 0.0

    print(f"图已生成：{out_path.resolve()}")
    print(f"地块数: {len(layers['parcels'])}")
    print(f"农田未填充残余面积: {residual_area:.6f}")


if __name__ == "__main__":
    main()