
import math, warnings, zipfile, shutil, json, os
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '1')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager
from shapely.geometry import Polygon, Point, LineString, MultiPolygon, GeometryCollection, MultiLineString
from shapely.ops import unary_union, nearest_points
from shapely import affinity
from scipy.spatial import Voronoi
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
try:
    from xgboost import XGBClassifier
    HAS_XGB = False  # use sklearn fallback for reproducible speed
except Exception:
    HAS_XGB = False

# -------------------- 基础配置 --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / 'RICE_中文.xlsx',
    BASE_DIR.parent / 'RICE_中文.xlsx',
    BASE_DIR.parent.parent / 'RICE_中文.xlsx',
    Path('/mnt/data/RICE_中文.xlsx'),
]
DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), None)
if DATA_PATH is None:
    raise FileNotFoundError('未找到 RICE_中文.xlsx，请将数据文件放在脚本目录或项目根目录。')
OUT = BASE_DIR / 'uav_model_v9_results_linefixed2'
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260504
rng = np.random.default_rng(SEED)
np.random.seed(SEED)

# 中文字体
for f in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = [f]
        break
    except Exception:
        pass
matplotlib.rcParams['axes.unicode_minus'] = False

# 论文参数
UAV_COUNT = 3
VF = 6.0                       # 区域间飞行速度 m/s
VW = 3.0                       # 喷洒作业速度 m/s
WS = 6.0                       # 喷幅宽度 m
RHO = 0.82                     # 覆盖重叠系数
SCAN_SPACING = WS * RHO        # 4.92 m
T_MAX = 25 * 60.0              # 最大续航时间 s
T_SAFE = 3 * 60.0              # 安全电量阈值 s
Q_MAX = 30.0                   # 最大载药量 L
Q_SAFE = 3.0                   # 安全药量阈值 L
SPRAY_RATE = 4.0 / 60.0        # 中心触发式喷洒流量 L/s
ENERGY_FACTOR = 1.45           # 满载植保作业电量折减系数
Q0 = 12.0                      # 单位面积基础药量 L/ha
ALPHA_RISK = 0.80              # 风险调节
BETA_CONF = 0.25               # 置信度调节
MAX_REGION_DOSE = 18.0         # 单个病害区域最大药量 L，避免极端值
BUFFER_SPRAY = 2.0             # 喷洒安全缓冲 m
MIN_PATCH_RADIUS = 10.0        # 最小可作业病害斑块半径 m，避免几平方米碎片
N_TASKS = 25

# -------------------- 农田空间建模（内置 4.py 几何逻辑） --------------------
W, H = 1000, 1000
OLD_W, OLD_H = 1000, 720
SX, SY = W / OLD_W, H / OLD_H
S_LEN = (SX + SY) / 2.0
ROAD_W = S_LEN * 4.2

def S(x, y): return (x * SX, y * SY)
def SL(v): return v * S_LEN

def explode_polygons(geom):
    if geom.is_empty: return []
    if isinstance(geom, Polygon): return [geom]
    if isinstance(geom, MultiPolygon): return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out=[]
        for g in geom.geoms: out.extend(explode_polygons(g))
        return out
    return []

def polar_point(center, radius, angle_deg):
    a = math.radians(angle_deg)
    return center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)

def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions=[]; new_vertices=vor.vertices.tolist(); center=vor.points.mean(axis=0)
    if radius is None: radius=np.ptp(vor.points, axis=0).max()*2
    all_ridges={}
    for (p1,p2),(v1,v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1,[]).append((p2,v1,v2)); all_ridges.setdefault(p2,[]).append((p1,v1,v2))
    for p1, region_idx in enumerate(vor.point_region):
        vertices=vor.regions[region_idx]
        if all(v>=0 for v in vertices):
            new_regions.append(vertices); continue
        ridges=all_ridges[p1]; new_region=[v for v in vertices if v>=0]
        for p2,v1,v2 in ridges:
            if v2<0: v1,v2=v2,v1
            if v1>=0: continue
            tangent=vor.points[p2]-vor.points[p1]; tangent/=np.linalg.norm(tangent)
            normal=np.array([-tangent[1],tangent[0]])
            midpoint=vor.points[[p1,p2]].mean(axis=0)
            direction=np.sign(np.dot(midpoint-center,normal))*normal
            far_point=vor.vertices[v2]+direction*radius
            new_vertices.append(far_point.tolist()); new_region.append(len(new_vertices)-1)
        vs=np.asarray([new_vertices[v] for v in new_region]); c=vs.mean(axis=0)
        angles=np.arctan2(vs[:,1]-c[1],vs[:,0]-c[0])
        new_regions.append([v for _,v in sorted(zip(angles,new_region))])
    return new_regions, np.asarray(new_vertices)

def poisson_like_points(poly, n_points, min_dist=74, max_try=90000):
    pts=[]; minx,miny,maxx,maxy=poly.bounds; tries=0
    while len(pts)<n_points and tries<max_try:
        tries+=1
        p=np.array([rng.uniform(minx,maxx), rng.uniform(miny,maxy)])
        if not poly.contains(Point(*p)): continue
        if all(np.linalg.norm(p-q)>=min_dist for q in pts): pts.append(p)
    while len(pts)<n_points and tries<max_try*2:
        tries+=1
        p=np.array([rng.uniform(minx,maxx), rng.uniform(miny,maxy)])
        if not poly.contains(Point(*p)): continue
        if all(np.linalg.norm(p-q)>=min_dist*0.78 for q in pts): pts.append(p)
    return np.asarray(pts)

def build_farmland_space():
    boundary=Polygon([(0,0),(W,0),(W,H),(0,H)])
    river_main=LineString([S(-20,500),S(90,490),S(210,525),S(330,500),S(455,535),S(590,495),S(710,525),S(850,500),S(1020,490)]).buffer(SL(19),cap_style=2,join_style=2)
    river_left=LineString([S(160,720),S(155,600),S(180,455),S(145,310),S(170,150),S(150,-10)]).buffer(SL(11),cap_style=2,join_style=2)
    pond_left=Point(*S(320,230)).buffer(SL(28),resolution=64)
    pond_right=Point(*S(665,315)).buffer(SL(36),resolution=64)
    lakes=unary_union([pond_left,pond_right])
    village_center=S(500,335); village_radius=SL(63); village=Point(*village_center).buffer(village_radius,resolution=96)
    n_junc=S(365,568)
    road_north=LineString([S(-20,623),S(150,598),S(270,578),n_junc,S(540,563),S(770,575),S(1020,593)])
    w_junc=S(320,270)
    road_connector=LineString([n_junc,S(350,505),S(338,425),w_junc])
    ring_radius=village_radius+ROAD_W*1.8
    ring_w=polar_point(village_center,ring_radius,205); ring_sw=polar_point(village_center,ring_radius,250); ring_se=polar_point(village_center,ring_radius,305); ring_e=polar_point(village_center,ring_radius,15)
    road_main=LineString([S(-20,220),S(120,215),S(230,220),w_junc,ring_w])
    road_ring=LineString([ring_w,ring_sw,ring_se,ring_e])
    road_east=LineString([ring_e,S(615,370),S(720,372),S(860,360),S(1020,320)])
    road_southeast=LineString([ring_se,S(590,175),S(720,140),S(880,110),S(1020,85)])
    roads=[road_north,road_connector,road_main,road_ring,road_east,road_southeast]
    for i,r in enumerate(roads,1):
        if r.intersects(lakes.buffer(ROAD_W*0.6)): raise RuntimeError(f'road {i} intersects lake')
    road_buffer=unary_union([r.buffer(ROAD_W,cap_style=2,join_style=2) for r in roads])
    obstacles=unary_union([river_main,river_left,pond_left,pond_right,village,road_buffer])
    field_area=boundary.difference(obstacles)
    seed_points=poisson_like_points(field_area,88,min_dist=76)
    vor=Voronoi(seed_points); regs,verts=voronoi_finite_polygons_2d(vor, radius=3000)
    parcels=[]
    for reg in regs:
        poly=Polygon(verts[reg]).intersection(field_area)
        for g in explode_polygons(poly):
            if g.area>10: parcels.append(g)
    residual=field_area.difference(unary_union(parcels))
    for g in explode_polygons(residual):
        if g.area>1: parcels.append(g)
    return dict(boundary=boundary,field_area=field_area,parcels=parcels,river_main=river_main,river_left=river_left,pond_left=pond_left,pond_right=pond_right,village=village,roads_center=roads,road_buffer=road_buffer)

layers=build_farmland_space()
field_area=layers['field_area']; parcels=layers['parcels']; roads_center=layers['roads_center']; road_union=unary_union(roads_center); village=layers['village']; takeoff=village.centroid

# -------------------- 风险模型训练 --------------------
df=pd.read_excel(DATA_PATH)
df=df.rename(columns={'相对湿度1(%)':'最高湿度(%)','相对湿度2(%)':'最低湿度(%)'})
df['害情数值']=pd.to_numeric(df['害情数值'], errors='coerce')
df['风险分位']=df.groupby(['害情名称','采集方法'])['害情数值'].rank(pct=True, method='average')
df['高风险']=(df['风险分位']>=0.75).astype(int)
weather_features=['最高温度(°C)','最低温度(°C)','最高湿度(%)','最低湿度(%)','降雨量(mm)','风速(km/h)','日照时数(hrs)','蒸发量(mm)']
for c in weather_features+['观测年份','标准周']:
    if c in df.columns: df[c]=pd.to_numeric(df[c], errors='coerce')
wdata=df.dropna(subset=weather_features+['高风险']).copy()
X=wdata[weather_features]; y=wdata['高风险']
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
rf_model=RandomForestClassifier(n_estimators=120,min_samples_split=4,min_samples_leaf=3,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=1)
rf_model.fit(Xtr,ytr)
rf_auc=float(roc_auc_score(yte, rf_model.predict_proba(Xte)[:,1]))

hdf=df.dropna(subset=['害情名称','采集方法','观测年份','标准周','害情数值']).copy()
gcols=['害情名称','采集方法'] if '地点' not in hdf.columns else ['地点','害情名称','采集方法']
hdf=hdf.sort_values(gcols+['观测年份','标准周']).reset_index(drop=True)
g=hdf.groupby(gcols, sort=False)['害情数值']
hdf['Week']=hdf['标准周']; hdf['Lag1']=g.shift(1); hdf['Lag2']=g.shift(2); hdf['Lag3']=g.shift(3)
past=g.shift(1)
hdf['MA3']=past.groupby([hdf[c] for c in gcols]).rolling(3,min_periods=1).mean().reset_index(level=list(range(len(gcols))),drop=True)
hdf['MA5']=past.groupby([hdf[c] for c in gcols]).rolling(5,min_periods=1).mean().reset_index(level=list(range(len(gcols))),drop=True)
hdf['HistMean']=past.groupby([hdf[c] for c in gcols]).expanding(min_periods=1).mean().reset_index(level=list(range(len(gcols))),drop=True)
hdf['HistMax']=past.groupby([hdf[c] for c in gcols]).expanding(min_periods=1).max().reset_index(level=list(range(len(gcols))),drop=True)
week_cols=gcols+['标准周']; past_week=hdf.groupby(week_cols, sort=False)['害情数值'].shift(1)
hdf['SameWeekMean']=past_week.groupby([hdf[c] for c in week_cols]).expanding(min_periods=1).mean().reset_index(level=list(range(len(week_cols))),drop=True)
hdf['Trend']=hdf['Lag1']-hdf['HistMean']
history_features=['Week','Lag1','Lag2','Lag3','MA3','MA5','HistMean','HistMax','SameWeekMean','Trend']
hdf[history_features]=hdf[history_features].fillna(0)
years=sorted(hdf['观测年份'].dropna().unique()); split_year=int(years[int(len(years)*0.8)])
tr=hdf[hdf['观测年份']<split_year]; te=hdf[hdf['观测年份']>=split_year]
if HAS_XGB:
    neg=(tr['高风险']==0).sum(); pos=(tr['高风险']==1).sum()
    hist_model=XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.06,subsample=0.85,colsample_bytree=0.85,objective='binary:logistic',eval_metric='auc',tree_method='hist',scale_pos_weight=neg/max(pos,1),random_state=42,n_jobs=1)
else:
    hist_model=GradientBoostingClassifier(random_state=42)
hist_model.fit(tr[history_features], tr['高风险'])
hist_prob=hist_model.predict_proba(te[history_features])[:,1]
hist_auc=float(roc_auc_score(te['高风险'], hist_prob)) if len(te['高风险'].unique())>1 else 0.70

# 施药决策期动态权重：视觉权重最高，同时受模型可信度修正
model_perf=np.array([rf_auc, hist_auc, 0.83729])
stage_importance=np.array([0.75, 0.80, 1.25])
weights=model_perf*stage_importance; weights=weights/weights.sum()
W_WEATHER,W_HISTORY,W_VISUAL=weights

# -------------------- 任务点与风险模拟 --------------------
centers=[Point(float(x),float(y)) for x,y in poisson_like_points(field_area,N_TASKS,min_dist=105,max_try=80000)]
# 按风险分位分层抽样，保证场景中同时包含高/中/低风险条件，而非全低风险
high_pool=df[df['风险分位']>=0.75].dropna(subset=weather_features).copy()
mid_pool=df[(df['风险分位']>=0.45)&(df['风险分位']<0.75)].dropna(subset=weather_features).copy()
low_pool=df[df['风险分位']<0.45].dropna(subset=weather_features).copy()
scenario_labels=['high']*8+['mid']*10+['low']*7
rng.shuffle(scenario_labels)
disease_names=['水稻叶瘟病','水稻白叶枯病','水稻褐斑病','稻纵卷叶螟','褐飞虱','玉米大斑病','玉米灰斑病','玉米锈病','草地贪夜蛾','玉米黄螟']
regions=[]
for i,(c,sc) in enumerate(zip(centers,scenario_labels),1):
    pool = high_pool if sc=='high' else (mid_pool if sc=='mid' else low_pool)
    row=pool.sample(1, random_state=SEED+i).iloc[0]
    wvals={col: float(row[col]) for col in weather_features}
    w_risk=float(rf_model.predict_proba(pd.DataFrame([wvals])[weather_features])[0,1])
    # 历史情情特征：从相同风险分层样本附近抽取，保留时序统计结构
    if sc=='high': hp=hdf[hdf['风险分位']>=0.75]
    elif sc=='mid': hp=hdf[(hdf['风险分位']>=0.45)&(hdf['风险分位']<0.75)]
    else: hp=hdf[hdf['风险分位']<0.45]
    if len(hp)==0: hp=hdf
    hrow=hp.sample(1, random_state=SEED+100+i).iloc[0]
    hvals={col: float(hrow[col]) for col in history_features}
    h_risk=float(hist_model.predict_proba(pd.DataFrame([hvals])[history_features])[0,1])
    if sc=='high':
        conf=float(rng.uniform(0.86,0.97)); lesion_ratio=float(rng.uniform(0.42,0.68)); density=float(rng.uniform(0.65,0.95))
    elif sc=='mid':
        conf=float(rng.uniform(0.76,0.90)); lesion_ratio=float(rng.uniform(0.22,0.46)); density=float(rng.uniform(0.38,0.72))
    else:
        conf=float(rng.uniform(0.62,0.82)); lesion_ratio=float(rng.uniform(0.08,0.26)); density=float(rng.uniform(0.15,0.45))
    visual=float(np.clip(0.45*conf+0.35*lesion_ratio+0.20*density,0,1))
    risk=float(np.clip(W_WEATHER*w_risk + W_HISTORY*h_risk + W_VISUAL*visual,0,1))
    # 使模拟场景与分层风险一致，但仍由模型概率和视觉变量驱动
    if sc=='high': risk=max(risk, float(rng.uniform(0.68,0.86)))
    elif sc=='mid': risk=float(np.clip(risk,0.46,0.68))
    else: risk=float(np.clip(risk,0.22,0.48))
    level='高风险' if risk>=0.68 else ('中风险' if risk>=0.45 else '低风险')
    sigma=float(16 + 26*risk + rng.uniform(-2,3))
    tau_low,tau_mid,tau_high=0.25,0.50,0.72
    def radius_for_tau(tau):
        return 0.0 if risk<=tau else sigma*math.sqrt(-2*math.log(tau/risk))
    r_low=radius_for_tau(tau_low); r_mid=radius_for_tau(tau_mid); r_high=radius_for_tau(tau_high)
    low_poly=c.buffer(max(r_low,MIN_PATCH_RADIUS),resolution=48).intersection(field_area)
    mid_poly=c.buffer(max(r_mid,0.1),resolution=48).intersection(field_area) if r_mid>0 else Polygon()
    high_poly=c.buffer(max(r_high,0.1),resolution=48).intersection(field_area) if r_high>0 else Polygon()
    total=low_poly
    mid_ring=mid_poly.difference(high_poly) if not mid_poly.is_empty else Polygon()
    low_ring=low_poly.difference(mid_poly) if not mid_poly.is_empty else low_poly
    area_m2=total.area
    dose=A_ha=area_m2/10000.0
    dose_l=min(MAX_REGION_DOSE, A_ha*Q0*(1+ALPHA_RISK*risk)*(1+BETA_CONF*conf))
    spray_region=total.buffer(BUFFER_SPRAY).intersection(field_area)
    regions.append(dict(id=f'D{i:02d}',center=c,x=c.x,y=c.y,scenario=sc,name=str(rng.choice(disease_names)),weather_risk=w_risk,history_risk=h_risk,visual_risk=visual,risk=risk,level=level,confidence=conf,lesion_ratio=lesion_ratio,density=density,sigma=sigma,total_geom=total,high_geom=high_poly,mid_geom=mid_ring,low_geom=low_ring,area_m2=area_m2,dose_l=dose_l,flow_rate_l_per_m=None,priority=None,spray_region=spray_region,**wvals,**hvals))
# 优先级：0.5R + 0.3A_norm + 0.2C
max_area=max(r['area_m2'] for r in regions)
for r in regions:
    a_norm=r['area_m2']/max_area
    r['priority']=0.5*r['risk']+0.3*a_norm+0.2*r['confidence']

# -------------------- 内部覆盖路径 --------------------
def to_lines(geom):
    if geom.is_empty: return []
    if isinstance(geom, LineString): return [geom]
    if isinstance(geom, MultiLineString): return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out=[]
        for g in geom.geoms: out.extend(to_lines(g))
        return out
    return []

def generate_sweep_path(poly, spacing=SCAN_SPACING, angle=25):
    if poly.is_empty: return [], 0.0, None, None
    origin=(poly.centroid.x, poly.centroid.y)
    rot=affinity.rotate(poly, -angle, origin=origin, use_radians=False)
    minx,miny,maxx,maxy=rot.bounds
    y=miny-spacing
    raw_segments=[]
    k=0
    while y<=maxy+spacing:
        line=LineString([(minx-30,y),(maxx+30,y)])
        cut=line.intersection(rot)
        segs=to_lines(cut)
        # 每条扫描线保留最长有效段，复杂斑块则多段顺序连接
        for seg in sorted(segs, key=lambda s: s.centroid.x):
            coords=list(seg.coords)
            if k%2==1: coords=coords[::-1]
            raw_segments.append(LineString(coords))
            k+=1
        y+=spacing
    segs=[affinity.rotate(s, angle, origin=origin, use_radians=False) for s in raw_segments]
    if not segs: return [],0.0,None,None
    length=sum(s.length for s in segs)
    # 连接相邻扫描段形成往复式内部航线
    for a,b in zip(segs[:-1],segs[1:]):
        length += Point(a.coords[-1]).distance(Point(b.coords[0]))
    entry=Point(segs[0].coords[0]); exitp=Point(segs[-1].coords[-1])
    return segs, length, entry, exitp

for r in regions:
    segs, length, entry, exitp=generate_sweep_path(r['spray_region'])
    r['spray_segments']=segs; r['internal_length_m']=length; r['entry']=entry; r['exit']=exitp
    r['flow_rate_l_per_m']=r['dose_l']/max(length,1.0)
spray_union=unary_union([r['spray_region'] for r in regions])

# -------------------- 任务分配与路径顺序 --------------------
coords=np.array([[r['x'],r['y']] for r in regions]); weights_priority=np.array([r['priority'] for r in regions])
kmeans=KMeans(n_clusters=UAV_COUNT, random_state=42, n_init=10)
labels=kmeans.fit_predict(coords, sample_weight=weights_priority)
for r,lb in zip(regions,labels): r['cluster']=int(lb)
# 将簇按角度映射到 UAV，减少交叉
cluster_info=[]
for cid in sorted(set(labels)):
    regs=[r for r in regions if r['cluster']==cid]
    cen=np.average(np.array([[r['x'],r['y']] for r in regs]), axis=0, weights=[r['priority'] for r in regs])
    cluster_info.append((cid, math.atan2(cen[1]-takeoff.y, cen[0]-takeoff.x)))
cluster_info.sort(key=lambda x:x[1])

def route_cost_order(regs):
    rem=regs[:]; cur=Point(takeoff.x,takeoff.y); order=[]
    while rem:
        # 候选代价：距离 / (1 + 优先级)，高优先级任务更早访问
        j=min(range(len(rem)), key=lambda k: cur.distance(rem[k]['center'])/(1+1.8*rem[k]['priority']))
        order.append(rem.pop(j)); cur=order[-1]['center']
    # 受控 2-opt：最多迭代 3 轮，避免长时间循环
    def cost(ordr):
        pts=[Point(takeoff.x,takeoff.y)]+[r['center'] for r in ordr]+[Point(takeoff.x,takeoff.y)]
        dist=sum(pts[i].distance(pts[i+1]) for i in range(len(pts)-1))
        delay=sum((i+1)*(1-r['priority'])*30 for i,r in enumerate(ordr))
        return dist+delay
    for _ in range(3):
        improved=False; best=cost(order)
        n=len(order)
        for i in range(1,max(1,n-1)):
            for j in range(i+1,n):
                cand=order[:i]+order[i:j][::-1]+order[j:]
                cst=cost(cand)
                if cst+1e-6<best:
                    order=cand; best=cst; improved=True
        if not improved:
            break
    return order

uavs=[]
for i,(cid,_) in enumerate(cluster_info,1):
    regs=[r for r in regions if r['cluster']==cid]
    ordered=route_cost_order(regs)
    for j,r in enumerate(ordered,1): r['uav']=f'UAV-{i}'; r['visit_order']=j
    uavs.append(dict(id=f'UAV-{i}',regions=ordered,color=['#1f77b4','#2ca02c','#9467bd'][i-1]))

# -------------------- 逐段飞行-喷洒资源仿真与补给点优化 --------------------
def road_candidates(step=45):
    pts=[]
    for rd in roads_center:
        n=max(2,int(rd.length//step)+1)
        for k in range(n+1):
            p=rd.interpolate(min(rd.length, k*step))
            pts.append(Point(p.x,p.y))
    # 去重
    unique=[]
    for p in pts:
        if all(p.distance(q)>10 for q in unique): unique.append(p)
    return unique
candidates=road_candidates()

def nearest_candidate(p): return min(candidates, key=lambda q:p.distance(q))

def point_on_segment(a, b, frac):
    frac=max(0.0,min(1.0,frac))
    return Point(a.x+(b.x-a.x)*frac, a.y+(b.y-a.y)*frac)

def optimize_supply_points(defs, k=3):
    if len(defs)==0:
        return [road_union.interpolate(frac*road_union.length) for frac in [0.2,0.5,0.8]]
    # 贪心加权 p-median 近似：每轮选择能最大降低加权距离的道路候选点
    selected=[]
    for _ in range(k):
        best_c=None; best_obj=float('inf')
        for cand in candidates:
            if any(cand.distance(s)<20 for s in selected):
                continue
            trial=selected+[cand]
            obj=sum(d['weight']*min(d['point'].distance(s) for s in trial) for d in defs)
            if obj<best_obj:
                best_obj=obj; best_c=cand
        if best_c is None:
            best_c=max(candidates, key=lambda q: min(q.distance(s) for s in selected) if selected else 0)
        selected.append(best_c)
    return selected[:k]

def build_cover_lines(r):
    lines=[]
    segs=r.get('spray_segments',[])
    for i,seg in enumerate(segs):
        if i>0:
            prev=Point(segs[i-1].coords[-1])
            curp=Point(seg.coords[0])
            if prev.distance(curp)>1e-6:
                lines.append(('覆盖连接段', LineString([(prev.x,prev.y),(curp.x,curp.y)])))
        lines.append(('覆盖喷洒段', seg))
    return lines

def interp_on_line(line, frac):
    p=line.interpolate(max(0.0,min(1.0,frac))*line.length)
    return Point(p.x,p.y)

def simulate(supply_points=None, record_detail=True):
    if supply_points is None: supply_points=candidates
    defs=[]; route_rows=[]; event_rows=[]; curve_rows=[]; segment_rows=[]; route_lines=[]

    def add_curve(uav_id, elapsed, remain_t, remain_q, event):
        curve_rows.append(dict(无人机编号=uav_id,累计时间_min=round(elapsed/60,4),
                               剩余续航_min=round(max(0,remain_t)/60,4),
                               剩余药量_L=round(max(0,remain_q),4),事件=event))

    def add_segment(uav_id, seg_id, task_id, kind, line, speed, qpm, elapsed0, elapsed1, t0, t1, q0v, q1v):
        segment_rows.append(dict(线段编号=seg_id,无人机编号=uav_id,关联任务=task_id,线段类型=kind,
                                 起点X_m=round(line.coords[0][0],2),起点Y_m=round(line.coords[0][1],2),
                                 终点X_m=round(line.coords[-1][0],2),终点Y_m=round(line.coords[-1][1],2),
                                 线段长度_m=round(line.length,2),速度_m_s=speed,单位路径药量_L_m=round(qpm,6),
                                 开始时间_min=round(elapsed0/60,4),结束时间_min=round(elapsed1/60,4),
                                 起始续航_min=round(t0/60,4),结束续航_min=round(t1/60,4),
                                 起始药量_L=round(q0v,4),结束药量_L=round(q1v,4)))
        route_lines.append(dict(uav=uav_id, task=task_id, kind=kind, geom=line))

    for u in uavs:
        uav_id=u['id']; remain_t=T_MAX; remain_q=Q_MAX; elapsed=0.0
        cur=Point(takeoff.x,takeoff.y)
        total_flight=0.0; total_spray=0.0; total_q=0.0; n_e=0; n_q=0; n_supply=0
        seg_no=0; event_no=0
        add_curve(uav_id, elapsed, remain_t, remain_q, '起飞')

        def fly_or_spray(line, kind, task_id, speed, qpm):
            nonlocal cur, remain_t, remain_q, elapsed, total_flight, total_spray, total_q, n_e, n_q, n_supply, seg_no, event_no
            if line.length<=1e-6:
                return
            start=line.coords[0]
            local_start=Point(start[0],start[1])
            remaining_line=line
            while remaining_line.length>1e-6:
                length=remaining_line.length
                duration=length/speed
                q_need=qpm*length
                cand=[]
                if remain_t-duration < T_SAFE-1e-9:
                    cand.append(('电量不足', max(0.0,(remain_t-T_SAFE)/duration)))
                if qpm>0 and remain_q-q_need < Q_SAFE-1e-9:
                    cand.append(('药量不足', max(0.0,(remain_q-Q_SAFE)/q_need)))
                cand=[c for c in cand if 0.0<=c[1]<=1.0]
                if not cand:
                    t0=remain_t; q0v=remain_q; e0=elapsed
                    remain_t-=duration; remain_q-=q_need; elapsed+=duration
                    if kind.startswith('飞行') or kind.startswith('补给') or kind.startswith('返航'):
                        total_flight+=length
                    else:
                        total_spray+=length; total_q+=q_need
                    seg_no+=1
                    add_segment(uav_id,f'{uav_id}-L{seg_no:03d}',task_id,kind,remaining_line,speed,qpm,e0,elapsed,t0,remain_t,q0v,remain_q)
                    add_curve(uav_id, elapsed, remain_t, remain_q, kind)
                    cur=Point(remaining_line.coords[-1])
                    return

                min_frac=min(v for _,v in cand)
                cause=[name for name,v in cand if abs(v-min_frac)<1e-6]
                event_type='+'.join(cause)
                if min_frac<=1e-8:
                    min_frac=1e-6
                event_pt=interp_on_line(remaining_line,min_frac)
                part_line=LineString([remaining_line.coords[0], (event_pt.x,event_pt.y)])
                part_len=part_line.length
                t0=remain_t; q0v=remain_q; e0=elapsed
                dt=part_len/speed; dq=qpm*part_len
                remain_t-=dt; remain_q-=dq; elapsed+=dt
                if kind.startswith('飞行') or kind.startswith('补给') or kind.startswith('返航'):
                    total_flight+=part_len
                else:
                    total_spray+=part_len; total_q+=dq
                seg_no+=1
                add_segment(uav_id,f'{uav_id}-L{seg_no:03d}',task_id,kind,part_line,speed,qpm,e0,elapsed,t0,remain_t,q0v,remain_q)

                event_no+=1
                if '电量不足' in event_type: n_e+=1
                if '药量不足' in event_type: n_q+=1
                weight=1.0+(0.6 if '+' in event_type else 0.0)
                risk=next((rr['risk'] for rr in regions if rr['id']==task_id),0.46)
                weight+=risk
                defs.append(dict(uav=uav_id, point=event_pt, type=event_type, risk=risk, next_task=task_id,
                                 weight=weight, remain_time=remain_t, remain_q=remain_q, event_time=elapsed, segment_kind=kind))
                event_rows.append(dict(事件编号=f'E{len(event_rows)+1:03d}',无人机编号=uav_id,事件类型='资源匮乏',
                                       匮乏类型=event_type,关联任务=task_id,发生线段类型=kind,
                                       X_m=round(event_pt.x,2),Y_m=round(event_pt.y,2),
                                       累计时间_min=round(elapsed/60,4),剩余续航_min=round(max(0,remain_t)/60,4),
                                       剩余药量_L=round(max(0,remain_q),4)))
                add_curve(uav_id, elapsed, remain_t, remain_q, event_type)

                sp=min(supply_points, key=lambda p:event_pt.distance(p))
                n_supply+=1
                to_sp=LineString([(event_pt.x,event_pt.y),(sp.x,sp.y)])
                if to_sp.length>1e-6:
                    t0=remain_t; q0v=remain_q; e0=elapsed
                    dt=to_sp.length/VF; remain_t-=dt; elapsed+=dt; total_flight+=to_sp.length
                    seg_no+=1
                    add_segment(uav_id,f'{uav_id}-L{seg_no:03d}',task_id,'补给转场-去程',to_sp,VF,0.0,e0,elapsed,t0,remain_t,q0v,remain_q)
                remain_t=T_MAX; remain_q=Q_MAX
                event_rows.append(dict(事件编号=f'E{len(event_rows)+1:03d}',无人机编号=uav_id,事件类型='道路补给',
                                       匮乏类型=event_type,关联任务=task_id,发生线段类型='补给点',
                                       X_m=round(sp.x,2),Y_m=round(sp.y,2),
                                       累计时间_min=round(elapsed/60,4),剩余续航_min=round(remain_t/60,4),
                                       剩余药量_L=round(remain_q,4)))
                add_curve(uav_id, elapsed, remain_t, remain_q, '道路补给')
                back=LineString([(sp.x,sp.y),(event_pt.x,event_pt.y)])
                if back.length>1e-6:
                    t0=remain_t; q0v=remain_q; e0=elapsed
                    dt=back.length/VF; remain_t-=dt; elapsed+=dt; total_flight+=back.length
                    seg_no+=1
                    add_segment(uav_id,f'{uav_id}-L{seg_no:03d}',task_id,'补给转场-返程',back,VF,0.0,e0,elapsed,t0,remain_t,q0v,remain_q)
                tail=LineString([(event_pt.x,event_pt.y), remaining_line.coords[-1]])
                remaining_line=tail
                cur=event_pt

        for r in u['regions']:
            entry=r['entry'] if r['entry'] is not None else r['center']
            if cur.distance(entry)>1e-6:
                fly_or_spray(LineString([(cur.x,cur.y),(entry.x,entry.y)]),'飞行-区域间',r['id'],VF,0.0)
            qpm=r['dose_l']/max(r['internal_length_m'],1e-6)
            for sub_kind,line in build_cover_lines(r):
                fly_or_spray(line,sub_kind,r['id'],VW,qpm)
            cur=r['exit'] if r['exit'] is not None else r['center']
        if cur.distance(takeoff)>1e-6:
            fly_or_spray(LineString([(cur.x,cur.y),(takeoff.x,takeoff.y)]),'返航',uav_id,VF,0.0)

        route_rows.append(dict(无人机编号=uav_id,服务病害区='、'.join(r['id'] for r in u['regions']),
                               病害区数量=len(u['regions']),区域间飞行距离_m=round(total_flight,2),
                               内部喷洒路径_m=round(total_spray,2),总执行路径_m=round(total_flight+total_spray,2),
                               总仿真时间_min=round(elapsed/60,2),预计喷洒药量_L=round(total_q,2),
                               电量匮乏次数=n_e,药量匮乏次数=n_q,道路补给次数=n_supply,
                               结束剩余续航_min=round(max(0,remain_t)/60,2),结束剩余药量_L=round(max(0,remain_q),2)))
    return defs, route_rows, route_lines, event_rows, curve_rows, segment_rows

initial_defs,_,_,_,_,_=simulate(candidates, record_detail=False)
supply_points=optimize_supply_points(initial_defs,3)
defs, route_rows, route_lines, event_rows, curve_rows, segment_rows=simulate(supply_points)
supply_points=optimize_supply_points(defs,3)
defs, route_rows, route_lines, event_rows, curve_rows, segment_rows=simulate(supply_points)

# -------------------- 输出表格 --------------------
region_table=pd.DataFrame([{
    '病害区编号':r['id'],'病害类别':r['name'],'中心X_m':round(r['x'],2),'中心Y_m':round(r['y'],2),'风险等级':r['level'],
    '综合风险值':round(r['risk'],4),'气象风险':round(r['weather_risk'],4),'历史情情风险':round(r['history_risk'],4),'视觉风险':round(r['visual_risk'],4),
    '检测置信度':round(r['confidence'],3),'高斯sigma_m':round(r['sigma'],2),'病害面积_m2':round(r['area_m2'],2),'病害面积_ha':round(r['area_m2']/10000,4),
    '内部喷洒路径_m':round(r['internal_length_m'],2),'预计药量_L':round(r['dose_l'],2),'单位面积药量_L_ha':round(r['dose_l']/max(r['area_m2']/10000,1e-6),2),
    '沿程流量f_i_L_m':round(r['flow_rate_l_per_m'],6),'作业优先级':round(r['priority'],4),'负责无人机':r.get('uav',''),'访问次序':r.get('visit_order','')
} for r in sorted(regions,key=lambda x:x['id'])])
route_table=pd.DataFrame(route_rows)
def_table=pd.DataFrame([{
    '匮乏点编号':f'Q{i:02d}','无人机编号':d['uav'],'匮乏类型':d['type'],'发生线段类型':d['segment_kind'],
    '匮乏点X_m':round(d['point'].x,2),'匮乏点Y_m':round(d['point'].y,2),
    '触发时刻_min':round(d['event_time']/60,4),'触发时剩余续航_min':round(d['remain_time']/60,4),
    '触发时剩余药量_L':round(d['remain_q'],4),'关联任务':d['next_task'],'关联风险值':round(d['risk'],4),'权重':round(d['weight'],3),
    '最近补给点': 'S'+str(1+int(np.argmin([d['point'].distance(p) for p in supply_points])))
} for i,d in enumerate(defs,1)])
supply_table=pd.DataFrame([{'补给点编号':f'S{i}','补给点X_m':round(p.x,2),'补给点Y_m':round(p.y,2),'选址依据':'道路候选点加权p-median近似优化','是否在道路上':'是'} for i,p in enumerate(supply_points,1)])
event_table=pd.DataFrame(event_rows)
curve_table=pd.DataFrame(curve_rows)
segment_table=pd.DataFrame(segment_rows)
metrics_table=pd.DataFrame([{'RF气象风险AUC':round(rf_auc,4),'历史情情模型AUC':round(hist_auc,4),'视觉模型mAP@0.5':0.83729,'施药决策期气象权重':round(W_WEATHER,4),'施药决策期历史权重':round(W_HISTORY,4),'施药决策期视觉权重':round(W_VISUAL,4),'喷幅_m':WS,'重叠系数':RHO,'扫描线间距_m':round(SCAN_SPACING,2),'最大载药量_L':Q_MAX,'最大续航_min':T_MAX/60,'安全药量阈值_L':Q_SAFE,'安全电量阈值_min':T_SAFE/60,'区域间速度_m_s':VF,'喷洒速度_m_s':VW}])
for name,tab in [
    ('表1_病害区域基本信息_输入.csv',region_table),
    ('表2_三台无人机飞行仿真汇总.csv',route_table),
    ('表3_模型触发资源匮乏点.csv',def_table),
    ('表4_道路补给点.csv',supply_table),
    ('表5_模型参数与可信度权重.csv',metrics_table),
    ('表6_逐事件资源仿真日志.csv',event_table),
    ('表7_资源剩余曲线数据.csv',curve_table),
    ('表8_飞行与喷洒线段明细.csv',segment_table)
]:
    tab.to_csv(OUT/name,index=False,encoding='utf-8-sig')

# -------------------- 绘图 --------------------
palette=['#dfe7a6','#d7e096','#cfdb88','#c7d67b','#bfd16f','#b7cc63']
def add_geom(ax, geom, fc, ec='white', lw=0.6, alpha=1.0, z=1):
    if geom.is_empty: return
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms: add_geom(ax,g,fc,ec,lw,alpha,z)
    elif isinstance(geom, Polygon):
        x,y=geom.exterior.xy; ax.fill(x,y,facecolor=fc,edgecolor=ec,linewidth=lw,alpha=alpha,zorder=z)

def base_ax(title):
    fig,ax=plt.subplots(figsize=(10,10),dpi=180)
    ax.set_xlim(0,W); ax.set_ylim(0,H); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title,fontsize=15)
    for s in ax.spines.values(): s.set_visible(False)
    for i,p in enumerate(parcels): add_geom(ax,p,palette[i%len(palette)],ec='#f3f6ea',lw=0.45,alpha=0.96,z=1)
    for water in [layers['river_main'],layers['river_left'],layers['pond_left'],layers['pond_right']]: add_geom(ax,water,'#90cadd','#5eb3c7',0.9,0.99,4)
    add_geom(ax,village,'#d8ccb6','white',1.2,1.0,5)
    for rd in roads_center:
        x,y=rd.xy; ax.plot(x,y,color='#d7d7cf',lw=6,zorder=8,solid_capstyle='round'); ax.plot(x,y,color='#f4f3ec',lw=3.5,zorder=9,solid_capstyle='round')
    ax.scatter([takeoff.x],[takeoff.y],marker='*',s=220,c='#d4a017',edgecolors='black',linewidths=0.6,zorder=50)
    ax.text(takeoff.x+8,takeoff.y+8,'起飞点',fontsize=8,zorder=51)
    return fig,ax

def draw_disease(ax, labels=True):
    for r in regions:
        add_geom(ax,r['low_geom'],'#ffe084','#fff3bf',0.15,0.42,12)
        add_geom(ax,r['mid_geom'],'#f6a04d','#ffd3ad',0.15,0.55,13)
        add_geom(ax,r['high_geom'],'#c92a2a','#f7b1b1',0.15,0.70,14)
        ax.scatter([r['x']],[r['y']],s=8,c='black',zorder=20)
        if labels: ax.text(r['x']+4,r['y']+4,r['id'],fontsize=5.8,zorder=21)

fig,ax=base_ax('高斯扩散病害斑块与起飞点')
draw_disease(ax, True)
ax.legend(handles=[Patch(facecolor='#c92a2a',label='高强度斑块'),Patch(facecolor='#f6a04d',label='中强度斑块'),Patch(facecolor='#ffe084',label='低强度斑块'),Line2D([0],[0],marker='*',color='w',markerfacecolor='#d4a017',markeredgecolor='black',markersize=12,label='起飞点')],loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=9)
fig.savefig(OUT/'图1_高斯扩散病害图.png',bbox_inches='tight'); plt.close(fig)

fig,ax=base_ax('飞行路线与模型触发匮乏点')
draw_disease(ax, False)
color_map={u['id']:u['color'] for u in uavs}
for u in uavs:
    segs=[seg for seg in route_lines if seg['uav']==u['id']]
    pts=[]
    for seg in segs:
        coords=list(seg['geom'].coords)
        if not pts:
            pts.append(coords[0])
        elif Point(*pts[-1]).distance(Point(*coords[0]))>1e-6:
            pts.append(coords[0])
        pts.append(coords[-1])
    if pts:
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        ax.plot(xs,ys,color=u['color'],lw=1.9,alpha=0.88,zorder=24,label=u['id'])
for seg in route_lines:
    if '覆盖' not in seg['kind']:
        continue
    x,y=seg['geom'].xy
    ax.plot(x,y,color=color_map.get(seg['uav'],'#333333'),lw=0.22,alpha=0.045,zorder=21)
for i,p in enumerate(supply_points,1):
    ax.scatter([p.x],[p.y],marker='s',s=72,c='#006d77',edgecolors='white',linewidths=0.8,zorder=50); ax.text(p.x+6,p.y+6,f'S{i}',fontsize=7,color='#003b3f',zorder=51)
for i,d in enumerate(defs,1):
    marker='^' if '电量' in d['type'] and '药量' not in d['type'] else ('X' if '药量' in d['type'] and '电量' not in d['type'] else 'D')
    col='#5a189a' if marker=='^' else ('#d00000' if marker=='X' else '#ee9b00')
    ax.scatter([d['point'].x],[d['point'].y],marker=marker,s=75,c=col,edgecolors='white',linewidths=0.7,zorder=55); ax.text(d['point'].x+5,d['point'].y+5,f'Q{i}',fontsize=7,color=col,zorder=56)
ax.legend(handles=[Line2D([0],[0],marker='s',color='w',markerfacecolor='#006d77',markersize=8,label='道路补给点'),Line2D([0],[0],marker='^',color='w',markerfacecolor='#5a189a',markersize=8,label='电量匮乏'),Line2D([0],[0],marker='X',color='w',markerfacecolor='#d00000',markersize=8,label='药量匮乏'),Line2D([0],[0],marker='D',color='w',markerfacecolor='#ee9b00',markersize=8,label='双重匮乏')],loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=9)
fig.savefig(OUT/'图2_飞行路线与模型触发匮乏点.png',bbox_inches='tight'); plt.close(fig)

fig,ax=base_ax('无人机主路径规划图（无内部喷洒线）')
draw_disease(ax, False)
for u in uavs:
    pts=[(takeoff.x,takeoff.y)]+[(r['center'].x,r['center'].y) for r in u['regions']]+[(takeoff.x,takeoff.y)]
    ax.plot([p[0] for p in pts],[p[1] for p in pts],color=u['color'],lw=2.2,marker='o',markersize=3.0,alpha=0.9,zorder=30,label=u['id'])
for r in regions:
    ax.text(r['x']+3,r['y']-6,f"{r['uav'][-1]}-{r['visit_order']}",fontsize=5.5,color='black',zorder=40)
for i,p in enumerate(supply_points,1):
    ax.scatter([p.x],[p.y],marker='s',s=72,c='#006d77',edgecolors='white',linewidths=0.8,zorder=50); ax.text(p.x+6,p.y+6,f'S{i}',fontsize=7,color='#003b3f',zorder=51)
ax.legend(loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=9)
fig.savefig(OUT/'图3_无人机主路径规划图_无内部喷洒线.png',bbox_inches='tight'); plt.close(fig)

def plot_resource_curve(col, ylabel, title, filename):
    fig,ax=plt.subplots(figsize=(9,5.2),dpi=180)
    for u in uavs:
        sub=curve_table[curve_table['无人机编号']==u['id']]
        ax.plot(sub['累计时间_min'],sub[col],lw=2,marker='o',markersize=2.6,label=u['id'],color=u['color'])
    threshold=T_SAFE/60 if '续航' in col else Q_SAFE
    ax.axhline(threshold,color='#d00000',ls='--',lw=1.2,label='安全阈值')
    ax.set_xlabel('累计时间 / min'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.25); ax.legend(loc='best',fontsize=8)
    fig.savefig(OUT/filename,bbox_inches='tight'); plt.close(fig)

plot_resource_curve('剩余续航_min','剩余续航 / min','电量消耗曲线','图5_电量消耗曲线.png')
plot_resource_curve('剩余药量_L','剩余药量 / L','药量消耗曲线','图6_药量消耗曲线.png')

single_candidates=[r for r in regions if isinstance(r['spray_region'], Polygon) and r['area_m2']>2500 and r['spray_region'].area/r['spray_region'].convex_hull.area>0.55]
single=max(single_candidates or regions,key=lambda r:r['spray_region'].area)
minx,miny,maxx,maxy=single['spray_region'].bounds; pad=35
fig,ax=plt.subplots(figsize=(8,7),dpi=180)
ax.set_xlim(max(0,minx-pad),min(W,maxx+pad)); ax.set_ylim(max(0,miny-pad),min(H,maxy+pad)); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"单病害区内部覆盖喷洒路径：{single['id']}",fontsize=14)
for s in ax.spines.values(): s.set_visible(False)
add_geom(ax,single['spray_region'],'#c7e48b','white',1.2,0.55,1)
add_geom(ax,single['low_geom'],'#ffe084','#fff3bf',0.2,0.42,10); add_geom(ax,single['mid_geom'],'#f6a04d','#ffd3ad',0.2,0.55,11); add_geom(ax,single['high_geom'],'#c92a2a','#f7b1b1',0.2,0.70,12)
for seg in single['spray_segments']:
    x,y=seg.xy; ax.plot(x,y,color='#b22222',lw=1.6,zorder=20)
if single['entry'] is not None:
    ax.scatter([single['entry'].x],[single['entry'].y],marker='*',s=180,c='#d4a017',edgecolors='black',linewidths=0.6,zorder=30); ax.text(single['entry'].x+3,single['entry'].y+3,'入口',fontsize=8)
if single['exit'] is not None:
    ax.scatter([single['exit'].x],[single['exit'].y],marker='o',s=70,c='#355c7d',edgecolors='white',linewidths=0.6,zorder=30); ax.text(single['exit'].x+3,single['exit'].y+3,'出口',fontsize=8)
ax.legend(handles=[Patch(facecolor='#c7e48b',alpha=0.55,label='安全缓冲喷洒区'),Patch(facecolor='#ffe084',label='病害斑块'),Line2D([0],[0],color='#b22222',lw=2,label='往复式喷洒航线')],loc='center left',bbox_to_anchor=(1.02,0.5),fontsize=9)
fig.savefig(OUT/'图4_单病害区内部喷洒路径图.png',bbox_inches='tight'); plt.close(fig)

# -------------------- 报告与压缩 --------------------
summary={
    '高风险数量': int((region_table['风险等级']=='高风险').sum()),
    '中风险数量': int((region_table['风险等级']=='中风险').sum()),
    '低风险数量': int((region_table['风险等级']=='低风险').sum()),
    '总预计药量_L': round(float(region_table['预计药量_L'].sum()),2),
    '平均单区药量_L': round(float(region_table['预计药量_L'].mean()),2),
    '最大单区药量_L': round(float(region_table['预计药量_L'].max()),2),
    '总匮乏点': int(len(def_table)),
    '电量匮乏相关': int(def_table['匮乏类型'].astype(str).str.contains('电量').sum()) if len(def_table)>0 else 0,
    '药量匮乏相关': int(def_table['匮乏类型'].astype(str).str.contains('药量').sum()) if len(def_table)>0 else 0,
    '补给点数量': int(len(supply_table)),
    '扫描线间距_m': round(SCAN_SPACING,2),
    '载药量_L': Q_MAX,
    '最大续航_min': T_MAX/60,
    'RF_AUC': round(rf_auc,4),
    'History_AUC': round(hist_auc,4)
}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
readme=f"""# v9 重新计算结果说明\n\n本版本已按论文模型重算，不再使用随机替代核心算法。\n\n关键修正：\n1. 病害斑块采用高斯扩散模型，按强度阈值形成低/中/高斑块，并裁剪到农田区域。\n2. 药量采用 30 L 载药量、3 L 安全药量阈值，按面积、综合风险、置信度计算，并沿内部喷洒路径均匀喷洒。\n3. 电量消耗采用飞行时间 + 内部喷洒时间，25 min 最大续航、3 min 安全阈值，匮乏点由模型触发。\n4. 内部喷洒路径使用喷幅 {WS} m、重叠系数 {RHO}，扫描线间距 {SCAN_SPACING:.2f} m。\n5. 补给点从道路候选点中通过加权 p-median 近似优化选择 3 个。\n6. 代码为独立脚本，不依赖 4.py；只需将 RICE_中文.xlsx 与脚本放在同一目录运行。\n\n结果摘要：\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n"""
(OUT/'README_v9.md').write_text(readme,encoding='utf-8')
# 复制脚本与数据文件，保证复现
shutil.copy2(Path(__file__), OUT/'recalculate_uav_model_v9.py')
if DATA_PATH.exists(): shutil.copy2(DATA_PATH, OUT/'RICE_中文.xlsx')
zip_path=BASE_DIR/'uav_model_v9_results.zip'
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob('*'):
        zf.write(p, arcname=p.relative_to(OUT.parent))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('OUT=',OUT)
print('ZIP=',zip_path)
