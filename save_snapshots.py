"""
save_snapshots.py — 애니메이션 키 타임스텝을 3D PNG로 저장한다.

실행:
    python3 save_snapshots.py

출력:
    snapshots/  — 각 타임스텝별 MPPI vs MPPI+DOB 나란히 비교 3D PNG
"""
import os, sys, math, types

import matplotlib
matplotlib.use('Agg')

_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    '/home/economy02/.local/lib/python3.10/site-packages',
    '/home/economy02/pytorch_mppi/src',
    '/home/economy02/mpc/acados/interfaces/acados_template',
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MTK_DIR = '/home/economy02/.local/lib/python3.10/site-packages/mpl_toolkits'
try:
    import mpl_toolkits as _mtk
except ModuleNotFoundError:
    _mtk = types.ModuleType('mpl_toolkits')
    _mtk.__path__ = []
    sys.modules['mpl_toolkits'] = _mtk
if os.path.isdir(_MTK_DIR) and _MTK_DIR not in _mtk.__path__:
    _mtk.__path__.insert(0, _MTK_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, _DIR)
import run_comparison as rc

N_STEPS = rc.N_STEPS
DT      = rc.DT
LABELS  = rc.LABELS
COLORS  = rc.COLORS

SNAP_DIR = os.path.join(_DIR, 'snapshots')
os.makedirs(SNAP_DIR, exist_ok=True)

# 키 타임스텝 (초)
KEY_TIMES = [0.0, 2.5, 5.0, 7.5, 9.8]

# 3D 뷰 앵글
ELEV, AZIM = 22, -55

_LINK_COLORS = ['#003f8a', '#1560bd', '#2878d8', '#4a94e8', '#70b0f0', '#98ccff']
_TH = np.linspace(0, 2 * math.pi, 60)
_LAT_OFFS = [-math.pi/3, -math.pi/6, 0.0, math.pi/6, math.pi/3]
_LON_OFFS = [0.0, math.pi/4, math.pi/2, 3*math.pi/4]
_OBS_COLORS = ['#ff3300', '#0044ff']
_TR_CMAP = plt.get_cmap('plasma')


def load_results():
    results = {}
    for key in ('MPPI_CROSS', 'MPPI_DOB_CROSS'):
        path = rc.RESULT_FILES[key]
        if not os.path.exists(path):
            print(f'[SKIP] {path} 없음'); continue
        d = dict(np.load(path, allow_pickle=True))
        for k, v in d.items():
            if v.dtype == object:
                d[k] = v.item()
        results[key] = d
        print(f'[LOAD] {key}')
    return results


def make_waypoints():
    t  = np.linspace(0, 2 * math.pi, N_STEPS + 1)[:-1]
    cx, cy, cz = rc.TRAJ_CENTER
    r = rc.TRAJ_RADIUS
    return np.stack([cx + r*np.cos(t), cy + r*np.sin(t), np.full(N_STEPS, cz)], axis=1)


def _draw_sphere(ax, cx, cy, cz, r, color, alpha=0.75):
    """위선 + 경선 + 중심점으로 구체를 그린다."""
    for lat in _LAT_OFFS:
        r_lat = r * math.cos(lat)
        z_lat = cz + r * math.sin(lat)
        ax.plot(cx + r_lat*np.cos(_TH), cy + r_lat*np.sin(_TH),
                np.full(len(_TH), z_lat), '-', color=color, lw=1.8, alpha=alpha, zorder=8)
    for lon in _LON_OFFS:
        ax.plot(cx + r*np.cos(_TH)*math.cos(lon),
                cy + r*np.cos(_TH)*math.sin(lon),
                cz + r*np.sin(_TH), '-', color=color, lw=1.8, alpha=alpha, zorder=8)
    ax.plot([cx], [cy], [cz], 'o', color=color, ms=9, alpha=1.0,
            markeredgecolor='white', markeredgewidth=1.0, zorder=9)


def _draw_arm(ax, q, highlight_color):
    """UR5 팔 링크를 두꺼운 선으로 그린다."""
    pos = rc.fk_joints_np(q)  # (7, 3)
    for j in range(6):
        p0, p1 = pos[j], pos[j+1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                '-', color=_LINK_COLORS[j], lw=5.5, solid_capstyle='round', zorder=7)
    # 관절 마커
    ax.plot(pos[:-1, 0], pos[:-1, 1], pos[:-1, 2],
            'o', color='#cc2200', ms=6, zorder=10)
    # 엔드이펙터
    ax.plot([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]],
            'D', color=highlight_color, ms=10, markeredgecolor='white',
            markeredgewidth=1.2, zorder=11)


def render_snapshot(ax, key, data, waypoints, frame, cross_paths):
    """하나의 3D ax에 특정 프레임의 상태를 그린다."""
    t_sec = frame * DT

    # ── 배경 설정
    ax.set_facecolor('white')
    ax.set_xlim(0.18, 0.62); ax.set_ylim(-0.22, 0.22); ax.set_zlim(0.13, 0.57)
    ax.set_xlabel('X [m]', fontsize=8, labelpad=2)
    ax.set_ylabel('Y [m]', fontsize=8, labelpad=2)
    ax.set_zlabel('Z [m]', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#cccccc')
    ax.grid(True, color='#dddddd', linewidth=0.4)
    ax.view_init(elev=25, azim=-55)

    # ── 레퍼런스 궤적 (점선)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
            '--', color='#33cc66', lw=1.2, alpha=0.45, zorder=2)

    # ── EE 궤적 흔적 (지나온 경로, plasma colormap)
    ee = data['ee']
    n_trace = min(frame + 1, len(ee))
    if n_trace >= 2:
        for i in range(1, n_trace):
            c = _TR_CMAP(i / max(N_STEPS, 1))
            ax.plot(ee[i-1:i+1, 0], ee[i-1:i+1, 1], ee[i-1:i+1, 2],
                    '-', color=c, lw=2.2, alpha=0.85, zorder=5)

    # ── MPPI 샘플 궤적 (해당 프레임, 최대 80개)
    if frame > 0:
        es_all = data.get('ee_samples')
        if es_all is not None and frame <= len(es_all):
            fi  = min(frame - 1, len(es_all) - 1)
            es  = es_all[fi]          # (K_vis, T_h, 3)
            K_d = min(80, es.shape[0])
            T_h = es.shape[1]
            seg = T_h + 1
            xs  = np.full(K_d * seg, np.nan)
            ys  = xs.copy(); zs = xs.copy()
            for i in range(3, K_d):
                b = i * seg
                xs[b:b+T_h] = es[i, :, 0]
                ys[b:b+T_h] = es[i, :, 1]
                zs[b:b+T_h] = es[i, :, 2]
            ax.plot(xs, ys, zs, '-', color='#aaaacc', lw=0.7, alpha=0.45, zorder=3)
            # 상위 3개 (노란색)
            xt = np.full(3 * seg, np.nan)
            yt = xt.copy(); zt = xt.copy()
            for i in range(min(3, K_d)):
                b = i * seg
                xt[b:b+T_h] = es[i, :, 0]
                yt[b:b+T_h] = es[i, :, 1]
                zt[b:b+T_h] = es[i, :, 2]
            ax.plot(xt, yt, zt, '-', color='#ffdd00', lw=1.6, alpha=0.88, zorder=4)

    # ── 장애물 구체
    if cross_paths is not None:
        fi = min(frame, cross_paths.shape[1] - 1)
        for i, clr in enumerate(_OBS_COLORS):
            cx, cy, cz = cross_paths[i, fi]
            _draw_sphere(ax, cx, cy, cz, rc.CROSS_RADIUS, clr, alpha=0.80)
            # 안전 마진 (점선 적도)
            r_safe = rc.CROSS_RADIUS + getattr(rc, '_CROSS_MARGIN', 0.005)
            ax.plot(cx + r_safe*np.cos(_TH), cy + r_safe*np.sin(_TH),
                    np.full(len(_TH), cz), '--', color=clr, lw=0.8, alpha=0.30, zorder=7)

    # ── UR5 팔
    if frame < len(data['q']):
        _draw_arm(ax, data['q'][frame], COLORS[key])

    # ── 제목
    err_now = data['err_mm'][frame-1] if frame > 0 else 0.0
    cs = data.get('collision_steps', np.array([]))
    coll_tag = '  ⚠ COLLISION' if frame in set(cs.tolist()) else ''
    ax.set_title(
        f'{LABELS[key]}\nt = {t_sec:.1f} s   err = {err_now:.1f} mm{coll_tag}',
        fontsize=10, color=COLORS[key], fontweight='bold', pad=6,
    )


def save_snapshot_at(results, waypoints, cross_paths, t_sec, label):
    frame = min(int(round(t_sec / DT)), N_STEPS - 1)
    keys  = list(results.keys())
    n     = len(keys)

    fig = plt.figure(figsize=(9 * n, 9), facecolor='white')
    fig.suptitle(
        f'UR5 MPPI vs MPPI+DOB  —  t = {t_sec:.1f} s\n'
        'Circle trajectory  |  2 crossing dynamic spheres  |  sinusoidal disturbance ON',
        fontsize=13, color='white', fontweight='bold', y=0.97,
    )

    for i, key in enumerate(keys):
        ax = fig.add_subplot(1, n, i+1, projection='3d')
        render_snapshot(ax, key, results[key], waypoints, frame, cross_paths)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(SNAP_DIR, f'snapshot_{label}.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved  {out}')


def save_all_snapshots_strip(results, waypoints, cross_paths):
    """모든 키 타임을 세로로 이어붙인 스트립 PNG."""
    keys  = list(results.keys())
    n_col = len(keys)
    n_row = len(KEY_TIMES)

    fig = plt.figure(figsize=(9 * n_col, 8 * n_row), facecolor='white')
    fig.suptitle(
        'UR5 MPPI vs MPPI+DOB  —  Simulation Snapshots\n'
        'Circle trajectory  |  2 crossing dynamic spheres  |  sinusoidal disturbance ON',
        fontsize=16, color='white', fontweight='bold', y=0.995,
    )
    gs = gridspec.GridSpec(n_row, n_col, hspace=0.10, wspace=0.05,
                           top=0.985, bottom=0.005, left=0.005, right=0.995)

    for row, t_sec in enumerate(KEY_TIMES):
        frame = min(int(round(t_sec / DT)), N_STEPS - 1)
        for col, key in enumerate(keys):
            ax = fig.add_subplot(gs[row, col], projection='3d')
            render_snapshot(ax, key, results[key], waypoints, frame, cross_paths)

    out = os.path.join(SNAP_DIR, 'snapshot_strip_all.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  saved  {out}')


if __name__ == '__main__':
    print('=' * 58)
    print('save_snapshots.py — 3D 실시간 스냅샷 PNG 생성')
    print('=' * 58)

    results   = load_results()
    if not results:
        sys.exit(1)
    waypoints = make_waypoints()

    first = next(iter(results.values()))
    cross_paths = (first['cross_paths'] if 'cross_paths' in first
                   else rc._crossing_obstacle_paths(N_STEPS + 1))

    print(f'\n키 타임스텝 {KEY_TIMES} 에서 개별 PNG 생성...')
    labels = ['t0p0', 't2p5', 't5p0', 't7p5', 't9p8']
    for t_sec, lbl in zip(KEY_TIMES, labels):
        save_snapshot_at(results, waypoints, cross_paths, t_sec, lbl)

    print('\n전체 스트립 PNG 생성...')
    save_all_snapshots_strip(results, waypoints, cross_paths)

    print('\n' + '=' * 58)
    print(f'완료  →  {SNAP_DIR}/')
    print('=' * 58)
