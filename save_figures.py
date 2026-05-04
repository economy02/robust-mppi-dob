"""
save_figures.py — 시뮬 결과(npz)로부터 논문용·발표용 PNG와 MP4를 저장한다.

실행:
    python3 save_figures.py

출력:
    paper_figures/        — 논문용 PNG (300 DPI)
    presentation_figures/ — 발표용 PNG (200 DPI, 큰 폰트)
    videos/               — MP4 애니메이션 (25 fps)
"""
import os, sys, math, types

# ── matplotlib backend: 디스플레이 없이 파일 저장 ────────────────────────────
import matplotlib
matplotlib.use('Agg')

# ── 경로 설정 ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in [
    '/home/economy02/.local/lib/python3.10/site-packages',
    '/home/economy02/pytorch_mppi/src',
    '/home/economy02/mpc/acados/interfaces/acados_template',
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── mpl_toolkits 버전 충돌 방지 (시스템/로컬 matplotlib 혼재 환경) ──────────
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
import matplotlib.animation as animation
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── run_comparison 임포트 ─────────────────────────────────────────────────────
sys.path.insert(0, _DIR)
import run_comparison as rc

N_STEPS = rc.N_STEPS
DT      = rc.DT
LABELS  = rc.LABELS
COLORS  = rc.COLORS

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
def _set_korean_font():
    for path in [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]:
        if os.path.exists(path):
            matplotlib.rcParams['font.family'] = fm.FontProperties(fname=path).get_name()
            return
_set_korean_font()

# ── 출력 폴더 ────────────────────────────────────────────────────────────────
PAPER_DIR = os.path.join(_DIR, 'paper_figures')
PRES_DIR  = os.path.join(_DIR, 'presentation_figures')
VIDEO_DIR = os.path.join(_DIR, 'videos')
for d in (PAPER_DIR, PRES_DIR, VIDEO_DIR):
    os.makedirs(d, exist_ok=True)

PRES_RC = {
    'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12,
    'lines.linewidth': 2.5,
}

def _save(fig, stem, paper_dpi=300, pres_dpi=200):
    for path, dpi in [(os.path.join(PAPER_DIR, f'{stem}.png'), paper_dpi),
                      (os.path.join(PRES_DIR,  f'{stem}.png'), pres_dpi)]:
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'  saved  {path}')
    plt.close(fig)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_results():
    results = {}
    for key in ('MPPI_CROSS', 'MPPI_DOB_CROSS'):
        path = rc.RESULT_FILES[key]
        if not os.path.exists(path):
            print(f'[SKIP] {path} 없음')
            continue
        d = dict(np.load(path, allow_pickle=True))
        for k, v in d.items():
            if v.dtype == object:
                d[k] = v.item()
        results[key] = d
        print(f'[LOAD] {key}  ({os.path.getsize(path)//1_000_000} MB)')
    return results

def make_waypoints():
    t  = np.linspace(0, 2 * math.pi, N_STEPS + 1)[:-1]
    cx, cy, cz = rc.TRAJ_CENTER
    r  = rc.TRAJ_RADIUS
    return np.stack([cx + r*np.cos(t), cy + r*np.sin(t), np.full(N_STEPS, cz)], axis=1)


# ════════════════════════════════════════════════════════
# PNG 그림들
# ════════════════════════════════════════════════════════
def save_metric_dashboard(results, waypoints):
    print('\n[FIG1] Metric Dashboard')
    metrics = {k: rc.compute_metrics(v, waypoints) for k, v in results.items()}
    with matplotlib.rc_context(PRES_RC):
        fig = rc.plot_metric_dashboard(metrics)
    _save(fig, 'metric_dashboard')

def save_timeseries(results, waypoints):
    print('\n[FIG2] Time Series')
    with matplotlib.rc_context(PRES_RC):
        fig = rc.plot_timeseries(results, waypoints)
    _save(fig, 'timeseries')

def save_3d_trajectory(results, waypoints):
    print('\n[FIG3] 3D Trajectory')
    with matplotlib.rc_context(PRES_RC):
        fig = rc.plot_3d_trajectory(results, waypoints, show_obstacles=False)
    _save(fig, '3d_trajectory')

def save_cross_comparison(results, waypoints):
    print('\n[FIG4] Cross Comparison')
    with matplotlib.rc_context(PRES_RC):
        fig = rc.plot_cross_comparison(results, waypoints)
    if fig is not None:
        _save(fig, 'cross_comparison')

def save_individual_trajectories(results, waypoints):
    """컨트롤러별 XY 전체 궤적 단독 그림."""
    print('\n[FIG5] Individual Full Trajectories')
    for key, data in results.items():
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
        ax.plot(waypoints[:, 0], waypoints[:, 1],
                '--', color='#008844', lw=2.0, alpha=0.6, label='Reference')
        ee = data['ee']
        ax.plot(ee[:, 0], ee[:, 1],
                color=COLORS[key], lw=2.2, alpha=0.92, label=LABELS[key])

        cs = data.get('collision_steps', np.array([]))
        if len(cs) > 0:
            ax.scatter(ee[cs, 0], ee[cs, 1],
                       c='red', s=40, marker='x', zorder=10,
                       label=f'Collision ({len(cs)} steps)')

        if 'cross_paths' in data:
            cp = data['cross_paths']
            for i, clr in enumerate(['#cc2200', '#0033cc']):
                ax.plot(cp[i, :, 0], cp[i, :, 1],
                        ':', color=clr, lw=1.4, alpha=0.55, label=f'Obstacle {i+1}')
                for frame in [0, 250, 499]:
                    fi = min(frame, cp.shape[1]-1)
                    ax.add_patch(plt.Circle(
                        (cp[i, fi, 0], cp[i, fi, 1]), rc.CROSS_RADIUS,
                        color=clr, alpha=0.22, zorder=4))

        err_mean = float(np.mean(data['err_mm']))
        ax.set_title(f'{LABELS[key]}\nMean error: {err_mean:.1f} mm',
                     fontsize=13, fontweight='bold', color=COLORS[key])
        ax.set_xlabel('X [m]', fontsize=11)
        ax.set_ylabel('Y [m]', fontsize=11)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, f'cross_fulltraj_{key.lower()}')

def save_zoom_figures(results, waypoints):
    """장애물 교차 구간 확대 (t=2.5s, t=2.7s)."""
    print('\n[FIG6] Zoom Figures')
    t = np.arange(N_STEPS) * DT
    for key, data in results.items():
        ee = data['ee'][:N_STEPS]
        for zt in [2.5, 2.7]:
            win  = 0.6
            mask = (t >= zt - win) & (t <= zt + win)
            if mask.sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
            ax.plot(waypoints[mask, 0], waypoints[mask, 1],
                    '--', color='#008844', lw=2.0, alpha=0.6, label='Reference')
            ax.plot(ee[mask, 0], ee[mask, 1],
                    color=COLORS[key], lw=2.4, alpha=0.92, label=LABELS[key])

            if 'cross_paths' in data:
                cp = data['cross_paths']
                fi = min(int(zt / DT), cp.shape[1] - 1)
                for i, clr in enumerate(['#cc2200', '#0033cc']):
                    ax.add_patch(plt.Circle(
                        (cp[i, fi, 0], cp[i, fi, 1]), rc.CROSS_RADIUS,
                        color=clr, alpha=0.35, zorder=5, label=f'Obs {i+1}'))
                    ax.add_patch(plt.Circle(
                        (cp[i, fi, 0], cp[i, fi, 1]), rc.CROSS_RADIUS,
                        color=clr, fill=False, lw=1.8, zorder=6))

            cs = data.get('collision_steps', np.array([]))
            idx_range = np.where(mask)[0]
            cs_in = cs[(cs >= idx_range[0]) & (cs <= idx_range[-1])]
            if len(cs_in) > 0:
                ax.scatter(ee[cs_in, 0], ee[cs_in, 1],
                           c='red', s=60, marker='x', zorder=10)

            ax.set_title(f'{LABELS[key]}  —  zoom  t = {zt:.1f} s',
                         fontsize=13, fontweight='bold', color=COLORS[key])
            ax.set_xlabel('X [m]', fontsize=11)
            ax.set_ylabel('Y [m]', fontsize=11)
            ax.set_aspect('equal')
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            zt_str = str(zt).replace('.', 'p')
            _save(fig, f'cross_zoom_{key.lower()}_t{zt_str}')


# ════════════════════════════════════════════════════════
# MP4 애니메이션
# ════════════════════════════════════════════════════════
def save_animation_mp4(results, waypoints):
    print('\n[VIDEO] 3D Animation')
    first = next(iter(results.values()))
    cross_paths = (first['cross_paths'] if 'cross_paths' in first
                   else rc._crossing_obstacle_paths(N_STEPS + 1))

    fig, anim_obj = rc.animate_comparison(
        results, waypoints,
        moving_obs_paths=cross_paths,
        moving_obs_radius=rc.CROSS_RADIUS,
        show_samples=True,
    )

    # 궤적 중심(X=0.40, Y=0.0, Z=0.35) 타이트 줌인
    # 반경 0.13m + 장애물·샘플 여유 0.09m
    for ax in fig.axes:
        if hasattr(ax, 'get_zlim'):
            ax.set_xlim(0.18, 0.62)
            ax.set_ylim(-0.22, 0.22)
            ax.set_zlim(0.13, 0.57)
            ax.view_init(elev=25, azim=-55)

    writers = animation.writers.list()
    if 'ffmpeg' in writers:
        out = os.path.join(VIDEO_DIR, 'mppi_cross_zoomed.mp4')
        writer = animation.FFMpegWriter(
            fps=25, bitrate=3000,
            metadata={'title': 'MPPI vs MPPI+DOB — UR5 Cross Obstacle'},
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        )
        print(f'  encoding → {out}')
        anim_obj.save(out, writer=writer, dpi=150,
                      progress_callback=lambda i, n: print(f'\r    frame {i}/{n}  ', end='', flush=True))
        print()
        print(f'  saved  {out}')
    else:
        out = os.path.join(VIDEO_DIR, 'mppi_cross_zoomed.gif')
        print(f'  [WARN] ffmpeg 없음 → GIF 저장: {out}')
        anim_obj.save(out, writer=animation.PillowWriter(fps=15), dpi=100,
                      progress_callback=lambda i, n: print(f'\r    frame {i}/{n}  ', end='', flush=True))
        print()
        print(f'  saved  {out}')
    plt.close(fig)


# ════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 58)
    print('save_figures.py — 논문·발표 그림 및 MP4 생성')
    print('=' * 58)

    results   = load_results()
    if not results:
        sys.exit(1)
    waypoints = make_waypoints()

    save_metric_dashboard(results, waypoints)
    save_timeseries(results, waypoints)
    save_3d_trajectory(results, waypoints)
    save_cross_comparison(results, waypoints)
    save_individual_trajectories(results, waypoints)
    save_zoom_figures(results, waypoints)
    save_animation_mp4(results, waypoints)

    print('\n' + '=' * 58)
    print('완료')
    print(f'  논문용 PNG  → {PAPER_DIR}/')
    print(f'  발표용 PNG  → {PRES_DIR}/')
    print(f'  MP4         → {VIDEO_DIR}/')
    print('=' * 58)
