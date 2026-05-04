"""
run_comparison.py — MPPI / SQP 컨트롤러 비교 실행기

ICROS 2026 논문 실험 재현 스크립트.
5가지 시뮬레이션 모드를 지원하며, 각 컨트롤러의 NPZ 결과를 로드하거나
없으면 새로 실행 후 저장한다.

실행:
    python run_comparison.py [base|obstacle|cross|full_sqp|cross_nodob]

Modes:
    base        — MPPI | MPPI+DOB | SQP+DOB  (장애물 없음, circle 궤적)
    obstacle    — 정적 원통 장애물 2개
    cross       — 교차 동적 구체 장애물 2개  ← 논문 Fig.1 결과
    full_sqp    — cross + SQP (Full SQP, max_iter=15) 비교
    cross_nodob — cross + DOB 유무 효과 비교

Controllers:
    mppi_dob_controller  — MPPI (PyTorch/GPU)
    sqp_controller       — SQP (acados, Full SQP max_iter=15)
    sqp_soft_controller  — SQP + soft constraint (acados)
    sqp_cross_controller — SQP + cross obstacle (acados)
    disturbance_observer — 공유 DOB 모듈 (α=40 rad/s)
"""
import sys, os, math, time
import types

# ── Path setup (identical to existing scripts) ────────────────────────────────
for _p in [
    '/home/economy02/.local/lib/python3.10/site-packages',
    '/home/economy02/pytorch_mppi/src',
    '/home/economy02/mppi_playground/src',
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/lib/python3/dist-packages',
    '/home/economy02/mpc/acados/interfaces/acados_template',
]:
    if _p not in sys.path:
        sys.path.append(_p)

_MPL_TOOLKITS_DIR = '/home/economy02/.local/lib/python3.10/site-packages/mpl_toolkits'
try:
    import mpl_toolkits as _mtk
except ModuleNotFoundError:
    _mtk = types.ModuleType('mpl_toolkits')
    _mtk.__path__ = []
    sys.modules['mpl_toolkits'] = _mtk

if os.path.isdir(_MPL_TOOLKITS_DIR) and _MPL_TOOLKITS_DIR not in _mtk.__path__:
    _mtk.__path__.insert(0, _MPL_TOOLKITS_DIR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import matplotlib.animation as animation

# ── Constants ─────────────────────────────────────────────────────────────────
DT        = 0.02            # 제어 주기 [s]
T_SIM     = 10.0            # 총 시뮬레이션 시간 [s]
N_STEPS   = int(T_SIM / DT)          # 500
TRAJ_TYPE = 'circle'        # 기본 궤적 종류

TRAJ_CENTER = np.array([0.40, 0.0, 0.35])  # 궤적 중심 좌표 [m]
TRAJ_RADIUS = 0.13          # 궤적 반경 [m]

_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_FILES = {
    'MPPI':           os.path.join(_DIR, 'MPPI_circle_results.npz'),
    'MPPI_DOB':       os.path.join(_DIR, 'MPPI_DOB_circle_results.npz'),
    'SQP_DOB':        os.path.join(_DIR, 'SQP_DOB_circle_results.npz'),
    'MPPI_OBS':       os.path.join(_DIR, 'MPPI_obs_results.npz'),
    'MPPI_DOB_OBS':   os.path.join(_DIR, 'MPPI_DOB_obs_results.npz'),
    'SQP_DOB_OBS':    os.path.join(_DIR, 'SQP_DOB_obs_results.npz'),
    'SQP_SOFT_OBS':   os.path.join(_DIR, 'SQP_soft_obs_results.npz'),
    'MPPI_CROSS':     os.path.join(_DIR, 'MPPI_cross_results.npz'),
    'MPPI_DOB_CROSS': os.path.join(_DIR, 'MPPI_DOB_cross_results.npz'),
    'SQP_SOFT_CROSS': os.path.join(_DIR, 'SQP_soft_cross_results.npz'),
    'FULL_SQP_SOFT_CROSS': os.path.join(_DIR, 'full_sqp_soft_cross_results.npz'),
    'SQP_CROSS_NO_DOB':   os.path.join(_DIR, 'SQP_cross_nodob_results.npz'),
}
LABELS = {
    'MPPI':           'Original MPPI',
    'MPPI_DOB':       'MPPI + DOB',
    'SQP_DOB':        'SQP + DOB',
    'MPPI_OBS':       'MPPI (Obstacle)',
    'MPPI_DOB_OBS':   'MPPI + DOB (Obstacle)',
    'SQP_DOB_OBS':    'SQP + DOB (No Constraint)',
    'SQP_SOFT_OBS':   'SQP + DOB (Soft Constraint)',
    'MPPI_CROSS':     'MPPI (Crossing Obs)',
    'MPPI_DOB_CROSS': 'MPPI + DOB (Crossing Obs)',
    'SQP_SOFT_CROSS': 'SQP + DOB (Crossing Soft)',
    'FULL_SQP_SOFT_CROSS': 'SQP + DOB (Crossing Soft)',
    'SQP_CROSS_NO_DOB':   'SQP (Crossing Soft, No DOB)',
}
COLORS = {
    'MPPI':           '#e05010',
    'MPPI_DOB':       '#1060c8',
    'SQP_DOB':        '#108840',
    'MPPI_OBS':       '#f0a020',
    'MPPI_DOB_OBS':   '#8030e0',
    'SQP_DOB_OBS':    '#d04040',
    'SQP_SOFT_OBS':   '#e08000',
    'MPPI_CROSS':     '#cc4400',
    'MPPI_DOB_CROSS': '#0050b0',
    'SQP_SOFT_CROSS': '#007744',
    'FULL_SQP_SOFT_CROSS': '#6600cc',
    'SQP_CROSS_NO_DOB':   '#ff8800',
}

# ── Cylinder obstacles (same as UR5_MPPI_DOB_Tuning.py) ──────────────────────
OBSTACLES = [
    {'center': (0.492,  0.092), 'r_obs': 0.008},   # θ=45°,  ~1.6cm 직경 얇은 봉
    {'center': (0.308, -0.092), 'r_obs': 0.008},   # θ=225°, ~1.6cm 직경 얇은 봉
]
_OBS_MARGIN = 0.045   # [m] 시각화용 안전 마진 (SQP soft constraint와 동기화)

# ── Scenario 3: Crossing dynamic sphere obstacles ─────────────────────────────
CROSS_RADIUS      = 0.030          # [m] 교차 장애물 반경 (r_obs=0.008 → 0.030)
_CROSS_MARGIN     = 0.026          # [m] 교차 장애물 안전 마진
_CROSS_SAFE_RADIUS = CROSS_RADIUS + _CROSS_MARGIN  # [m] 유효 안전 반경 (0.056 m)
_CROSS_W_SLACK    = 5e11           # 교차 장애물 slack 페널티 가중치
_CROSS_MAX_ITER   = 15             # Full SQP 최대 outer iteration 수 [회]
_CROSS_EXP_W      = 160.0    # MPPI 교차 장애물 지수 페널티 진폭
_CROSS_EXP_K      = 38.0     # 지수 기울기: 경계 즉시 신호, 심부 지수 급증
_CROSS_EXP_ZONE   = 0.012    # [m] 검출 구역 확장: r_outer = r_safe + 0.012

# ── Shared physics ────────────────────────────────────────────────────────────
_DH = [
    ( 0.0,      0.089159,  math.pi/2,  0.0),
    (-0.425,    0.0,       0.0,        0.0),
    (-0.39225,  0.0,       0.0,        0.0),
    ( 0.0,      0.10915,   math.pi/2,  0.0),
    ( 0.0,      0.09465,  -math.pi/2,  0.0),
    ( 0.0,      0.0823,    0.0,        0.0),
]
_I_EFF    = np.array([3.70, 8.40, 2.30, 1.20, 1.40, 0.30])  # 관절별 유효 관성 [kg·m²]
_B_DAMP   = np.array([0.12, 0.12, 0.10, 0.08, 0.08, 0.06])  # 관절별 점성 감쇠 계수 [N·m·s/rad]
_TAU_MAX  = np.array([150., 150., 150., 28., 28., 28.])      # 관절별 최대 토크 한계 [N·m]
_DQ_MAX   = 6.0                                              # 관절 최대 각속도 [rad/s]
D_AMP     = (15.0, -12.0, 10.0, 3.0, -3.0,  1.5)  # 관절별 외란 진폭 [N·m]
D_FREQ    = ( 1.0,   1.5,  0.8, 1.2,  0.9,  1.1)  # 관절별 외란 주파수 [Hz]
from disturbance_observer import DisturbanceObserver, TorchDisturbanceObserver, ALPHA_DOB


# ── Scenario 3 helpers ────────────────────────────────────────────────────────

def _deriv_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """연속시간 joint-space 동역학 미분: [dq; (u - B*dq)/I]."""
    dq  = x[6:]
    ddq = (u - _B_DAMP * dq) / _I_EFF
    return np.concatenate([dq, ddq])


def rk4_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """RK4 단일 적분 스텝. 반환: 다음 상태 x_{k+1}."""
    k1 = _deriv_np(x,                  u)
    k2 = _deriv_np(x + .5 * DT * k1,  u)
    k3 = _deriv_np(x + .5 * DT * k2,  u)
    k4 = _deriv_np(x + DT * k3,        u)
    ns = x + (DT / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    ns[6:] = np.clip(ns[6:], -_DQ_MAX, _DQ_MAX)
    return ns


def _crossing_obstacle_paths(n: int) -> np.ndarray:
    """두 구체 장애물이 독립 궤도로 이동하다 t=2.5s(EE 원 궤도 1/4지점)에서 동시 충돌.

    장애물 A: 0.18 Hz (x±0.13, y±0.13), 위상 −2π/5
    장애물 B: 0.23 Hz (x±0.10, y±0.13), 위상 0.85π
    두 장애물이 t=2.5s에 EE 1/4지점 (0.40, 0.13, 0.35)에서 동시 도달
    → SQP 이중 제약 동시 활성화, MPPI는 GPU 병렬 비용 평가로 속도 영향 없음
    반환: (2, n, 3)
    """
    tt = np.arange(n) * DT
    z  = np.full(n, TRAJ_CENTER[2])

    # 장애물 A — 0.18 Hz, 위상 −2π/5 → t=2.5s에 cos(π/2)=0, sin(π/2)=1 → (0.40, 0.13)
    x1 = TRAJ_CENTER[0] + 0.13  * np.cos(2*np.pi * 0.18 * tt - 2*np.pi/5)
    y1 =                   0.13  * np.sin(2*np.pi * 0.18 * tt - 2*np.pi/5)

    # 장애물 B — 0.23 Hz, 위상 0.85π → t=2.5s에 sin(2π)=0, cos(2π)=1 → (0.40, 0.13)
    x2 = TRAJ_CENTER[0] + 0.10  * np.sin(2*np.pi * 0.23 * tt + 0.85*np.pi)
    y2 =                   0.13  * np.cos(2*np.pi * 0.23 * tt + 0.85*np.pi)

    return np.stack([
        np.column_stack([x1, y1, z]),
        np.column_stack([x2, y2, z]),
    ], axis=0)  # (2, n, 3)


def _cross_collision_steps(ee: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """EE 궤적과 교차 장애물 경로를 비교하여 충돌 스텝 인덱스 배열을 반환한다."""
    hits = []
    for i, pt in enumerate(ee):
        idx = min(i, paths.shape[1] - 1)
        for obs_pos in paths[:, idx, :]:
            if np.linalg.norm(pt - obs_pos) < CROSS_RADIUS:
                hits.append(i)
                break
    return np.array(hits, dtype=int)


def fk_numpy(q: np.ndarray) -> np.ndarray:
    """q: (6,) → EE XYZ (3,) — DH 연쇄 순기구학."""
    T = np.eye(4)
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta = q[i] + th0
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        T = T @ np.array([[ct, -st*ca,  st*sa, a*ct],
                          [st,  ct*ca, -ct*sa, a*st],
                          [ 0,     sa,     ca,    d ],
                          [ 0,      0,      0,    1 ]])
    return T[:3, 3]


def fk_joints_np(q: np.ndarray) -> np.ndarray:
    """q: (6,) → (7, 3) all joint XYZ positions."""
    T, pts = np.eye(4), [np.zeros(3)]
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta = q[i] + th0
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        T = T @ np.array([[ct, -st*ca,  st*sa, a*ct],
                          [st,  ct*ca, -ct*sa, a*st],
                          [ 0,     sa,     ca,    d ],
                          [ 0,      0,      0,    1 ]])
        pts.append(T[:3, 3].copy())
    return np.array(pts)


def _circle(n, c, r):
    """n 포인트 원형 궤적 생성. 반환: (n, 3) ndarray."""
    ts  = np.linspace(0, 2*math.pi, n, endpoint=False)
    pts = np.tile(c.copy(), (n, 1))
    pts[:, 0] += r * np.cos(ts)
    pts[:, 1] += r * np.sin(ts)
    return pts


def _ik_np(target, q0, n_iter=300, tol=1e-5, lam=0.01, alpha=0.5):
    """Damped-least-squares IK. target: (3,) EE 목표 위치, 반환: (6,) 관절각."""
    q = q0.copy()
    for _ in range(n_iter):
        ee  = fk_numpy(q)
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        eps = 1e-6
        J   = np.zeros((3, 6))
        for j in range(6):
            dq_      = q.copy(); dq_[j] += eps
            J[:, j]  = (fk_numpy(dq_) - ee) / eps
        JJT = J @ J.T + lam * np.eye(3)
        q   = np.clip(q + alpha * (J.T @ np.linalg.solve(JJT, err)), -math.pi, math.pi)
    return q


# ══════════════════════════════════════════════════════════════════════════════
# Simulation runners
# ══════════════════════════════════════════════════════════════════════════════

def run_mppi(waypoints: np.ndarray, use_dob: bool = False,
             use_obstacle: bool = False,
             save_path: str = None) -> dict:
    """Run MPPI (±DOB, ±장애물 회피) with per-step solve timing."""
    import torch
    import mppi_dob_controller as _M
    _M.USE_OBSTACLE = use_obstacle          # 장애물 코스트 활성/비활성
    if hasattr(_M, 'set_cross_obstacles'):
        _M.set_cross_obstacles(None)
    from mppi_dob_controller import UR5Tracker, fk_batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float32   # compile+f32: ~3ms vs float64: ~9ms
    tag    = '+OBS' if use_obstacle else ''
    label  = f'MPPI+DOB{tag}' if use_dob else f'Original MPPI{tag}'
    print(f"\n[SIM] {label}  (device={device})")

    tracker = UR5Tracker(waypoints, device, dtype)

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0_np  = _ik_np(waypoints[0], q_hint)
    q0     = torch.tensor(q0_np, dtype=dtype, device=device)
    xk     = torch.cat([q0, torch.zeros(6, dtype=dtype, device=device)])

    u_min = tracker.u_min
    u_max = tracker.u_max

    # torch.compile 워밍업 — 첫 JIT 컴파일(~4000ms)을 시뮬레이션 루프 밖에서 수행
    # 워밍업 없으면 step-0 타이밍이 4000ms+ 로 오염되고, 원본 MPPI는 그 동안
    # 엉망인 제어 입력을 받아 DOB 없이 회복 불가 → mean err 100mm 이상
    print(f"  [compile warmup] torch.compile + CUDA graph 초기화 중...", end='', flush=True)
    _xk_w = xk.clone()
    for _ in range(20):   # 8 → 20: CUDA 그래프 완전 초기화 보장
        tracker.ctrl.command(_xk_w)
    torch.cuda.synchronize()
    print(" 완료")

    q_hist, ee_hist, u_hist         = [q0_np.copy()], [fk_numpy(q0_np)], []
    err_mm_hist, solve_times        = [], []
    d_hat_hist, d_true_hist         = [np.zeros(6)], [np.zeros(6)]
    _use_cuda = torch.cuda.is_available()

    for step in range(N_STEPS):
        # ── timed MPPI solve (GPU sync 으로 async 큐 오염 제거) ──────────────
        if _use_cuda: torch.cuda.synchronize()
        t0      = time.perf_counter()
        delta_u = tracker.ctrl.command(xk).clamp(u_min, u_max)
        if _use_cuda: torch.cuda.synchronize()
        solve_times.append(time.perf_counter() - t0)

        # ── disturbance ──────────────────────────────────────────────────────
        t_sim    = step * DT
        d_np     = np.array([D_AMP[i] * math.sin(2*math.pi*D_FREQ[i]*t_sim)
                              for i in range(6)])
        d_true   = torch.tensor(d_np, dtype=dtype, device=device)

        # ── apply control + DOB compensation ─────────────────────────────────
        u_app   = delta_u + tracker._d_hat if use_dob else delta_u
        xk_prev = xk.clone()
        u_eff   = (u_app - d_true).clamp(u_min, u_max)
        xk      = tracker._dynamics(xk.unsqueeze(0), u_eff.unsqueeze(0), 0).squeeze(0)

        # ── DOB update ───────────────────────────────────────────────────────
        if use_dob:
            tracker._dob.update(u_app, xk_prev[6:], xk[6:])
            tracker._d_hat = tracker._dob.d_hat

        tracker._tau_prev  = delta_u.detach().clone()
        tracker._step     += 1

        q_np  = xk[:6].cpu().numpy()
        ee_np = fk_numpy(q_np)
        wp    = waypoints[(step + 1) % len(waypoints)]
        err_mm_hist.append(np.linalg.norm(ee_np - wp) * 1e3)

        q_hist    .append(q_np.copy())
        ee_hist   .append(ee_np.copy())
        u_hist    .append(u_app.cpu().numpy().copy())
        d_hat_hist .append(tracker._d_hat.cpu().numpy().copy())
        d_true_hist.append(d_np.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  "
                  f"err={err_mm_hist[-1]:.1f}mm  "
                  f"solve={solve_times[-1]*1e3:.1f}ms")

    data = dict(
        q       = np.array(q_hist),
        ee      = np.array(ee_hist),
        u       = np.array(u_hist),
        d_hat   = np.array(d_hat_hist),
        d_true  = np.array(d_true_hist),
        solve_t = np.array(solve_times),
        err_mm  = np.array(err_mm_hist),
    )
    print(f"  → mean err={np.mean(err_mm_hist):.1f}mm  "
          f"mean solve={np.mean(solve_times)*1e3:.2f}ms")
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_sqp_dob(waypoints: np.ndarray, save_path: str = None) -> dict:
    """Run SQP+DOB via UR5SQPTracker (acados). solve_t already recorded."""
    sys.path.insert(0, _DIR)
    from sqp_controller import UR5SQPTracker

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)

    print("\n[SIM] SQP+DOB (acados)")
    tracker = UR5SQPTracker(waypoints)
    (q, ee, u, err_mm, d_hat, d_true, solve_t) = tracker.run(
        q0, has_disturbance=True, use_dob=True
    )
    data = dict(q=q, ee=ee, u=u, d_hat=d_hat, d_true=d_true,
                solve_t=solve_t, err_mm=err_mm)
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_sqp_dob_obstacle(waypoints: np.ndarray, save_path: str = None) -> dict:
    """SQP+DOB를 장애물 회피 없이 실행 — 충돌 여부 기록.
    SQP는 비선형 장애물 제약 추가 시 구조 전체 재설계 필요 → 여기서는
    '제약 없이 그냥 달리면 어떻게 되는가'를 보여주는 baseline.
    """
    sys.path.insert(0, _DIR)
    from sqp_controller import UR5SQPTracker

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)

    print("\n[SIM] SQP+DOB (acados) — 장애물 회피 제약 없음 (충돌 baseline)")
    tracker = UR5SQPTracker(waypoints)
    (q, ee, u, err_mm, d_hat, d_true, solve_t) = tracker.run(
        q0, has_disturbance=True, use_dob=True
    )

    # 충돌 판정: EE XY와 각 장애물 원통 간 거리 확인
    collision_steps = []
    for i, (ex, ey, _) in enumerate(ee):
        for obs in OBSTACLES:
            cx, cy = obs['center']
            dist = np.sqrt((ex - cx)**2 + (ey - cy)**2)
            if dist < obs['r_obs']:
                collision_steps.append(i)
                break
    print(f"  충돌 스텝 수: {len(collision_steps)} / {len(ee)}")
    if collision_steps:
        print(f"  첫 충돌: step {collision_steps[0]}  "
              f"t={collision_steps[0]*DT:.2f}s")

    data = dict(q=q, ee=ee, u=u, d_hat=d_hat, d_true=d_true,
                solve_t=solve_t, err_mm=err_mm,
                collision_steps=np.array(collision_steps))
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_sqp_soft_obstacle(waypoints: np.ndarray, save_path: str = None) -> dict:
    """SQP+DOB + soft 장애물 제약 실행.
    W_SLACK을 낮게 설정 → 추적 비용이 슬랙 비용 압도 → 여전히 충돌.
    연산 시간은 제약 없는 버전보다 증가 (NLP 전환 + 슬랙 최적화).
    """
    sys.path.insert(0, _DIR)
    from sqp_soft_controller import UR5SQPSoftTracker, W_SLACK

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)

    print(f"\n[SIM] SQP+DOB (soft constraint, W_SLACK={W_SLACK:.0e})")
    tracker = UR5SQPSoftTracker(waypoints)
    (q, ee, u, err_mm, d_hat, d_true, solve_t) = tracker.run(
        q0, has_disturbance=True, use_dob=True
    )

    # 충돌 판정
    collision_steps = []
    for i, (ex, ey, _) in enumerate(ee):
        for obs in OBSTACLES:
            cx, cy = obs['center']
            if np.sqrt((ex - cx)**2 + (ey - cy)**2) < obs['r_obs']:
                collision_steps.append(i)
                break
    print(f"  충돌 스텝 수: {len(collision_steps)} / {len(ee)}")
    if collision_steps:
        print(f"  첫 충돌: step {collision_steps[0]}  "
              f"t={collision_steps[0]*DT:.2f}s")
    solve_ms = np.array([s * 1e3 for s in solve_t])
    print(f"  solve time: mean={solve_ms.mean():.2f}ms  max={solve_ms.max():.2f}ms")

    data = dict(q=q, ee=ee, u=u, d_hat=d_hat, d_true=d_true,
                solve_t=solve_t, err_mm=err_mm,
                collision_steps=np.array(collision_steps))
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3 — Crossing dynamic sphere obstacles
# ══════════════════════════════════════════════════════════════════════════════

def run_mppi_cross(waypoints: np.ndarray, use_dob: bool = False,
                   save_path: str = None) -> dict:
    """MPPI (±DOB) with two crossing dynamic sphere obstacles."""
    import torch
    import mppi_dob_controller as _M
    _M.USE_OBSTACLE = True
    _M.set_cross_obstacles(
        cross_paths := _crossing_obstacle_paths(N_STEPS + 50),
        safe_radius=_CROSS_SAFE_RADIUS,
        exp_w=_CROSS_EXP_W,
        exp_k=_CROSS_EXP_K,
        exp_zone=_CROSS_EXP_ZONE,
    )
    from mppi_dob_controller import UR5Tracker, fk_batch

    cross_paths_np = cross_paths

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float32
    label  = 'MPPI+DOB (Cross)' if use_dob else 'MPPI (Cross)'
    print(f"\n[SIM] {label}  (device={device})")

    tracker = UR5Tracker(waypoints, device, dtype)
    q_hint  = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0_np   = _ik_np(waypoints[0], q_hint)
    q0      = torch.tensor(q0_np, dtype=dtype, device=device)
    xk      = torch.cat([q0, torch.zeros(6, dtype=dtype, device=device)])

    print("  [compile warmup]...", end='', flush=True)
    for _ in range(20):
        tracker.ctrl.command(xk.clone())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(" 완료")

    q_hist, ee_hist, u_hist         = [q0_np.copy()], [fk_numpy(q0_np)], []
    err_mm_hist, solve_times        = [], []
    d_hat_hist, d_true_hist         = [np.zeros(6)], [np.zeros(6)]
    ee_samples_hist, w_samples_hist = [], []   # (N_STEPS, K_vis, T_h, 3), (N_STEPS, K_vis)
    _use_cuda = torch.cuda.is_available()
    _K_VIS    = 1000   # 시각화할 샘플 수

    for step in range(N_STEPS):
        if _use_cuda: torch.cuda.synchronize()
        t0      = time.perf_counter()
        delta_u = tracker.ctrl.command(xk).clamp(tracker.u_min, tracker.u_max)
        if _use_cuda: torch.cuda.synchronize()
        solve_times.append(time.perf_counter() - t0)

        # ── 샘플 궤적 추출 (command 직후) ────────────────────────────────────
        with torch.no_grad():
            ctrl  = tracker.ctrl
            K_all = ctrl.perturbed_action.shape[0]
            T_h   = ctrl.perturbed_action.shape[1]
            K_vis = min(_K_VIS, K_all)

            # omega 상위 K_vis 샘플 선택
            top_idx = ctrl.omega.topk(K_vis).indices            # (K_vis,)
            pa_top  = ctrl.perturbed_action[top_idx]            # (K_vis, T_h, 6)
            w_top   = ctrl.omega[top_idx].cpu().numpy()         # (K_vis,)

            # 현재 xk에서 K_vis개 동시 롤아웃
            st = xk.view(1, -1).expand(K_vis, -1).clone()      # (K_vis, 12)
            q_traj = torch.empty(K_vis, T_h, 6, dtype=dtype, device=device)
            for t in range(T_h):
                u  = ctrl.u_scale * pa_top[:, t]
                st = ctrl._dynamics_fn(st, u, t)
                q_traj[:, t] = st[:, :6]

            ee_samp = fk_batch(q_traj.reshape(-1, 6)) \
                          .reshape(K_vis, T_h, 3).cpu().numpy()  # (K_vis, T_h, 3)
            ee_samples_hist.append(ee_samp)
            w_samples_hist .append(w_top)
        # ─────────────────────────────────────────────────────────────────────

        t_sim  = step * DT
        d_np   = np.array([D_AMP[i] * math.sin(2*math.pi*D_FREQ[i]*t_sim) for i in range(6)])
        d_true = torch.tensor(d_np, dtype=dtype, device=device)

        u_app   = delta_u + tracker._d_hat if use_dob else delta_u
        xk_prev = xk.clone()
        u_eff   = (u_app - d_true).clamp(tracker.u_min, tracker.u_max)
        xk      = tracker._dynamics(xk.unsqueeze(0), u_eff.unsqueeze(0), 0).squeeze(0)

        if use_dob:
            tracker._dob.update(u_app, xk_prev[6:], xk[6:])
            tracker._d_hat = tracker._dob.d_hat

        tracker._tau_prev = delta_u.detach().clone()
        tracker._step    += 1

        q_np  = xk[:6].cpu().numpy()
        ee_np = fk_numpy(q_np)
        wp    = waypoints[(step + 1) % len(waypoints)]
        err_mm_hist.append(np.linalg.norm(ee_np - wp) * 1e3)

        q_hist    .append(q_np.copy())
        ee_hist   .append(ee_np.copy())
        u_hist    .append(u_app.cpu().numpy().copy())
        d_hat_hist .append(tracker._d_hat.cpu().numpy().copy())
        d_true_hist.append(d_np.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  "
                  f"err={err_mm_hist[-1]:.1f}mm  "
                  f"solve={solve_times[-1]*1e3:.1f}ms")

    ee_arr          = np.array(ee_hist)
    collision_steps = _cross_collision_steps(ee_arr, cross_paths_np[:, :N_STEPS+1, :])
    data = dict(
        q=np.array(q_hist), ee=ee_arr, u=np.array(u_hist),
        d_hat=np.array(d_hat_hist), d_true=np.array(d_true_hist),
        solve_t=np.array(solve_times), err_mm=np.array(err_mm_hist),
        collision_steps=collision_steps,
        cross_paths=cross_paths_np[:, :N_STEPS+1, :],
        ee_samples=np.array(ee_samples_hist, dtype=np.float32),  # (N_STEPS, K_vis, T_h, 3)
        w_samples =np.array(w_samples_hist,  dtype=np.float32),  # (N_STEPS, K_vis)
    )
    print(f"  → mean err={np.mean(err_mm_hist):.1f}mm  "
          f"mean solve={np.mean(solve_times)*1e3:.2f}ms  "
          f"collisions={len(collision_steps)}")
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_sqp_soft_cross(waypoints: np.ndarray, save_path: str = None) -> dict:
    """SQP+DOB + soft 구속으로 2개 교차 동적 구체 장애물 회피."""
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    from sqp_controller import _casadi_fk, HORIZON, W_POS, W_VEL, W_ACT, W_TERM

    cross_paths = _crossing_obstacle_paths(N_STEPS + HORIZON + 2)
    n_obs = cross_paths.shape[0]   # 2
    p_dim = 3 * n_obs              # 6

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)
    x0     = np.concatenate([q0, np.zeros(6)])

    q_sym   = ca.SX.sym('q',   6)
    dq_sym  = ca.SX.sym('dq',  6)
    x_sym   = ca.vertcat(q_sym, dq_sym)
    tau_sym = ca.SX.sym('tau', 6)
    p_sym   = ca.SX.sym('p',   p_dim)

    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    f_expl = ca.vertcat(dq_sym, (tau_sym - B_ca * dq_sym) / I_ca)

    ee_ca    = _casadi_fk(q_sym)
    J_ee     = ca.jacobian(ee_ca, q_sym)
    v_ee     = ca.mtimes(J_ee, dq_sym)
    y_expr   = ca.vertcat(ee_ca, v_ee, tau_sym)
    y_expr_e = ee_ca

    h_list = []
    for i in range(n_obs):
        cx = p_sym[3*i + 0]
        cy = p_sym[3*i + 1]
        cz = p_sym[3*i + 2]
        dist_sq = (ee_ca[0]-cx)**2 + (ee_ca[1]-cy)**2 + (ee_ca[2]-cz)**2
        h_list.append(dist_sq - _CROSS_SAFE_RADIUS**2)
    h_expr = ca.vertcat(*h_list)

    model = AcadosModel()
    model.name          = 'ur5_sqp_cross'
    model.x             = x_sym
    model.u             = tau_sym
    model.p             = p_sym
    model.xdot          = ca.SX.sym('xdot', 12)
    model.f_expl_expr   = f_expl
    model.f_impl_expr   = model.xdot - f_expl
    model.cost_y_expr   = y_expr
    model.cost_y_expr_e = y_expr_e
    model.con_h_expr    = h_expr
    model.con_h_expr_e  = h_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.tf        = HORIZON * DT
    ocp.solver_options.N_horizon = HORIZON
    ocp.dims.np = p_dim

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W           = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e         = np.diag([W_TERM]*3)
    ocp.cost.yref        = np.zeros(12)
    ocp.cost.yref_e      = np.zeros(3)

    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()
    ocp.constraints.lbu     = -_TAU_MAX
    ocp.constraints.ubu     =  _TAU_MAX
    ocp.constraints.idxbu   = np.arange(6)
    lbx, ubx = np.full(nx, -1e9), np.full(nx, 1e9)
    lbx[6:] = -_DQ_MAX; ubx[6:] = _DQ_MAX
    ocp.constraints.lbx    = lbx; ocp.constraints.ubx    = ubx
    ocp.constraints.idxbx  = np.arange(nx)
    ocp.constraints.lbx_e  = lbx; ocp.constraints.ubx_e  = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    ocp.constraints.lh   = np.zeros(n_obs); ocp.constraints.uh   = np.full(n_obs, 1e9)
    ocp.constraints.lh_e = np.zeros(n_obs); ocp.constraints.uh_e = np.full(n_obs, 1e9)
    ocp.constraints.idxsh   = np.arange(n_obs)
    ocp.constraints.idxsh_e = np.arange(n_obs)

    ocp.cost.Zl   = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu   = np.zeros(n_obs)
    ocp.cost.zl   = np.zeros(n_obs)
    ocp.cost.zu   = np.zeros(n_obs)
    ocp.cost.Zl_e = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu_e = np.zeros(n_obs)
    ocp.cost.zl_e = np.zeros(n_obs)
    ocp.cost.zu_e = np.zeros(n_obs)

    ocp.parameter_values = cross_paths[:, 0, :].reshape(-1)
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.nlp_solver_max_iter   = _CROSS_MAX_ITER
    ocp.solver_options.tol                   = 1e-4
    ocp.solver_options.qp_tol               = 1e-4
    ocp.solver_options.print_level           = 0

    json_file = os.path.join(_DIR, 'ur5_sqp_cross_acados_ocp.json')
    out_dir   = os.path.join(_DIR, 'c_generated_code_ur5sqp_cross')
    try:
        ocp.code_gen_opts.code_export_directory = out_dir
    except AttributeError:
        ocp.code_export_directory = out_dir

    print(f"\n[SIM] SQP+DOB (cross soft, r_safe={_CROSS_SAFE_RADIUS:.3f}m, max_iter={_CROSS_MAX_ITER})")
    t_build = time.perf_counter()
    solver  = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
    print(f"  → solver ready in {time.perf_counter()-t_build:.1f}s")

    x_cur  = x0.copy()
    dob    = DisturbanceObserver(_I_EFF, _B_DAMP, DT)
    q_hist, ee_hist, u_hist       = [q0.copy()], [fk_numpy(q0)], []
    err_mm_list, d_hat_hist       = [], [dob.d_hat.copy()]
    d_true_hist, solve_times      = [np.zeros(6)], []
    n_warn = 0

    for step in range(N_STEPS):
        solver.set(0, 'lbx', x_cur)
        solver.set(0, 'ubx', x_cur)
        for k in range(HORIZON):
            wp  = waypoints[min(step + k,     len(waypoints) - 1)]
            nxt = waypoints[min(step + k + 1, len(waypoints) - 1)]
            solver.set(k, 'yref', np.concatenate([wp, (nxt-wp)/DT, np.zeros(6)]))
            solver.set(k, 'p', cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))
        solver.set(HORIZON, 'yref', waypoints[min(step+HORIZON, len(waypoints)-1)])
        solver.set(HORIZON, 'p',
                   cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))

        t0     = time.perf_counter()
        status = solver.solve()
        solve_times.append(time.perf_counter() - t0)
        if status not in (0, 2):
            n_warn += 1
            if n_warn <= 5:
                print(f"  [WARN] acados status={status} at step {step}")

        u_sqp  = solver.get(0, 'u')
        u_app  = np.clip(u_sqp + dob.d_hat, -_TAU_MAX, _TAU_MAX)
        d_true = np.array([D_AMP[i]*math.sin(2*math.pi*D_FREQ[i]*step*DT) for i in range(6)])
        u_eff  = np.clip(u_app - d_true, -_TAU_MAX, _TAU_MAX)
        x_prev = x_cur.copy()
        x_cur  = rk4_step(x_cur, u_eff)
        dob.update(u_app, x_prev[6:], x_cur[6:])

        q_np  = x_cur[:6]
        ee_np = fk_numpy(q_np)
        e_mm  = np.linalg.norm(ee_np - waypoints[min(step+1, len(waypoints)-1)]) * 1e3
        q_hist    .append(q_np.copy())
        ee_hist   .append(ee_np.copy())
        u_hist    .append(u_app.copy())
        err_mm_list.append(e_mm)
        d_hat_hist .append(dob.d_hat.copy())
        d_true_hist.append(d_true.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  err={e_mm:.1f}mm  "
                  f"solve={solve_times[-1]*1e3:.1f}ms")

    ee_arr          = np.array(ee_hist)
    collision_steps = _cross_collision_steps(ee_arr, cross_paths[:, :N_STEPS+1, :])
    solve_ms        = np.array(solve_times) * 1e3
    print(f"  → mean err={np.mean(err_mm_list):.1f}mm  "
          f"solve mean={solve_ms.mean():.2f}ms  max={solve_ms.max():.2f}ms  "
          f"collisions={len(collision_steps)}  solver_warns={n_warn}")
    data = dict(
        q=np.array(q_hist), ee=ee_arr, u=np.array(u_hist),
        d_hat=np.array(d_hat_hist), d_true=np.array(d_true_hist),
        solve_t=np.array(solve_times), err_mm=np.array(err_mm_list),
        collision_steps=collision_steps,
        cross_paths=cross_paths[:, :N_STEPS+1, :],
        solver_warns=np.array([n_warn]),
    )
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_full_sqp_soft_cross(waypoints: np.ndarray, save_path: str = None) -> dict:
    """Full SQP+DOB + soft 구속으로 2개 교차 동적 구체 장애물 회피.

    run_sqp_soft_cross와 동일한 시나리오·파라미터이나,
    별도 acados 빌드(모델명·코드디렉터리 분리)로 Full_SQP.py 모듈과 동등한 조건에서 실행.
    """
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    from sqp_controller import _casadi_fk, HORIZON, W_POS, W_VEL, W_ACT, W_TERM

    cross_paths = _crossing_obstacle_paths(N_STEPS + HORIZON + 2)
    n_obs = cross_paths.shape[0]   # 2
    p_dim = 3 * n_obs              # 6

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)
    x0     = np.concatenate([q0, np.zeros(6)])

    q_sym   = ca.SX.sym('q',   6)
    dq_sym  = ca.SX.sym('dq',  6)
    x_sym   = ca.vertcat(q_sym, dq_sym)
    tau_sym = ca.SX.sym('tau', 6)
    p_sym   = ca.SX.sym('p',   p_dim)

    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    f_expl = ca.vertcat(dq_sym, (tau_sym - B_ca * dq_sym) / I_ca)

    ee_ca    = _casadi_fk(q_sym)
    J_ee     = ca.jacobian(ee_ca, q_sym)
    v_ee     = ca.mtimes(J_ee, dq_sym)
    y_expr   = ca.vertcat(ee_ca, v_ee, tau_sym)
    y_expr_e = ee_ca

    h_list = []
    for i in range(n_obs):
        cx = p_sym[3*i + 0]; cy = p_sym[3*i + 1]; cz = p_sym[3*i + 2]
        dist_sq = (ee_ca[0]-cx)**2 + (ee_ca[1]-cy)**2 + (ee_ca[2]-cz)**2
        h_list.append(dist_sq - _CROSS_SAFE_RADIUS**2)
    h_expr = ca.vertcat(*h_list)

    model = AcadosModel()
    model.name          = 'ur5_full_sqp_cross'   # 별도 빌드 — ur5_sqp_cross와 충돌 방지
    model.x             = x_sym
    model.u             = tau_sym
    model.p             = p_sym
    model.xdot          = ca.SX.sym('xdot', 12)
    model.f_expl_expr   = f_expl
    model.f_impl_expr   = model.xdot - f_expl
    model.cost_y_expr   = y_expr
    model.cost_y_expr_e = y_expr_e
    model.con_h_expr    = h_expr
    model.con_h_expr_e  = h_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.tf        = HORIZON * DT
    ocp.solver_options.N_horizon = HORIZON
    ocp.dims.np = p_dim

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W           = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e         = np.diag([W_TERM]*3)
    ocp.cost.yref        = np.zeros(12)
    ocp.cost.yref_e      = np.zeros(3)

    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()
    ocp.constraints.lbu     = -_TAU_MAX
    ocp.constraints.ubu     =  _TAU_MAX
    ocp.constraints.idxbu   = np.arange(6)
    lbx, ubx = np.full(nx, -1e9), np.full(nx, 1e9)
    lbx[6:] = -_DQ_MAX; ubx[6:] = _DQ_MAX
    ocp.constraints.lbx    = lbx; ocp.constraints.ubx    = ubx
    ocp.constraints.idxbx  = np.arange(nx)
    ocp.constraints.lbx_e  = lbx; ocp.constraints.ubx_e  = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    ocp.constraints.lh   = np.zeros(n_obs); ocp.constraints.uh   = np.full(n_obs, 1e9)
    ocp.constraints.lh_e = np.zeros(n_obs); ocp.constraints.uh_e = np.full(n_obs, 1e9)
    ocp.constraints.idxsh   = np.arange(n_obs)
    ocp.constraints.idxsh_e = np.arange(n_obs)

    ocp.cost.Zl   = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu   = np.zeros(n_obs)
    ocp.cost.zl   = np.zeros(n_obs)
    ocp.cost.zu   = np.zeros(n_obs)
    ocp.cost.Zl_e = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu_e = np.zeros(n_obs)
    ocp.cost.zl_e = np.zeros(n_obs)
    ocp.cost.zu_e = np.zeros(n_obs)

    ocp.parameter_values = cross_paths[:, 0, :].reshape(-1)
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.nlp_solver_max_iter   = _CROSS_MAX_ITER
    ocp.solver_options.tol                   = 1e-4
    ocp.solver_options.qp_tol               = 1e-4
    ocp.solver_options.print_level           = 0

    json_file = os.path.join(_DIR, 'ur5_full_sqp_cross_acados_ocp.json')
    out_dir   = os.path.join(_DIR, 'c_generated_code_ur5full_sqp_cross')
    try:
        ocp.code_gen_opts.code_export_directory = out_dir
    except AttributeError:
        ocp.code_export_directory = out_dir

    print(f"\n[SIM] Full SQP+DOB (cross soft, r_safe={_CROSS_SAFE_RADIUS:.3f}m, max_iter={_CROSS_MAX_ITER})")
    t_build = time.perf_counter()
    solver  = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
    print(f"  → solver ready in {time.perf_counter()-t_build:.1f}s")

    x_cur  = x0.copy()
    dob    = DisturbanceObserver(_I_EFF, _B_DAMP, DT)
    q_hist, ee_hist, u_hist       = [q0.copy()], [fk_numpy(q0)], []
    err_mm_list, d_hat_hist       = [], [dob.d_hat.copy()]
    d_true_hist, solve_times      = [np.zeros(6)], []
    sqp_iter_list = []   # 매 스텝 실제 SQP outer-iteration 수 (Jacobian 재계산 횟수)
    n_warn = 0

    for step in range(N_STEPS):
        solver.set(0, 'lbx', x_cur)
        solver.set(0, 'ubx', x_cur)
        for k in range(HORIZON):
            wp  = waypoints[min(step + k,     len(waypoints) - 1)]
            nxt = waypoints[min(step + k + 1, len(waypoints) - 1)]
            solver.set(k, 'yref', np.concatenate([wp, (nxt-wp)/DT, np.zeros(6)]))
            solver.set(k, 'p', cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))
        solver.set(HORIZON, 'yref', waypoints[min(step+HORIZON, len(waypoints)-1)])
        solver.set(HORIZON, 'p',
                   cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))

        t0     = time.perf_counter()
        status = solver.solve()
        solve_times.append(time.perf_counter() - t0)

        # 실제 SQP outer-iteration 수 기록 (= Jacobian 재계산 횟수)
        try:
            sqp_iter_list.append(int(solver.get_stats('sqp_iter')))
        except Exception:
            sqp_iter_list.append(-1)

        if status not in (0, 2):
            n_warn += 1
            if n_warn <= 5:
                print(f"  [WARN] acados status={status} at step {step}")

        u_sqp  = solver.get(0, 'u')
        u_app  = np.clip(u_sqp + dob.d_hat, -_TAU_MAX, _TAU_MAX)
        d_true = np.array([D_AMP[i]*math.sin(2*math.pi*D_FREQ[i]*step*DT) for i in range(6)])
        u_eff  = np.clip(u_app - d_true, -_TAU_MAX, _TAU_MAX)
        x_prev = x_cur.copy()
        x_cur  = rk4_step(x_cur, u_eff)
        dob.update(u_app, x_prev[6:], x_cur[6:])

        q_np  = x_cur[:6]
        ee_np = fk_numpy(q_np)
        e_mm  = np.linalg.norm(ee_np - waypoints[min(step+1, len(waypoints)-1)]) * 1e3
        q_hist    .append(q_np.copy())
        ee_hist   .append(ee_np.copy())
        u_hist    .append(u_app.copy())
        err_mm_list.append(e_mm)
        d_hat_hist .append(dob.d_hat.copy())
        d_true_hist.append(d_true.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  err={e_mm:.1f}mm  "
                  f"solve={solve_times[-1]*1e3:.1f}ms")

    ee_arr          = np.array(ee_hist)
    collision_steps = _cross_collision_steps(ee_arr, cross_paths[:, :N_STEPS+1, :])
    solve_ms        = np.array(solve_times) * 1e3
    iters = np.array(sqp_iter_list)
    if iters[0] >= 0:
        from collections import Counter
        iter_dist = dict(sorted(Counter(iters.tolist()).items()))
        print(f"  → SQP iteration 분포 (Jacobian 재계산 횟수/스텝): {iter_dist}")
        print(f"     mean={iters.mean():.2f}  max={iters.max()}  "
              f"(iter=1 비율: {100*np.mean(iters==1):.0f}%)")
    print(f"  → mean err={np.mean(err_mm_list):.1f}mm  "
          f"solve mean={solve_ms.mean():.2f}ms  max={solve_ms.max():.2f}ms  "
          f"collisions={len(collision_steps)}  solver_warns={n_warn}")
    data = dict(
        q=np.array(q_hist), ee=ee_arr, u=np.array(u_hist),
        d_hat=np.array(d_hat_hist), d_true=np.array(d_true_hist),
        solve_t=np.array(solve_times), err_mm=np.array(err_mm_list),
        collision_steps=collision_steps,
        cross_paths=cross_paths[:, :N_STEPS+1, :],
        solver_warns=np.array([n_warn]),
    )
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


def run_sqp_cross_no_dob(waypoints: np.ndarray, save_path: str = None) -> dict:
    """SQP soft 구속으로 2개 교차 동적 구체 장애물 회피 — DOB 없음 (baseline).

    run_sqp_soft_cross와 동일한 시나리오·파라미터이나 DOB 보상을 적용하지 않음.
    u_app = u_sqp (d_hat 미가산), 외란은 환경에서 그대로 인가됨.
    """
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    from sqp_controller import _casadi_fk, HORIZON, W_POS, W_VEL, W_ACT, W_TERM

    cross_paths = _crossing_obstacle_paths(N_STEPS + HORIZON + 2)
    n_obs = cross_paths.shape[0]   # 2
    p_dim = 3 * n_obs              # 6

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0     = _ik_np(waypoints[0], q_hint)
    x0     = np.concatenate([q0, np.zeros(6)])

    q_sym   = ca.SX.sym('q',   6)
    dq_sym  = ca.SX.sym('dq',  6)
    x_sym   = ca.vertcat(q_sym, dq_sym)
    tau_sym = ca.SX.sym('tau', 6)
    p_sym   = ca.SX.sym('p',   p_dim)

    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    f_expl = ca.vertcat(dq_sym, (tau_sym - B_ca * dq_sym) / I_ca)

    ee_ca    = _casadi_fk(q_sym)
    J_ee     = ca.jacobian(ee_ca, q_sym)
    v_ee     = ca.mtimes(J_ee, dq_sym)
    y_expr   = ca.vertcat(ee_ca, v_ee, tau_sym)
    y_expr_e = ee_ca

    h_list = []
    for i in range(n_obs):
        cx = p_sym[3*i + 0]
        cy = p_sym[3*i + 1]
        cz = p_sym[3*i + 2]
        dist_sq = (ee_ca[0]-cx)**2 + (ee_ca[1]-cy)**2 + (ee_ca[2]-cz)**2
        h_list.append(dist_sq - _CROSS_SAFE_RADIUS**2)
    h_expr = ca.vertcat(*h_list)

    model = AcadosModel()
    model.name          = 'ur5_sqp_cross_nd'   # 별도 빌드 — DOB 없음 버전
    model.x             = x_sym
    model.u             = tau_sym
    model.p             = p_sym
    model.xdot          = ca.SX.sym('xdot', 12)
    model.f_expl_expr   = f_expl
    model.f_impl_expr   = model.xdot - f_expl
    model.cost_y_expr   = y_expr
    model.cost_y_expr_e = y_expr_e
    model.con_h_expr    = h_expr
    model.con_h_expr_e  = h_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.tf        = HORIZON * DT
    ocp.solver_options.N_horizon = HORIZON
    ocp.dims.np = p_dim

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W           = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e         = np.diag([W_TERM]*3)
    ocp.cost.yref        = np.zeros(12)
    ocp.cost.yref_e      = np.zeros(3)

    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()
    ocp.constraints.lbu     = -_TAU_MAX
    ocp.constraints.ubu     =  _TAU_MAX
    ocp.constraints.idxbu   = np.arange(6)
    lbx, ubx = np.full(nx, -1e9), np.full(nx, 1e9)
    lbx[6:] = -_DQ_MAX; ubx[6:] = _DQ_MAX
    ocp.constraints.lbx    = lbx; ocp.constraints.ubx    = ubx
    ocp.constraints.idxbx  = np.arange(nx)
    ocp.constraints.lbx_e  = lbx; ocp.constraints.ubx_e  = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    ocp.constraints.lh   = np.zeros(n_obs); ocp.constraints.uh   = np.full(n_obs, 1e9)
    ocp.constraints.lh_e = np.zeros(n_obs); ocp.constraints.uh_e = np.full(n_obs, 1e9)
    ocp.constraints.idxsh   = np.arange(n_obs)
    ocp.constraints.idxsh_e = np.arange(n_obs)

    ocp.cost.Zl   = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu   = np.zeros(n_obs)
    ocp.cost.zl   = np.zeros(n_obs)
    ocp.cost.zu   = np.zeros(n_obs)
    ocp.cost.Zl_e = _CROSS_W_SLACK * np.ones(n_obs)
    ocp.cost.Zu_e = np.zeros(n_obs)
    ocp.cost.zl_e = np.zeros(n_obs)
    ocp.cost.zu_e = np.zeros(n_obs)

    ocp.parameter_values = cross_paths[:, 0, :].reshape(-1)
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.nlp_solver_max_iter   = _CROSS_MAX_ITER
    ocp.solver_options.tol                   = 1e-4
    ocp.solver_options.qp_tol               = 1e-4
    ocp.solver_options.print_level           = 0

    json_file = os.path.join(_DIR, 'ur5_sqp_cross_nd_acados_ocp.json')
    out_dir   = os.path.join(_DIR, 'c_generated_code_ur5sqp_cross_nd')
    try:
        ocp.code_gen_opts.code_export_directory = out_dir
    except AttributeError:
        ocp.code_export_directory = out_dir

    print(f"\n[SIM] SQP (No DOB, cross soft, r_safe={_CROSS_SAFE_RADIUS:.3f}m, max_iter={_CROSS_MAX_ITER})")
    t_build = time.perf_counter()
    solver  = AcadosOcpSolver(ocp, json_file=json_file, verbose=False)
    print(f"  → solver ready in {time.perf_counter()-t_build:.1f}s")

    x_cur  = x0.copy()
    q_hist, ee_hist, u_hist       = [q0.copy()], [fk_numpy(q0)], []
    err_mm_list, d_hat_hist       = [], [np.zeros(6)]
    d_true_hist, solve_times      = [np.zeros(6)], []
    n_warn = 0

    for step in range(N_STEPS):
        solver.set(0, 'lbx', x_cur)
        solver.set(0, 'ubx', x_cur)
        for hk in range(HORIZON):
            wp  = waypoints[min(step + hk,     len(waypoints) - 1)]
            nxt = waypoints[min(step + hk + 1, len(waypoints) - 1)]
            solver.set(hk, 'yref', np.concatenate([wp, (nxt-wp)/DT, np.zeros(6)]))
            solver.set(hk, 'p', cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))
        solver.set(HORIZON, 'yref', waypoints[min(step+HORIZON, len(waypoints)-1)])
        solver.set(HORIZON, 'p',
                   cross_paths[:, min(step, cross_paths.shape[1]-1), :].reshape(-1))

        t0     = time.perf_counter()
        status = solver.solve()
        solve_times.append(time.perf_counter() - t0)
        if status not in (0, 2):
            n_warn += 1
            if n_warn <= 5:
                print(f"  [WARN] acados status={status} at step {step}")

        u_sqp  = solver.get(0, 'u')
        u_app  = np.clip(u_sqp, -_TAU_MAX, _TAU_MAX)   # DOB 미적용
        d_true = np.array([D_AMP[i]*math.sin(2*math.pi*D_FREQ[i]*step*DT) for i in range(6)])
        u_eff  = np.clip(u_app - d_true, -_TAU_MAX, _TAU_MAX)
        x_cur  = rk4_step(x_cur, u_eff)

        q_np  = x_cur[:6]
        ee_np = fk_numpy(q_np)
        e_mm  = np.linalg.norm(ee_np - waypoints[min(step+1, len(waypoints)-1)]) * 1e3
        q_hist    .append(q_np.copy())
        ee_hist   .append(ee_np.copy())
        u_hist    .append(u_app.copy())
        err_mm_list.append(e_mm)
        d_hat_hist .append(np.zeros(6))
        d_true_hist.append(d_true.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  err={e_mm:.1f}mm  "
                  f"solve={solve_times[-1]*1e3:.1f}ms")

    ee_arr          = np.array(ee_hist)
    collision_steps = _cross_collision_steps(ee_arr, cross_paths[:, :N_STEPS+1, :])
    solve_ms        = np.array(solve_times) * 1e3
    print(f"  → mean err={np.mean(err_mm_list):.1f}mm  "
          f"solve mean={solve_ms.mean():.2f}ms  max={solve_ms.max():.2f}ms  "
          f"collisions={len(collision_steps)}  solver_warns={n_warn}")
    data = dict(
        q=np.array(q_hist), ee=ee_arr, u=np.array(u_hist),
        d_hat=np.array(d_hat_hist), d_true=np.array(d_true_hist),
        solve_t=np.array(solve_times), err_mm=np.array(err_mm_list),
        collision_steps=collision_steps,
        cross_paths=cross_paths[:, :N_STEPS+1, :],
        solver_warns=np.array([n_warn]),
    )
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Data loading  (auto-run if NPZ missing)
# ══════════════════════════════════════════════════════════════════════════════

def load_or_run(waypoints: np.ndarray,
                run_obstacle: bool = False,
                run_cross: bool = False,
                do_mppi_cross: bool = False,
                run_cross_nodob: bool = False) -> dict:
    """NPZ가 있으면 로드, 없으면 시뮬레이션 실행 후 저장. 결과 dict 반환."""
    if do_mppi_cross:
        keys = ['MPPI_CROSS', 'MPPI_DOB_CROSS']
    elif run_cross_nodob:
        keys = ['MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS', 'SQP_CROSS_NO_DOB']
    else:
        base_keys  = ['MPPI', 'MPPI_DOB', 'SQP_DOB']
        obs_keys   = ['MPPI_OBS', 'MPPI_DOB_OBS', 'SQP_SOFT_OBS']
        cross_keys = ['MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS']
        keys = (base_keys
                + (obs_keys   if run_obstacle else [])
                + (cross_keys if run_cross    else []))

    results = {}
    for key in keys:
        path = RESULT_FILES[key]
        if os.path.exists(path):
            print(f"[LOAD] {LABELS[key]}: {os.path.basename(path)}")
            d = np.load(path, allow_pickle=True)
            results[key] = {k: d[k] for k in d.files}
            if 'err_mm' not in results[key]:
                ee = results[key]['ee']
                wp = waypoints[:N_STEPS]
                results[key]['err_mm'] = np.linalg.norm(ee[1:] - wp, axis=1) * 1e3
        else:
            print(f"[RUN ] {LABELS[key]} not found — running simulation...")
            if key == 'MPPI':
                results[key] = run_mppi(waypoints, use_dob=False, save_path=path)
            elif key == 'MPPI_DOB':
                results[key] = run_mppi(waypoints, use_dob=True, save_path=path)
            elif key == 'SQP_DOB':
                results[key] = run_sqp_dob(waypoints, save_path=path)
            elif key == 'MPPI_OBS':
                results[key] = run_mppi(waypoints, use_dob=False,
                                        use_obstacle=True, save_path=path)
            elif key == 'MPPI_DOB_OBS':
                results[key] = run_mppi(waypoints, use_dob=True,
                                        use_obstacle=True, save_path=path)
            elif key == 'SQP_DOB_OBS':
                results[key] = run_sqp_dob_obstacle(waypoints, save_path=path)
            elif key == 'SQP_SOFT_OBS':
                results[key] = run_sqp_soft_obstacle(waypoints, save_path=path)
            elif key == 'MPPI_CROSS':
                results[key] = run_mppi_cross(waypoints, use_dob=False, save_path=path)
            elif key == 'MPPI_DOB_CROSS':
                results[key] = run_mppi_cross(waypoints, use_dob=True, save_path=path)
            elif key == 'SQP_SOFT_CROSS':
                results[key] = run_sqp_soft_cross(waypoints, save_path=path)
            elif key == 'FULL_SQP_SOFT_CROSS':
                results[key] = run_full_sqp_soft_cross(waypoints, save_path=path)
            elif key == 'SQP_CROSS_NO_DOB':
                results[key] = run_sqp_cross_no_dob(waypoints, save_path=path)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════════

def _iqr_filter(st: np.ndarray) -> np.ndarray:
    """Tukey IQR fence: Q3 + 1.5*IQR 초과값 제거 (GPU 열throttle·캐시·OS preemption 등 실험 외적 이상값)."""
    q1, q3 = np.percentile(st, [25, 75])
    return st[st <= q3 + 1.5 * (q3 - q1)]


def compute_metrics(data: dict, waypoints: np.ndarray) -> tuple:
    """정확도·속도·강인성 메트릭을 계산하여 (acc, spd, rob) 튜플로 반환한다."""
    err  = data['err_mm']                              # (500,) mm
    st   = data['solve_t'] * 1e3                       # (500,) ms
    st_clean = _iqr_filter(st)                         # IQR 이상값 제거

    acc = dict(
        mean   = float(np.mean(err)),
        rmse   = float(np.sqrt(np.mean(err**2))),
        max    = float(np.max(err)),
        steady = float(np.mean(err[250:])),            # second 5s
    )
    spd = dict(
        mean       = float(np.mean(st)),
        max        = float(np.max(st)),
        p99        = float(np.percentile(st, 99)),
        max_clean  = float(np.max(st_clean)),          # IQR 필터링 후 worst-case
        n_outlier  = int(len(st) - len(st_clean)),     # 제거된 이상값 수
        rtf        = float(DT * 1e3 / np.mean(st)),   # 평균 기준 RTF
        rtf_wc     = float(DT * 1e3 / np.max(st)),    # raw worst-case RTF
        rtf_p99    = float(DT * 1e3 / np.percentile(st, 99)),
        rtf_iqr_wc = float(DT * 1e3 / np.max(st_clean)),  # IQR 필터링 후 WC-RTF
    )
    d_hat  = data['d_hat'][1:]                        # (500, 6)
    d_true = data['d_true'][1:]
    rob = dict(
        err_std  = float(np.std(err)),
        dob_rmse = float(np.sqrt(np.mean((d_hat - d_true)**2))),
        err_cv   = float(np.std(err) / (np.mean(err) + 1e-9)),
    )
    return acc, spd, rob


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Metric Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_dashboard(all_metrics: dict):
    """6-panel 메트릭 대시보드 figure를 생성하여 반환한다."""
    keys   = list(all_metrics.keys())
    colors = [COLORS[k] for k in keys]
    xlbls  = [LABELS[k] for k in keys]
    x      = np.arange(len(keys))
    w      = 0.5

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('UR5 Controller Comparison — Metric Dashboard\n'
                 '(Circle trajectory, 10 s, sinusoidal disturbance ON)',
                 fontsize=13, fontweight='bold')

    def _bar(ax, vals, title, ylabel, fmt='.1f', lower_is_better=True):
        bars = ax.bar(x, vals, color=colors, width=w,
                      edgecolor='white', linewidth=1.2)
        best = int(np.argmin(vals) if lower_is_better else np.argmax(vals))
        bars[best].set_edgecolor('gold')
        bars[best].set_linewidth(2.8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis='y', alpha=0.35)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w/2, b.get_height() * 1.02,
                    f'{v:{fmt}}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    _bar(axes[0, 0],
         [all_metrics[k][0]['mean']   for k in keys],
         'Mean EE Error ↓',       '[mm]')
    _bar(axes[0, 1],
         [all_metrics[k][0]['rmse']   for k in keys],
         'RMSE EE Error ↓',       '[mm]')
    _bar(axes[0, 2],
         [all_metrics[k][0]['steady'] for k in keys],
         'Steady-State Error ↓',  '[mm]')

    # ── WC-RTF bar (Mean Solve Time 대체) ────────────────────────────────────
    # MPPI: p99 기준 (GPU 지터 제외), SQP: max 기준 (알고리즘 고유 스파이크)
    wc_rtf_vals = [
        all_metrics[k][1]['rtf_p99'] if 'SQP' not in k
        else all_metrics[k][1]['rtf_wc']
        for k in keys
    ]
    mean_st_vals = [all_metrics[k][1]['mean'] for k in keys]
    ax_wc = axes[1, 0]
    bar_clrs = ['#d04040' if v < 1.0 else c
                for v, c in zip(wc_rtf_vals, colors)]
    bars_wc = ax_wc.bar(x, wc_rtf_vals, color=bar_clrs, width=w,
                        edgecolor='white', linewidth=1.2)
    best_wc = int(np.argmax(wc_rtf_vals))
    bars_wc[best_wc].set_edgecolor('gold')
    bars_wc[best_wc].set_linewidth(2.8)
    ax_wc.axhline(1.0, color='red', lw=2.0, ls='--', alpha=0.85,
                  label='RT limit  1.0×')
    ax_wc.set_xticks(x)
    ax_wc.set_xticklabels(xlbls, fontsize=8)
    ax_wc.set_title('Worst-Case RTF ↑   (< 1.0× = RT 위반)', fontsize=10)
    ax_wc.set_ylabel('[×]', fontsize=9)
    ax_wc.grid(axis='y', alpha=0.35)
    ax_wc.legend(fontsize=8, loc='upper right')
    # WC-RTF 값 + 위반 표시
    for b, v in zip(bars_wc, wc_rtf_vals):
        violation = '!!!' if v < 1.0 else ''
        ax_wc.text(b.get_x() + w/2, b.get_height() * 1.02,
                   f'{v:.1f}× {violation}',
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   color='#d04040' if v < 1.0 else '#111')
    # mean solve time 을 bar 안쪽에 주석
    for b, ms in zip(bars_wc, mean_st_vals):
        if b.get_height() > 0.15:
            ax_wc.text(b.get_x() + w/2, b.get_height() * 0.42,
                       f'μ={ms:.1f}ms',
                       ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
    ax_wc.text(0.01, 0.01,
               '† MPPI: p99 기준   SQP: max 기준',
               transform=ax_wc.transAxes,
               fontsize=7, color='#666', style='italic')

    _bar(axes[1, 1],
         [all_metrics[k][2]['err_std'] for k in keys],
         'Error Std (Robustness) ↓', '[mm]')
    _bar(axes[1, 2],
         [all_metrics[k][2]['dob_rmse'] for k in keys],
         'DOB Estimation RMSE ↓', '[Nm]')

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Time Series Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeseries(results: dict, waypoints: np.ndarray):
    """추적 오차·연산 시간·DOB 추정 품질을 3행 time-series figure로 생성한다."""
    t = np.arange(N_STEPS) * DT

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle('UR5 Controller Comparison — Time Series\n'
                 '(Circle, 10 s, sinusoidal disturbance ON)',
                 fontsize=13, fontweight='bold')

    # ① Tracking error
    ax = axes[0]
    for k, data in results.items():
        err = data['err_mm']
        ax.plot(t, err, color=COLORS[k], lw=1.5, label=LABELS[k])
        ax.axhline(np.mean(err), color=COLORS[k], lw=0.9, ls='--', alpha=0.6)
    ax.set_ylabel('EE Error [mm]', fontsize=9)
    ax.set_title('① Tracking Accuracy', fontsize=10, loc='left')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ② Solve time
    ax = axes[1]
    for k, data in results.items():
        st = data['solve_t'] * 1e3
        ax.plot(t, st, color=COLORS[k], lw=1.2, alpha=0.75, label=LABELS[k])
        ax.axhline(np.mean(st), color=COLORS[k], lw=1.2, ls='--', alpha=0.8,
                   label=f'{LABELS[k]} μ={np.mean(st):.1f}ms')
    ax.axhline(DT * 1e3, color='red', lw=1.5, ls='-.', alpha=0.8,
               label=f'RT limit ({DT*1e3:.0f} ms)')
    ax.set_ylabel('Solve Time [ms]', fontsize=9)
    ax.set_title('② Computational Speed', fontsize=10, loc='left')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # ③ DOB estimation RMSE (per-step, across 6 joints)
    ax   = axes[2]
    ax2  = ax.twinx()
    for k in ('MPPI_DOB', 'SQP_DOB'):
        if k not in results:
            continue
        d_hat  = results[k]['d_hat'][1:]     # (500, 6)
        d_true = results[k]['d_true'][1:]
        rms    = np.sqrt(np.mean((d_hat - d_true)**2, axis=1))
        ax.plot(t, rms, color=COLORS[k], lw=1.5, label=f'{LABELS[k]} DOB err')
    if 'MPPI' in results:
        ax2.plot(t, results['MPPI']['err_mm'],
                 color=COLORS['MPPI'], lw=1.2, ls='--', alpha=0.5,
                 label='MPPI EE err (ref)')
        ax2.set_ylabel('MPPI EE Error [mm]', fontsize=8, color=COLORS['MPPI'])
        ax2.tick_params(axis='y', labelcolor=COLORS['MPPI'])
    ax.set_xlabel('Time [s]', fontsize=9)
    ax.set_ylabel('DOB Est. RMS [Nm]', fontsize=9)
    ax.set_title('③ Robustness — DOB Estimation Quality', fontsize=10, loc='left')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3D 원통 장애물 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _draw_cylinders_3d(ax, z_bot=0.15, z_top=0.55, alpha=0.75):
    """3D axes에 원통 장애물을 solid 면으로 그린다 (얇은 봉 대응)."""
    th = np.linspace(0, 2 * np.pi, 64)
    zz = np.array([z_bot, z_top])
    for obs in OBSTACLES:
        cx, cy = obs['center']
        r      = obs['r_obs']
        # ── 측면 solid surface
        xs = cx + r * np.outer(np.cos(th), np.ones(2))
        ys = cy + r * np.outer(np.sin(th), np.ones(2))
        zs = np.outer(np.ones(len(th)), zz)
        ax.plot_surface(xs, ys, zs, color='#cc3300', alpha=alpha,
                        linewidth=0, antialiased=True, zorder=5)
        # ── 위·아래 캡
        r_cap = np.array([0, r])
        xs_c  = cx + np.outer(r_cap, np.cos(th))
        ys_c  = cy + np.outer(r_cap, np.sin(th))
        ax.plot_surface(xs_c, ys_c, np.full_like(xs_c, z_top),
                        color='#cc3300', alpha=alpha, linewidth=0, zorder=5)
        ax.plot_surface(xs_c, ys_c, np.full_like(xs_c, z_bot),
                        color='#cc3300', alpha=alpha, linewidth=0, zorder=5)
        # ── 안전 마진 점선 (r_safe = r + _OBS_MARGIN)
        r_safe = r + _OBS_MARGIN
        xs_m = cx + r_safe * np.cos(th)
        ys_m = cy + r_safe * np.sin(th)
        for z in [z_bot, z_top]:
            ax.plot(xs_m, ys_m, z, color='#cc3300', lw=0.8,
                    alpha=0.35, linestyle='--')


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — 3D Trajectory Comparison (static)
# ══════════════════════════════════════════════════════════════════════════════

def plot_3d_trajectory(results: dict, waypoints: np.ndarray,
                       show_obstacles: bool = False):
    """EE 3D 궤적 및 추적 오차 비교 figure를 생성하여 반환한다."""
    fig = plt.figure(figsize=(18, 8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1.0],
                            wspace=0.12, left=0.04, right=0.97,
                            top=0.90, bottom=0.10)
    ax     = fig.add_subplot(gs[0], projection='3d')
    ax_err = fig.add_subplot(gs[1])

    title_tag = '+ Cylinder Obstacles' if show_obstacles else 'sinusoidal disturbance ON'
    fig.suptitle(f'3D EE Trajectory & Tracking Error  (Circle, 10 s, {title_tag})',
                 fontsize=12, fontweight='bold')

    ax.plot(*waypoints.T, '--', color='#008844', lw=2.0, alpha=0.7,
            label='Reference')

    t = np.arange(N_STEPS) * DT
    for k, data in results.items():
        ee = data['ee']
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2],
                color=COLORS[k], lw=1.8, alpha=0.85, label=LABELS[k])
        ax.scatter(*ee[0],  color=COLORS[k], s=60, marker='o', zorder=10)
        ax.scatter(*ee[-1], color=COLORS[k], s=80, marker='*', zorder=10)
        cs = data.get('collision_steps', np.array([]))
        if len(cs) > 0:
            ax.scatter(ee[cs, 0], ee[cs, 1], ee[cs, 2],
                       c='red', s=20, marker='x', zorder=11,
                       label=f'{LABELS[k]} collision ({len(cs)} pts)')

        # ── tracking error panel ──────────────────────────────────────────
        err = data['err_mm']
        mu  = np.mean(err)
        ax_err.plot(t, err, color=COLORS[k], lw=1.8, alpha=0.90,
                    label=f'{LABELS[k]}  μ={mu:.1f} mm')
        ax_err.axhline(mu, color=COLORS[k], lw=1.0, ls=':', alpha=0.55)

    if show_obstacles:
        _draw_cylinders_3d(ax)

    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.legend(fontsize=9)
    ax.set_xlim(0.20, 0.60)
    ax.set_ylim(-0.22, 0.22)
    ax.set_zlim(0.20, 0.60)
    ax.grid(True, alpha=0.3)

    ax_err.set_xlabel('Time [s]', fontsize=10)
    ax_err.set_ylabel('EE Error [mm]', fontsize=10)
    ax_err.set_title('Tracking Error Comparison', fontsize=11)
    ax_err.set_xlim(0, T_SIM)
    ax_err.set_ylim(bottom=0)
    ax_err.legend(fontsize=9)
    ax_err.grid(alpha=0.3)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Obstacle Scenario Comparison (2D top-down + error timeseries)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_cylinders(ax, z_level=None, alpha=0.35):
    """ax가 2D면 원, 3D면 원통을 그린다."""
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for obs in OBSTACLES:
        cx, cy = obs['center']
        r = obs['r_obs']
        if z_level is None:          # 2D
            circ = plt.Circle((cx, cy), r, color='#cc3300',
                               alpha=alpha, zorder=5)
            ax.add_patch(circ)
            circ2 = plt.Circle((cx, cy), r + _OBS_MARGIN, color='#cc3300',
                                fill=False, ls='--', lw=1.2, alpha=0.5, zorder=5)
            ax.add_patch(circ2)
        else:                        # 3D 원통 근사 (다각형 링)
            theta = np.linspace(0, 2*np.pi, 30)
            z_bot, z_top = 0.15, 0.55
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            for z in np.linspace(z_bot, z_top, 8):
                ax.plot(xs, ys, z, color='#cc3300', lw=0.8, alpha=alpha)


def plot_obstacle_comparison(results: dict, waypoints: np.ndarray):
    """장애물 시나리오: MPPI+DOB vs SQP+DOB 비교 (XY 평면 + 오차 시계열)."""
    obs_keys  = [k for k in results if 'OBS' in k]
    base_keys = [k for k in results if k in ('MPPI_DOB', 'SQP_DOB')]
    all_keys  = base_keys + obs_keys
    if not obs_keys:
        print("[WARN] 장애물 결과 없음 — plot_obstacle_comparison 건너뜀")
        return None

    t = np.arange(N_STEPS) * DT

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Obstacle Avoidance: MPPI+DOB vs SQP+DOB\n'
                 '(Circle, 10 s, 2 cylinder obstacles, sinusoidal disturbance ON)',
                 fontsize=12, fontweight='bold')

    # ── 왼쪽: XY 평면 궤적 ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(waypoints[:, 0], waypoints[:, 1],
            '--', color='#008844', lw=1.8, alpha=0.6, label='Reference')
    for k in all_keys:
        if k not in results:
            continue
        ee = results[k]['ee']
        ax.plot(ee[:, 0], ee[:, 1], color=COLORS[k],
                lw=1.8 if 'OBS' in k else 1.0,
                alpha=0.9 if 'OBS' in k else 0.45,
                ls='-' if 'OBS' in k else ':',
                label=LABELS[k])
    _draw_cylinders(ax)
    ax.set_xlabel('X [m]', fontsize=10)
    ax.set_ylabel('Y [m]', fontsize=10)
    ax.set_title('EE Trajectory (XY view)', fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # 충돌 스텝 마킹 (SQP_OBS)
    if 'SQP_DOB_OBS' in results:
        cs = results['SQP_DOB_OBS'].get('collision_steps', np.array([]))
        if len(cs) > 0:
            ee = results['SQP_DOB_OBS']['ee']
            ax.scatter(ee[cs, 0], ee[cs, 1],
                       c='red', s=18, zorder=10, label='Collision pts', marker='x')
            ax.legend(fontsize=8, loc='upper right')

    # ── 오른쪽: 오차 시계열 ──────────────────────────────────────────────────
    ax = axes[1]
    for k in all_keys:
        if k not in results:
            continue
        err = results[k]['err_mm']
        lw  = 2.0 if 'OBS' in k else 1.0
        ls  = '-' if 'OBS' in k else ':'
        al  = 0.95 if 'OBS' in k else 0.45
        ax.plot(t, err, color=COLORS[k], lw=lw, ls=ls, alpha=al,
                label=f'{LABELS[k]}  μ={np.mean(err):.1f}mm')
    ax.axhline(10, color='red', lw=1.2, ls='--', alpha=0.7, label='1 cm limit')
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('EE Error [mm]', fontsize=10)
    ax.set_title('Tracking Error (ref = circle, ignoring obstacle deviation)',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 솔브 시간 비교 텍스트
    txt = "Solve Time Comparison:\n"
    for k in all_keys:
        if k not in results:
            continue
        st = results[k]['solve_t'] * 1e3
        txt += f"  {LABELS[k]}: {np.mean(st):.2f}ms (RTF={20/np.mean(st):.1f}x)\n"
    fig.text(0.50, 0.01, txt, ha='center', fontsize=8,
             family='monospace', color='#333')

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5b — Crossing Obstacle Scenario Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_cross_comparison(results: dict, waypoints: np.ndarray):
    """교차 동적 장애물 시나리오: 3종 컨트롤러 XY 궤적 + 오차 + solve time."""
    cross_keys = [k for k in results if 'CROSS' in k]
    if not cross_keys:
        print("[WARN] 교차 장애물 결과 없음 — plot_cross_comparison 건너뜀")
        return None

    t = np.arange(N_STEPS) * DT

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Scenario 3 — Crossing Dynamic Obstacles\n'
                 '(Circle, 10 s, 2 crossing spheres r=0.030 m, sinusoidal disturbance ON)',
                 fontsize=12, fontweight='bold')

    # ── 왼쪽: XY 궤적 ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(waypoints[:, 0], waypoints[:, 1],
            '--', color='#008844', lw=1.8, alpha=0.6, label='Reference')

    for k in cross_keys:
        ee = results[k]['ee']
        ax.plot(ee[:, 0], ee[:, 1], color=COLORS[k], lw=1.8, alpha=0.9, label=LABELS[k])
        cs = results[k].get('collision_steps', np.array([]))
        if len(cs) > 0:
            ax.scatter(ee[cs, 0], ee[cs, 1],
                       c='red', s=20, marker='x', zorder=10,
                       label=f'{LABELS[k]} collision ({len(cs)})')

    # 교차 장애물 경로 표시 (대표 컨트롤러에서 경로 가져오기)
    ref_key = cross_keys[0]
    if 'cross_paths' in results[ref_key]:
        cp = results[ref_key]['cross_paths']     # (2, N+1, 3)
        for i, clr in enumerate(['#aa2200', '#0022aa']):
            ax.plot(cp[i, :, 0], cp[i, :, 1],
                    color=clr, lw=1.2, ls=':', alpha=0.55,
                    label=f'Obs {i+1} path')
            # 키 프레임 (t=0, 5s, 10s) 위치 표시
            for frame in [0, 250, 499]:
                fi = min(frame, cp.shape[1]-1)
                ax.add_patch(plt.Circle(
                    (cp[i, fi, 0], cp[i, fi, 1]), CROSS_RADIUS,
                    color=clr, alpha=0.25, zorder=4))
        # 안전 마진 원 (교차 순간 t=5s)
        for i, clr in enumerate(['#aa2200', '#0022aa']):
            fi = min(250, cp.shape[1]-1)
            ax.add_patch(plt.Circle(
                (cp[i, fi, 0], cp[i, fi, 1]), _CROSS_SAFE_RADIUS,
                color=clr, fill=False, ls='--', lw=1.0, alpha=0.4, zorder=4))

    ax.set_xlabel('X [m]', fontsize=10)
    ax.set_ylabel('Y [m]', fontsize=10)
    ax.set_title('EE Trajectory (XY) — t=0/5/10 s circles', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # ── 가운데: 추적 오차 ──────────────────────────────────────────────────
    ax = axes[1]
    for k in cross_keys:
        err = results[k]['err_mm']
        ax.plot(t, err, color=COLORS[k], lw=1.8, alpha=0.9,
                label=f'{LABELS[k]}  μ={np.mean(err):.1f}mm')
    ax.axhline(10, color='red', lw=1.2, ls='--', alpha=0.7, label='1 cm limit')
    ax.axvline(5.0, color='gray', lw=1.0, ls=':', alpha=0.6, label='Crossing t=5s')
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('EE Error [mm]', fontsize=10)
    ax.set_title('Tracking Error', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 오른쪽: Solve time ─────────────────────────────────────────────────
    ax = axes[2]
    for k in cross_keys:
        st = results[k]['solve_t'] * 1e3
        ax.plot(t, st, color=COLORS[k], lw=1.2, alpha=0.8,
                label=f'{LABELS[k]}  μ={np.mean(st):.1f}ms')
    ax.axhline(DT * 1e3, color='red', lw=1.5, ls='-.', alpha=0.8,
               label=f'RT limit ({DT*1e3:.0f} ms)')
    ax.axvline(5.0, color='gray', lw=1.0, ls=':', alpha=0.6)
    ax.set_xlabel('Time [s]', fontsize=10)
    ax.set_ylabel('Solve Time [ms]', fontsize=10)
    ax.set_title('Computational Cost — RT limit violations', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 충돌 / solver warn 요약 텍스트
    txt = "Collision / Solver Summary:\n"
    for k in cross_keys:
        cs   = results[k].get('collision_steps', np.array([]))
        warn = int(results[k].get('solver_warns', np.array([0]))[0])
        st   = results[k]['solve_t'] * 1e3
        rt_miss = int(np.sum(st > DT * 1e3))
        txt += (f"  {LABELS[k]}: collision={len(cs)}  "
                f"solver_warn={warn}  RT_miss={rt_miss}\n")
    fig.text(0.50, 0.01, txt, ha='center', fontsize=8,
             family='monospace', color='#333')

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — 3D Animation  (3 subplots + shared error panel)
# ══════════════════════════════════════════════════════════════════════════════

def animate_comparison(results: dict, waypoints: np.ndarray,
                       show_obstacles: bool = False,
                       moving_obs_paths: np.ndarray = None,
                       moving_obs_radius: float = 0.030,
                       show_samples: bool = False):
    """3D animation.
    moving_obs_paths : ndarray (n_obs, N+1, 3) — 매 프레임 실시간 구체 렌더링.
    show_samples     : True이면 MPPI 샘플 궤적 1000개를 반투명 선으로 표시 (4번 전용).
    """
    keys       = list(results.keys())
    n_ctrl     = len(keys)
    use_moving = (moving_obs_paths is not None)
    use_coll   = show_obstacles or use_moving

    _LINK_CLR  = ['#003f8a', '#1560bd', '#2878d8',
                  '#4a94e8', '#70b0f0', '#98ccff']
    _tr_cmap   = plt.get_cmap('plasma')
    # 구체 wireframe용 샘플 각도 (닫힌 원)
    _TH        = np.linspace(0, 2 * math.pi, 60)
    # 위선/경선 각도 오프셋 목록 (각도 단위: rad)
    _LAT_OFFS  = [-math.pi/3, -math.pi/6, 0.0, math.pi/6, math.pi/3]  # 5 lat rings
    _LON_OFFS  = [0.0, math.pi/4, math.pi/2, 3*math.pi/4]             # 4 meridians
    _OBS_CLR   = ['#ff4400', '#0044ff']   # 장애물 A: 주황빨강, B: 파랑

    if use_moving:
        tag = ' + Moving Sphere Obstacles'
    elif show_obstacles:
        tag = ' + Cylinder Obstacles'
    else:
        tag = ''

    fig = plt.figure(figsize=(18, 10), facecolor='white')
    fig.suptitle(f'UR5 3D Simulation — Circle (10 s, disturbance ON{tag})',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(2, n_ctrl, height_ratios=[2.5, 1],
                           hspace=0.30, wspace=0.20, top=0.92, bottom=0.06)
    axes3d = [fig.add_subplot(gs[0, i], projection='3d') for i in range(n_ctrl)]
    ax_err = fig.add_subplot(gs[1, :])

    # ── 3D axes 설정 ──────────────────────────────────────────────────────────
    for ax, k in zip(axes3d, keys):
        ax.set_facecolor('white')
        ax.set_xlim(-0.15, 0.70)
        ax.set_ylim(-0.45, 0.45)
        ax.set_zlim(0.00, 0.80)
        ax.set_xlabel('X', fontsize=6); ax.set_ylabel('Y', fontsize=6)
        ax.set_zlabel('Z', fontsize=6); ax.tick_params(labelsize=5)
        ax.set_title(LABELS[k], fontsize=10, color=COLORS[k], fontweight='bold')
        ax.plot(*waypoints.T, '--', color='#008844', lw=1.2, alpha=0.45)
        ax.grid(True, color='#cccccc', linewidth=0.4)
        if show_obstacles and not use_moving:
            _draw_cylinders_3d(ax, z_bot=0.00, z_top=0.80, alpha=0.20)

    # ── 오차 패널 ─────────────────────────────────────────────────────────────
    all_err = [results[k]['err_mm'] for k in keys]
    y_max   = max(e.max() for e in all_err) * 1.18 + 1.0
    ax_err.set_xlim(0, N_STEPS); ax_err.set_ylim(0, y_max)
    ax_err.set_xlabel('Step', fontsize=9); ax_err.set_ylabel('EE Error [mm]', fontsize=9)
    ax_err.set_title('Tracking Error Comparison', fontsize=10)
    ax_err.grid(True, alpha=0.3)
    for k, err in zip(keys, all_err):
        ax_err.axhline(np.mean(err), color=COLORS[k], lw=1.0, ls=':',
                       label=f'{LABELS[k]}  μ={np.mean(err):.1f}mm')
    if use_coll:
        ax_err.axhline(10, color='red', lw=1.2, ls='--', alpha=0.7, label='1 cm limit')
    ax_err.legend(fontsize=8, loc='upper right')

    # ── 충돌 집합 ─────────────────────────────────────────────────────────────
    collision_sets = {}
    if use_coll:
        for k in keys:
            cs = results[k].get('collision_steps', np.array([]))
            collision_sets[k] = set(cs.tolist())

    # ── 팔·EE 아티스트 ────────────────────────────────────────────────────────
    arm_all, jt_all, ee_all, err_lines, stxt_all = [], [], [], [], []
    traces = [{'xs': [], 'ys': [], 'zs': [], 'segs': []} for _ in keys]
    for ax, k in zip(axes3d, keys):
        segs = [ax.plot([], [], [], '-', color=_LINK_CLR[j], lw=4.0)[0] for j in range(6)]
        arm_all.append(segs)
        jt,  = ax.plot([], [], [], 'o', color='#cc2200', ms=5, zorder=9)
        eed, = ax.plot([], [], [], 'D', color='#e06000', ms=7, zorder=10)
        jt_all.append(jt); ee_all.append(eed)
        stxt_all.append(ax.text2D(0.03, 0.95, '', transform=ax.transAxes,
                                  fontsize=8, color='#111'))
    for k in keys:
        ln, = ax_err.plot([], [], color=COLORS[k], lw=1.8, alpha=0.9)
        err_lines.append(ln)

    step_label = fig.text(0.50, 0.955, '', ha='center', fontsize=10,
                          color='#333', fontweight='bold')

    # ── 움직이는 구체 아티스트 ────────────────────────────────────────────────
    # sphere_arts[ax_idx][obs_idx] = {
    #   'lats': [Line3D × 5],   위선 5개
    #   'lons': [Line3D × 4],   경선 4개
    #   'ctr' : Line3D,         중심 마커
    #   'safe': Line3D,         안전 마진 적도 (점선)
    # }
    sphere_arts: list = []
    if use_moving:
        n_obs = moving_obs_paths.shape[0]
        for ax in axes3d:
            per_ax = []
            for i in range(n_obs):
                clr  = _OBS_CLR[i % len(_OBS_CLR)]
                lats = [ax.plot([], [], [], '-',  color=clr, lw=2.2, alpha=0.85,  zorder=8)[0]
                        for _ in _LAT_OFFS]
                lons = [ax.plot([], [], [], '-',  color=clr, lw=2.2, alpha=0.85,  zorder=8)[0]
                        for _ in _LON_OFFS]
                ctr, = ax.plot([], [], [], 'o', color=clr, ms=11, alpha=1.0, zorder=9,
                               markeredgecolor='white', markeredgewidth=1.2)
                safe, = ax.plot([], [], [], '--', color=clr, lw=1.0, alpha=0.35, zorder=7)
                per_ax.append({'lats': lats, 'lons': lons, 'ctr': ctr, 'safe': safe})
            sphere_arts.append(per_ax)

    # ── MPPI 샘플 궤적 아티스트 ──────────────────────────────────────────────
    # NaN-separator 트릭: 100개 × (T_h+1) 포인트짜리 단일 선 → set_data 1번으로 갱신
    _K_DISP = 100   # 화면에 표시할 샘플 수 (npz에는 1000개 저장)
    sample_arts: list = []  # sample_arts[ax_idx] = {'bulk': Line3D, 'top': Line3D}

    if show_samples:
        # 데이터가 있는 첫 번째 컨트롤러에서 T_h 크기 파악
        _T_h = 0
        for k in keys:
            es = results[k].get('ee_samples')
            if es is not None and len(es) > 0:
                _T_h = es.shape[2]
                break
        _seg = _T_h + 1              # 궤적 1개당 포인트 수 (끝에 NaN 삽입)
        _nan_xs = np.full(_K_DISP * _seg, np.nan)
        _nan_ys = _nan_xs.copy()
        _nan_zs = _nan_xs.copy()

        for ax in axes3d:
            # 하위 샘플: 연한 회색 반투명
            bulk, = ax.plot(_nan_xs, _nan_ys, _nan_zs,
                            '-', color='#666666', lw=0.8, alpha=0.55, zorder=3)
            # 상위 3개 (가중치 최상위): 밝은 노란색
            top10, = ax.plot([], [], [], '-', color='#ffdd00',
                             lw=1.6, alpha=0.90, zorder=4)
            sample_arts.append({'bulk': bulk, 'top10': top10, 'T_h': _T_h})

    def _fill_nan_lines(xs_buf, ys_buf, zs_buf, ee_s, T_h):
        """ee_s: (K, T_h, 3) → NaN-separated flat arrays."""
        seg = T_h + 1
        for i in range(ee_s.shape[0]):
            base = i * seg
            xs_buf[base:base + T_h] = ee_s[i, :, 0]
            ys_buf[base:base + T_h] = ee_s[i, :, 1]
            zs_buf[base:base + T_h] = ee_s[i, :, 2]
            # NaN은 초기화 시 이미 들어 있음
        return xs_buf, ys_buf, zs_buf

    # ── init ─────────────────────────────────────────────────────────────────
    def _blank_line(ln):
        ln.set_data([], []); ln.set_3d_properties([])

    def init():
        for segs in arm_all:
            for s in segs: _blank_line(s)
        for a in jt_all + ee_all: _blank_line(a)
        for ln in err_lines: ln.set_data([], [])
        for txt in stxt_all: txt.set_text('')
        step_label.set_text('')
        if use_moving:
            for per_ax in sphere_arts:
                for ob in per_ax:
                    for ln in ob['lats'] + ob['lons']:
                        _blank_line(ln)
                    _blank_line(ob['ctr']); _blank_line(ob['safe'])
        if show_samples:
            for sa in sample_arts:
                _blank_line(sa['bulk']); _blank_line(sa['top10'])
        return []

    # ── update ───────────────────────────────────────────────────────────────
    def update(frame):
        # ── 구체 위치 갱신 (가장 먼저 — arm 뒤에 가릴 수 있으므로 z-order로 처리)
        if use_moving:
            fi  = min(frame, moving_obs_paths.shape[1] - 1)
            r   = moving_obs_radius
            r_s = r + _CROSS_MARGIN   # 안전 마진 반경
            for per_ax in sphere_arts:
                for i, ob in enumerate(per_ax):
                    cx, cy, cz = moving_obs_paths[i, fi]
                    # ① 위선 (latitude rings): z 고정 평면의 원
                    for lat_ln, lat_off in zip(ob['lats'], _LAT_OFFS):
                        r_lat = r * math.cos(lat_off)
                        z_lat = cz + r * math.sin(lat_off)
                        lat_ln.set_data(cx + r_lat * np.cos(_TH),
                                        cy + r_lat * np.sin(_TH))
                        lat_ln.set_3d_properties(np.full(len(_TH), z_lat))
                    # ② 경선 (meridian circles): 수직 대원
                    for lon_ln, lon_off in zip(ob['lons'], _LON_OFFS):
                        lon_ln.set_data(cx + r * np.cos(_TH) * math.cos(lon_off),
                                        cy + r * np.cos(_TH) * math.sin(lon_off))
                        lon_ln.set_3d_properties(cz + r * np.sin(_TH))
                    # ③ 중심 마커
                    ob['ctr'].set_data([cx], [cy])
                    ob['ctr'].set_3d_properties([cz])
                    # ④ 안전 마진 적도 (점선)
                    ob['safe'].set_data(cx + r_s * np.cos(_TH),
                                        cy + r_s * np.sin(_TH))
                    ob['safe'].set_3d_properties(np.full(len(_TH), cz))

        # ── 팔 + EE + trace 갱신
        for idx, (ax, k, segs, jt, eed, tr, err_ln, txt) in enumerate(
                zip(axes3d, keys, arm_all, jt_all,
                    ee_all, traces, err_lines, stxt_all)):
            q_h = results[k]['q']
            if frame >= len(q_h):
                continue
            pos = fk_joints_np(q_h[frame])

            for j in range(6):
                p0, p1 = pos[j], pos[j + 1]
                segs[j].set_data([p0[0], p1[0]], [p0[1], p1[1]])
                segs[j].set_3d_properties([p0[2], p1[2]])

            is_coll  = use_coll and (frame in collision_sets.get(k, set()))
            ee_color = '#ff0000' if is_coll else '#e06000'
            ee_size  = 14 if is_coll else 7
            eed.set_color(ee_color); eed.set_markersize(ee_size)
            jt.set_data(pos[:-1, 0], pos[:-1, 1])
            jt.set_3d_properties(pos[:-1, 2])
            eed.set_data([pos[-1, 0]], [pos[-1, 1]])
            eed.set_3d_properties([pos[-1, 2]])

            tr['xs'].append(pos[-1, 0])
            tr['ys'].append(pos[-1, 1])
            tr['zs'].append(pos[-1, 2])
            if len(tr['xs']) >= 2:
                seg_color = '#ff2200' if is_coll else _tr_cmap(frame / max(N_STEPS, 1))
                seg_ln, = ax.plot(tr['xs'][-2:], tr['ys'][-2:], tr['zs'][-2:],
                                  '-', color=seg_color, lw=2.0, alpha=0.88, zorder=6)
                tr['segs'].append(seg_ln)

            if frame > 0:
                n = min(frame, len(all_err[idx]))
                err_ln.set_data(np.arange(n), all_err[idx][:n])

            e_now   = all_err[idx][frame - 1] if frame > 0 else 0.0
            col_tag = '  ⚠ COLLISION' if is_coll else ''
            txt.set_text(f'err = {e_now:.1f} mm{col_tag}')

        # ── MPPI 샘플 궤적 갱신 ──────────────────────────────────────────────
        if show_samples:
            for sa, k in zip(sample_arts, keys):
                es_all = results[k].get('ee_samples')  # (N_STEPS, K_vis, T_h, 3)
                if es_all is None or frame == 0 or frame > len(es_all):
                    continue
                fi = min(frame - 1, len(es_all) - 1)
                es = es_all[fi]          # (K_vis, T_h, 3) — omega 내림차순 저장
                T_h    = sa['T_h']
                K_vis  = es.shape[0]

                # 상위 K_DISP개 (이미 omega 순 정렬로 저장됨)
                K_d    = min(_K_DISP, K_vis)
                es_d   = es[:K_d]        # (K_d, T_h, 3)

                # bulk: 인덱스 3~K_d (상위 3개 제외 나머지)
                seg = T_h + 1
                xs_b = np.full(K_d * seg, np.nan)
                ys_b = xs_b.copy(); zs_b = xs_b.copy()
                for i in range(3, K_d):
                    b = i * seg
                    xs_b[b:b+T_h] = es_d[i, :, 0]
                    ys_b[b:b+T_h] = es_d[i, :, 1]
                    zs_b[b:b+T_h] = es_d[i, :, 2]
                sa['bulk'].set_data(xs_b, ys_b)
                sa['bulk'].set_3d_properties(zs_b)

                # top3: 가중치 최상위 3개 (밝은 노란색)
                xs_t = np.full(3 * seg, np.nan)
                ys_t = xs_t.copy(); zs_t = xs_t.copy()
                for i in range(min(3, K_d)):
                    b = i * seg
                    xs_t[b:b+T_h] = es_d[i, :, 0]
                    ys_t[b:b+T_h] = es_d[i, :, 1]
                    zs_t[b:b+T_h] = es_d[i, :, 2]
                sa['top10'].set_data(xs_t, ys_t)
                sa['top10'].set_3d_properties(zs_t)

        step_label.set_text(f'Step {frame}/{N_STEPS}   t = {frame * DT:.2f} s')
        return []

    anim = animation.FuncAnimation(
        fig, update,
        frames=N_STEPS + 1,
        init_func=init,
        interval=int(DT * 1000),
        blit=False,
        repeat=True,
    )

    # ── Playback controls ────────────────────────────────────────────────────
    # Space      : pause / resume
    # ← / →     : (paused) −1초 / +1초 이동 (50 프레임)
    # ↑ / ↓     : speed ×2 / ÷2  (min 5 ms, max 2000 ms)
    # r          : reset to frame 0 + pause
    _paused    = [False]
    _cur_frame = [0]          # frame counter shared with update() hook
    _interval  = [int(DT * 1000)]   # current interval [ms]

    # wrap update so we always know the current frame
    _orig_update = update
    def update(frame):          # noqa: F811
        _cur_frame[0] = frame
        return _orig_update(frame)

    def _refresh_label():
        speed = _interval[0] / (DT * 1000)
        tag   = '  ⏸ PAUSED' if _paused[0] else f'  ▶ ×{1/speed:.1f}'
        step_label.set_text(
            f'Step {_cur_frame[0]}/{N_STEPS}   '
            f't = {_cur_frame[0] * DT:.2f} s{tag}'
        )

    def _on_key(event):
        k = event.key

        if k == ' ':                         # ── pause / resume ──
            if _paused[0]:
                anim.resume()
            else:
                anim.pause()
            _paused[0] = not _paused[0]

        elif k == 'right' and _paused[0]:    # ── +1초 (50 프레임) ──
            _cur_frame[0] = min(_cur_frame[0] + 50, N_STEPS)
            update(_cur_frame[0])

        elif k == 'left' and _paused[0]:     # ── −1초 (50 프레임) ──
            _cur_frame[0] = max(_cur_frame[0] - 50, 0)
            update(_cur_frame[0])

        elif k == 'up':                      # ── speed ×2 ──
            _interval[0] = max(5, _interval[0] // 2)
            anim.event_source.interval = _interval[0]

        elif k == 'down':                    # ── speed ÷2 ──
            _interval[0] = min(2000, _interval[0] * 2)
            anim.event_source.interval = _interval[0]

        elif k == 'r':                       # ── reset ──
            anim.pause()
            _paused[0]    = True
            _cur_frame[0] = 0
            update(0)

        else:
            return

        _refresh_label()
        fig.canvas.draw_idle()

    # help overlay (bottom-left)
    fig.text(0.01, 0.01,
             'Space: pause/play   ←→: ±1s (paused)   ↑↓: speed   r: reset',
             fontsize=7, color='#666', va='bottom')

    fig.canvas.mpl_connect('key_press_event', _on_key)

    return fig, anim


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _select_mode() -> str:
    """실행 시 시나리오 선택 메뉴."""
    print()
    print('╔══════════════════════════════════════════════════════╗')
    print('║   UR5 Controller Comparison — 모드 선택              ║')
    print('╠══════════════════════════════════════════════════════╣')
    print('║  [1] 기본 비교  (장애물 없음)                        ║')
    print('║      MPPI | MPPI+DOB | SQP+DOB                      ║')
    print('║                                                      ║')
    print('║  [2] 정적 장애물  (얇은 원통 2개)                    ║')
    print('║      MPPI | MPPI+DOB | SQP+DOB (Soft Constraint)    ║')
    print('║                                                      ║')
    print('║  [3] 교차 동적 장애물  (구체 2개, 수직 교차)         ║')
    print('║      MPPI | MPPI+DOB | SQP+DOB (Soft) ← SQP 치명적 ║')
    print('║                                                      ║')
    print('║  [4] MPPI 비교  (교차 동적 장애물, 논문 그림용)      ║')
    print('║      MPPI | MPPI+DOB                                ║')
    print('║                                                      ║')
    print('║  [5] DOB 효과 비교  (교차 동적 장애물)               ║')
    print('║      MPPI | MPPI+DOB | SQP+DOB (Soft) | SQP (Soft) ║')
    print('╚══════════════════════════════════════════════════════╝')
    while True:
        ch = input('  선택 [1/2/3/4/5] (기본=1): ').strip()
        if ch in ('', '1'):
            return 'base'
        if ch == '2':
            return 'obstacle'
        if ch == '3':
            return 'cross'
        if ch == '4':
            return 'mppi_cross'
        if ch == '5':
            return 'cross_nodob'
        print('  1, 2, 3, 4, 5 중에 입력해주세요.')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        _arg = sys.argv[1].strip()
        if _arg in ('2', 'obstacle'):
            mode = 'obstacle'
        elif _arg in ('3', 'cross'):
            mode = 'cross'
        elif _arg in ('4', 'mppi_cross'):
            mode = 'mppi_cross'
        elif _arg in ('5', 'cross_nodob'):
            mode = 'cross_nodob'
        else:
            mode = 'base'
    else:
        mode = _select_mode()
    run_obstacle    = (mode == 'obstacle')
    run_cross       = (mode == 'cross')
    do_mppi_cross   = (mode == 'mppi_cross')
    run_cross_nodob = (mode == 'cross_nodob')

    print()
    print('=' * 65)
    print('UR5 Controller Comparison Analysis')
    if run_obstacle:
        print('  모드: 정적 장애물  (얇은 원통 2개, 궤적 위)')
        print('  Controllers : MPPI | MPPI+DOB | SQP+DOB (Soft Constraint)')
    elif run_cross:
        print('  모드: 교차 동적 장애물  (구체 r=0.030 m × 2개 수직 교차)')
        print('  Controllers : MPPI | MPPI+DOB | SQP+DOB (Soft Constraint)')
        print('  특징 : t=5s에 두 장애물 동시 교차 → SQP constraint 2개 활성화 → solve time spike')
    elif do_mppi_cross:
        print('  모드: MPPI 비교  (교차 동적 장애물, 논문 그림용)')
        print('  Controllers : MPPI | MPPI+DOB')
        print('  특징 : t=2.5s에 두 장애물 동시 교차 → DOB 유무에 따른 MPPI 성능 비교')
    elif run_cross_nodob:
        print('  모드: DOB 효과 비교  (교차 동적 장애물)')
        print('  Controllers : MPPI | MPPI+DOB | SQP+DOB (Soft) | SQP (Soft, No DOB)')
        print('  특징 : SQP_DOB에서 DOB만 제거 → DOB 유무에 따른 외란 보상 효과 비교')
    else:
        print('  모드: 기본 비교  (장애물 없음)')
        print('  Controllers : Original MPPI | MPPI+DOB | SQP+DOB')
    print('  Trajectory  : circle (r=0.13 m, 10 s, sinusoidal disturbance)')
    print('=' * 65)

    waypoints = _circle(N_STEPS, TRAJ_CENTER, TRAJ_RADIUS)

    # ── Step 1: 결과 로드 or 시뮬레이션 자동 실행 ───────────────────────────
    results = load_or_run(waypoints,
                          run_obstacle=run_obstacle,
                          run_cross=run_cross,
                          do_mppi_cross=do_mppi_cross,
                          run_cross_nodob=run_cross_nodob)

    # ── Step 2: 메트릭 출력 ─────────────────────────────────────────────────
    all_metrics = {}
    RT_LIMIT = DT * 1e3   # 20ms
    print('\n─── Metric Summary ───────────────────────────────────────────────────────────────')
    print('%-28s %7s %7s %7s │ %8s %8s %8s │ %6s %7s' %
          ('Controller', 'MeanErr', 'RMSE', 'MaxErr',
           'Mean(ms)', 'P99(ms)', 'MAX(ms)', 'RTF', 'WC-RTF'))
    print('─' * 88)
    for k, data in results.items():
        acc, spd, rob = compute_metrics(data, waypoints)
        all_metrics[k] = (acc, spd, rob)
        is_mppi = 'SQP' not in k
        wc_rtf  = spd['rtf_p99'] if is_mppi else spd['rtf_wc']
        wc_flag = ' !!!' if wc_rtf < 1.0 else ''
        wc_note = '†' if is_mppi else ' '
        print('%-28s %7.1f %7.1f %7.1f │ %8.2f %8.2f %8.2f │ %5.1fx %6.1fx%s%s' % (
            LABELS[k], acc['mean'], acc['rmse'], acc['max'],
            spd['mean'], spd['p99'], spd['max'],
            spd['rtf'], wc_rtf, wc_note, wc_flag))
    print('─' * 88)
    print(f'  WC-RTF < 1.0 → worst-case 스텝이 {RT_LIMIT:.0f}ms 초과 (실시간 불가) !!!')
    print('  † MPPI WC-RTF: p99 기준 — 상위 1% GPU 스케줄링 지터 제외 (알고리즘 무관)')
    print('    SQP  WC-RTF: max 기준 — 제약 활성화 스파이크 포함 (알고리즘 고유)')

    if run_obstacle:
        print('\n─── 충돌 판정 (장애물 반경 내 진입) ──────────────────────')
        for k in ('MPPI_OBS', 'MPPI_DOB_OBS', 'SQP_SOFT_OBS'):
            if k not in results:
                continue
            cs = results[k].get('collision_steps', np.array([]))
            tag = '✓ 충돌 없음' if len(cs) == 0 else f'✗ {len(cs)}스텝 충돌'
            print(f'  {LABELS[k]:<30} {tag}')
        print()

    if do_mppi_cross:
        print('\n─── 교차 장애물 충돌 판정 ─────────────────────────────────')
        for k in ('MPPI_CROSS', 'MPPI_DOB_CROSS'):
            if k not in results:
                continue
            cs = results[k].get('collision_steps', np.array([]))
            st = results[k]['solve_t'] * 1e3
            rt_miss = int(np.sum(st > RT_LIMIT))
            tag = '✓ 충돌 없음' if len(cs) == 0 else f'✗ {len(cs)}스텝 충돌'
            print(f'  {LABELS[k]:<30} {tag}  RT_miss={rt_miss}')
        print()

    if run_cross:
        print('\n─── 교차 장애물 충돌 판정 / Solver 경고 ──────────────────')
        for k in ('MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS'):
            if k not in results:
                continue
            cs   = results[k].get('collision_steps', np.array([]))
            warn = int(results[k].get('solver_warns', np.array([0]))[0])
            st   = results[k]['solve_t'] * 1e3
            rt_miss = int(np.sum(st > RT_LIMIT))
            tag  = '✓ 충돌 없음' if len(cs) == 0 else f'✗ {len(cs)}스텝 충돌'
            print(f'  {LABELS[k]:<30} {tag}  solver_warn={warn}  RT_miss={rt_miss}')
        print()

    if run_cross_nodob:
        print('\n─── 교차 장애물 충돌 판정 / Solver 경고 (DOB 효과 비교) ──')
        for k in ('MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS', 'SQP_CROSS_NO_DOB'):
            if k not in results:
                continue
            cs   = results[k].get('collision_steps', np.array([]))
            warn = int(results[k].get('solver_warns', np.array([0]))[0])
            st   = results[k]['solve_t'] * 1e3
            rt_miss = int(np.sum(st > RT_LIMIT))
            tag  = '✓ 충돌 없음' if len(cs) == 0 else f'✗ {len(cs)}스텝 충돌'
            print(f'  {LABELS[k]:<36} {tag}  solver_warn={warn}  RT_miss={rt_miss}')
        print()

        # SQP 계열 RT 초과 상세 분석 (장애물 교차 구간 집중)
        CROSS_STEP = int(2.5 / DT)   # t=2.5s → step 125
        CROSS_WIN  = int(1.0 / DT)   # ±1s 윈도우 (step ±50)
        print('─── SQP 계열 RT 초과 상세 분석 (RT limit = {:.0f} ms) ─────────'.format(RT_LIMIT))
        for k in ('SQP_SOFT_CROSS', 'SQP_CROSS_NO_DOB'):
            if k not in results:
                continue
            st      = results[k]['solve_t'] * 1e3
            over    = np.where(st > RT_LIMIT)[0]
            print(f'\n  [{LABELS[k]}]')
            print(f'    총 RT 초과 스텝 수 : {len(over)} / {len(st)}')
            if len(over) > 0:
                over_ms   = st[over]
                overshoot = over_ms - RT_LIMIT
                print(f'    초과량 (ms over)  : mean={overshoot.mean():.2f}  '
                      f'max={overshoot.max():.2f}  (최대 step={over[np.argmax(overshoot)]}  '
                      f't={over[np.argmax(overshoot)]*DT:.2f}s)')
                # 장애물 교차 구간(±1s) 내 초과 여부
                near_cross = over[(over >= CROSS_STEP - CROSS_WIN) &
                                  (over <= CROSS_STEP + CROSS_WIN)]
                print(f'    교차 구간 내 초과  : {len(near_cross)}스텝  '
                      f'(step {CROSS_STEP-CROSS_WIN}~{CROSS_STEP+CROSS_WIN}, '
                      f't={( CROSS_STEP-CROSS_WIN)*DT:.1f}s~{(CROSS_STEP+CROSS_WIN)*DT:.1f}s)')
                # 초과 스텝 목록 (최대 10개)
                show = over[:10] if len(over) > 10 else over
                rows = [f'step {s:>3} (t={s*DT:.2f}s)  {st[s]:.2f}ms  +{st[s]-RT_LIMIT:.2f}ms'
                        for s in show]
                print('    초과 스텝 목록 (최대 10개):')
                for r in rows:
                    print(f'      {r}')
                if len(over) > 10:
                    print(f'      ... 외 {len(over)-10}스텝')
            else:
                print('    RT 초과 없음 ✓')
        print()

    # ── Step 3: 시각화 ──────────────────────────────────────────────────────
    if run_obstacle:
        obs_results = {k: results[k]
                       for k in ('MPPI_OBS', 'MPPI_DOB_OBS', 'SQP_SOFT_OBS')
                       if k in results}
        obs_metrics = {k: all_metrics[k] for k in obs_results}

        fig1     = plot_metric_dashboard(obs_metrics)
        fig2     = plot_timeseries(obs_results, waypoints)
        fig3     = plot_3d_trajectory(obs_results, waypoints, show_obstacles=True)
        fig_obs  = plot_obstacle_comparison(results, waypoints)
        fig4, anim = animate_comparison(obs_results, waypoints, show_obstacles=True)

    elif run_cross:
        cross_results = {k: results[k]
                         for k in ('MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS')
                         if k in results}
        cross_metrics = {k: all_metrics[k] for k in cross_results}

        # 저장된 cross_paths (프레임별 구체 위치) 복원
        _first_cross = next(iter(cross_results.values()))
        if 'cross_paths' in _first_cross:
            _cross_anim_paths = _first_cross['cross_paths']  # (2, N+1, 3)
        else:
            _N = next(iter(cross_results.values()))['q'].shape[0] - 1
            _cross_anim_paths = _crossing_obstacle_paths(_N + 1)

        fig1       = plot_metric_dashboard(cross_metrics)
        fig2       = plot_timeseries(cross_results, waypoints)
        fig3       = plot_3d_trajectory(cross_results, waypoints, show_obstacles=False)
        fig_cross  = plot_cross_comparison(results, waypoints)
        fig4, anim = animate_comparison(cross_results, waypoints,
                                        moving_obs_paths=_cross_anim_paths,
                                        moving_obs_radius=CROSS_RADIUS)

    elif run_cross_nodob:
        nodob_keys    = ('MPPI_CROSS', 'MPPI_DOB_CROSS', 'SQP_SOFT_CROSS', 'SQP_CROSS_NO_DOB')
        nodob_results = {k: results[k] for k in nodob_keys if k in results}
        nodob_metrics = {k: all_metrics[k] for k in nodob_results}

        _first_nodob = next(iter(nodob_results.values()))
        if 'cross_paths' in _first_nodob:
            _nodob_anim_paths = _first_nodob['cross_paths']
        else:
            _N = _first_nodob['q'].shape[0] - 1
            _nodob_anim_paths = _crossing_obstacle_paths(_N + 1)

        fig1       = plot_metric_dashboard(nodob_metrics)
        fig2       = plot_timeseries(nodob_results, waypoints)
        fig3       = plot_3d_trajectory(nodob_results, waypoints, show_obstacles=False)
        fig_cross  = plot_cross_comparison(nodob_results, waypoints)
        fig4, anim = animate_comparison(nodob_results, waypoints,
                                        moving_obs_paths=_nodob_anim_paths,
                                        moving_obs_radius=CROSS_RADIUS)

    elif do_mppi_cross:
        mc_keys    = ('MPPI_CROSS', 'MPPI_DOB_CROSS')
        mc_results = {k: results[k] for k in mc_keys if k in results}
        mc_metrics = {k: all_metrics[k] for k in mc_results}

        _first_mc = next(iter(mc_results.values()))
        if 'cross_paths' in _first_mc:
            _mc_anim_paths = _first_mc['cross_paths']
        else:
            _N = _first_mc['q'].shape[0] - 1
            _mc_anim_paths = _crossing_obstacle_paths(_N + 1)

        fig1       = plot_metric_dashboard(mc_metrics)
        fig2       = plot_timeseries(mc_results, waypoints)
        fig3       = plot_3d_trajectory(mc_results, waypoints, show_obstacles=False)
        fig_cross  = plot_cross_comparison(mc_results, waypoints)
        fig4, anim = animate_comparison(mc_results, waypoints,
                                        moving_obs_paths=_mc_anim_paths,
                                        moving_obs_radius=CROSS_RADIUS,
                                        show_samples=True)

    else:
        base_results = {k: v for k, v in results.items()
                        if k in ('MPPI', 'MPPI_DOB', 'SQP_DOB')}
        base_metrics = {k: v for k, v in all_metrics.items()
                        if k in ('MPPI', 'MPPI_DOB', 'SQP_DOB')}
        fig1     = plot_metric_dashboard(base_metrics)
        fig2     = plot_timeseries(base_results, waypoints)
        fig3     = plot_3d_trajectory(base_results, waypoints)
        fig4, anim = animate_comparison(base_results, waypoints)

    plt.show()
