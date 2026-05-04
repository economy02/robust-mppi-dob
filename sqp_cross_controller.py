"""
sqp_cross_controller.py — SQP + DOB + Soft Constraint (교차 동적 장애물)

run_comparison.py의 full_sqp 모드 전용.
동적으로 교차하는 2개 구체 장애물 환경에서 Full SQP + DOB 성능 측정.

Solver: acados SQP (max_iter=15) + soft h(x) constraint
DOB:    disturbance_observer.DisturbanceObserver (α=40 rad/s)
"""
import math
import os
import sys
import time
import types

# ── Path setup (identical to UR5_SQP_DOB.py) ─────────────────────────────────
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
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ── 공통 물리/시뮬 상수 재사용 ──────────────────────────────────────────────────
from sqp_controller import (
    _I_EFF, _B_DAMP, _TAU_MAX, _DQ_MAX,
    DT, N_STEPS, HORIZON,
    W_POS, W_VEL, W_ACT, W_TERM,
    D_AMP, D_FREQ,
    TRAJ_CENTER, TRAJ_RADIUS,
    _casadi_fk, fk_numpy, rk4_step,
)
from disturbance_observer import DisturbanceObserver, ALPHA_DOB

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Full SQP 파라미터 ─────────────────────────────────────────────────────────
MAX_ITER = 15        # 매 스텝 최대 SQP outer iteration 수 [회]

# ── Soft Obstacle 파라미터 ────────────────────────────────────────────────────
OBSTACLES = [
    {'center': (0.492,  0.092), 'r_obs': 0.008},   # θ=45°
    {'center': (0.308, -0.092), 'r_obs': 0.008},   # θ=225°
]
SOFT_OBS_MARGIN = 0.045   # [m] 장애물 안전 마진
W_SLACK         = 5e11    # slack 페널티 가중치 (sqp_soft_controller.py와 동일)

# ── acados 출력 경로 ──────────────────────────────────────────────────────────
_OUT_DIR  = os.path.join(_DIR, 'c_generated_code_ur5full_sqp')
_JSON     = os.path.join(_DIR, 'ur5_full_sqp_acados_ocp.json')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OCP 빌더  (mpc_3dof.py의 create_ocp()에 해당)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ocp(x0: np.ndarray) -> AcadosOcp:
    """
    UR5 Full SQP OCP 구성.
    · 동역학  : ddq = (tau - B·dq) / I  (ERK4)
    · 비용    : NONLINEAR_LS  [EE(3), v_ee(3), tau(6)]
    · 제약    : soft 장애물 h_i = dist² - r_safe² ≥ 0  (slack 허용)
    · solver  : nlp_solver_type='SQP',  max_iter=MAX_ITER
    """
    n_obs = len(OBSTACLES)

    # ── 심볼릭 변수 ──────────────────────────────────────────────────────────
    q   = ca.SX.sym('q',   6)
    dq  = ca.SX.sym('dq',  6)
    x   = ca.vertcat(q, dq)
    tau = ca.SX.sym('tau', 6)

    # ── 동역학 (simple decoupled: ddq = (tau - B*dq) / I) ────────────────────
    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    f_expl = ca.vertcat(dq, (tau - B_ca * dq) / I_ca)

    # ── 비용 출력식 ───────────────────────────────────────────────────────────
    ee      = _casadi_fk(q)               # EE 위치 (3D)
    J_ee    = ca.jacobian(ee, q)          # EE Jacobian
    v_ee    = ca.mtimes(J_ee, dq)         # EE 속도 (3D)
    y_expr  = ca.vertcat(ee, v_ee, tau)   # stage cost residual (12D)
    y_expr_e = ee                         # terminal cost residual (3D)

    # ── Soft 장애물 제약: h_i = ||EE_xy - obs_i||² - r_safe_i² ≥ 0 ─────────
    h_list = []
    for obs in OBSTACLES:
        cx, cy = obs['center']
        r_safe = obs['r_obs'] + SOFT_OBS_MARGIN
        dist_sq = (ee[0] - cx)**2 + (ee[1] - cy)**2
        h_list.append(dist_sq - r_safe**2)
    h_expr = ca.vertcat(*h_list)          # (n_obs,)

    # ── AcadosModel ──────────────────────────────────────────────────────────
    model = AcadosModel()
    model.name           = 'ur5_full_sqp'
    model.x              = x
    model.u              = tau
    model.xdot           = ca.SX.sym('xdot', 12)
    model.f_expl_expr    = f_expl
    model.f_impl_expr    = model.xdot - f_expl
    model.cost_y_expr    = y_expr
    model.cost_y_expr_e  = y_expr_e
    model.con_h_expr     = h_expr         # running stages
    model.con_h_expr_e   = h_expr         # terminal stage

    # ── AcadosOcp ────────────────────────────────────────────────────────────
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.tf        = HORIZON * DT
    ocp.solver_options.N_horizon = HORIZON

    # 비용 가중치
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W     = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e   = np.diag([W_TERM]*3)
    ocp.cost.yref  = np.zeros(12)
    ocp.cost.yref_e = np.zeros(3)

    # 초기 상태 등식 제약
    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()

    # 제어 한계 (토크)
    ocp.constraints.lbu   = -_TAU_MAX
    ocp.constraints.ubu   =  _TAU_MAX
    ocp.constraints.idxbu = np.arange(6)

    # 상태 한계 (속도)
    lbx = np.full(nx, -1e9); ubx = np.full(nx, 1e9)
    lbx[6:] = -_DQ_MAX;      ubx[6:] =  _DQ_MAX
    ocp.constraints.lbx    = lbx; ocp.constraints.ubx    = ubx
    ocp.constraints.idxbx  = np.arange(nx)
    ocp.constraints.lbx_e  = lbx; ocp.constraints.ubx_e  = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    # Soft 장애물 제약 설정 ─────────────────────────────────────────────────
    # h_i ≥ 0 (하한 0, 상한 없음); 위반 시 slack 허용 → 항상 feasible
    ocp.constraints.lh   = np.zeros(n_obs)
    ocp.constraints.uh   = np.full(n_obs, 1e9)
    ocp.constraints.lh_e = np.zeros(n_obs)
    ocp.constraints.uh_e = np.full(n_obs, 1e9)
    ocp.constraints.idxsh   = np.arange(n_obs)   # 모두 soft
    ocp.constraints.idxsh_e = np.arange(n_obs)

    # slack L2 페널티 (Zl: lower-slack 이차, Zu=0: upper-slack 없음)
    ocp.cost.Zl   = W_SLACK * np.ones(n_obs)
    ocp.cost.Zu   = np.zeros(n_obs)
    ocp.cost.zl   = np.zeros(n_obs)
    ocp.cost.zu   = np.zeros(n_obs)
    ocp.cost.Zl_e = W_SLACK * np.ones(n_obs)
    ocp.cost.Zu_e = np.zeros(n_obs)
    ocp.cost.zl_e = np.zeros(n_obs)
    ocp.cost.zu_e = np.zeros(n_obs)

    # ── Solver 옵션 ───────────────────────────────────────────────────────────
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4   # RK4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'          # ← Full SQP
    ocp.solver_options.nlp_solver_max_iter   = MAX_ITER        # ← 최대 15회
    ocp.solver_options.tol                   = 1e-4
    ocp.solver_options.qp_tol               = 1e-4
    ocp.solver_options.print_level           = 0

    try:
        ocp.code_gen_opts.code_export_directory = _OUT_DIR
    except AttributeError:
        ocp.code_export_directory = _OUT_DIR

    return ocp


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Tracker 클래스  (mpc_3dof.py 스타일 — 상속 없는 독립 클래스)
# ═══════════════════════════════════════════════════════════════════════════════

class UR5FullSQPTracker:
    """
    Full SQP + DOB + Soft Obstacle Constraint 트래커.
    상속 없이 독립적으로 동작. mpc_3dof.py의 구조를 따름.
    """

    def __init__(self, waypoints: np.ndarray, x0: np.ndarray):
        """waypoints와 x0을 받아 OCP를 빌드하고 솔버를 초기화한다."""
        # horizon look-ahead를 위해 waypoints 패딩
        pad         = np.tile(waypoints[-1:], (HORIZON + 2, 1))
        self._wp    = np.vstack([waypoints, pad])

        # EE 속도 reference (유한 차분)
        vel         = np.zeros_like(waypoints)
        vel[:-1]    = (waypoints[1:] - waypoints[:-1]) / DT
        vel[-1]     = vel[-2]
        vel_pad     = np.tile(vel[-1:], (HORIZON + 2, 1))
        self._vref  = np.vstack([vel, vel_pad])

        # solver 빌드
        print(f"[Full SQP] OCP 빌드 중 (첫 실행 시 C코드 컴파일 30~90s) ...")
        t0 = time.perf_counter()
        ocp = build_ocp(x0)
        self._solver = AcadosOcpSolver(ocp, json_file=_JSON, verbose=False)
        print(f"  → solver ready  ({time.perf_counter()-t0:.1f}s)  "
              f"[Full SQP, max_iter={MAX_ITER}, "
              f"n_obs={len(OBSTACLES)}, W_SLACK={W_SLACK:.0e}]")

    def step(self, x_cur: np.ndarray, step_idx: int) -> tuple[np.ndarray, float, int]:
        """
        한 스텝 SQP solve.
        반환: (u_sqp, solve_time_s, acados_status)
        """
        solver = self._solver

        # 초기 상태 등식 제약 업데이트
        solver.set(0, 'lbx', x_cur)
        solver.set(0, 'ubx', x_cur)

        # horizon 각 stage에 reference 설정
        for k in range(HORIZON):
            wp   = self._wp  [step_idx + k]
            vref = self._vref[step_idx + k]
            solver.set(k, 'yref', np.concatenate([wp, vref, np.zeros(6)]))  # 12D

        # terminal reference (EE only, 3D)
        solver.set(HORIZON, 'yref', self._wp[step_idx + HORIZON])

        # Full SQP solve (최대 MAX_ITER회 수렴)
        t0     = time.perf_counter()
        status = solver.solve()
        dt_s   = time.perf_counter() - t0

        u_sqp = solver.get(0, 'u')
        return u_sqp, dt_s, status


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 시뮬레이션 러너  (analysis.py 호환 인터페이스)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_sqp(waypoints: np.ndarray, save_path: str = None) -> dict:
    """
    Full SQP + DOB + Soft Obstacle 시뮬레이션 실행.
    analysis.py의 run_sqp_soft_obstacle()과 동일한 dict 반환.
    """
    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])

    # IK로 초기 관절각 계산
    from sqp_controller import _ik_np
    q0    = _ik_np(waypoints[0], q_hint)
    x0    = np.concatenate([q0, np.zeros(6)])

    tracker = UR5FullSQPTracker(waypoints, x0)

    x_cur       = x0.copy()
    dob         = DisturbanceObserver(_I_EFF, _B_DAMP, DT)
    n_warn      = 0

    q_hist      = [q0.copy()]
    ee_hist     = [fk_numpy(q0)]
    u_hist      = []
    err_mm_list = []
    d_hat_hist  = [dob.d_hat.copy()]
    d_true_hist = [np.zeros(6)]
    solve_times = []
    collision_steps = []

    print(f"\n[SIM] UR5 Full SQP+DOB+Soft  "
          f"(horizon={HORIZON}, max_iter={MAX_ITER}, "
          f"r_safe=r_obs+{SOFT_OBS_MARGIN:.3f}m, W_SLACK={W_SLACK:.0e})")

    for step in range(N_STEPS):
        # ── Full SQP solve ────────────────────────────────────────────────────
        u_sqp, dt_s, status = tracker.step(x_cur, step)
        solve_times.append(dt_s)

        if status not in (0, 2):
            n_warn += 1
            if n_warn <= 10:
                print(f"  [WARN] acados status={status} @ step {step}")

        # ── 외란 계산 ─────────────────────────────────────────────────────────
        t_sim  = step * DT
        d_true = np.array([D_AMP[i] * math.sin(2*math.pi*D_FREQ[i]*t_sim)
                           for i in range(6)])

        # ── DOB 보상 적용 ─────────────────────────────────────────────────────
        u_app  = np.clip(u_sqp + dob.d_hat, -_TAU_MAX, _TAU_MAX)
        u_eff  = np.clip(u_app - d_true, -_TAU_MAX, _TAU_MAX)

        # ── RK4 플랜트 적분 ───────────────────────────────────────────────────
        x_prev = x_cur.copy()
        x_cur  = rk4_step(x_cur, u_eff)

        # ── DOB 업데이트 ──────────────────────────────────────────────────────
        dob.update(u_app, x_prev[6:], x_cur[6:])

        # ── 로그 ─────────────────────────────────────────────────────────────
        q_np  = x_cur[:6]
        ee_np = fk_numpy(q_np)
        wp    = waypoints[(step + 1) % len(waypoints)]
        e_mm  = np.linalg.norm(ee_np - wp) * 1e3

        # 충돌 판정 (EE xy 기준)
        for obs in OBSTACLES:
            cx, cy = obs['center']
            if math.sqrt((ee_np[0]-cx)**2 + (ee_np[1]-cy)**2) < obs['r_obs']:
                collision_steps.append(step)
                break

        q_hist     .append(q_np.copy())
        ee_hist    .append(ee_np.copy())
        u_hist     .append(u_app.copy())
        err_mm_list.append(e_mm)
        d_hat_hist .append(dob.d_hat.copy())
        d_true_hist.append(d_true.copy())

        if (step + 1) % 100 == 0:
            print(f"  step {step+1:>3}/{N_STEPS}  "
                  f"err={e_mm:.1f}mm  "
                  f"solve={dt_s*1e3:.1f}ms")

    solve_ms = np.array(solve_times) * 1e3
    print(f"  → mean err={np.mean(err_mm_list):.1f}mm  "
          f"solve mean={solve_ms.mean():.2f}ms  max={solve_ms.max():.2f}ms  "
          f"collisions={len(collision_steps)}  solver_warns={n_warn}")

    data = dict(
        q               = np.array(q_hist),
        ee              = np.array(ee_hist),
        u               = np.array(u_hist),
        d_hat           = np.array(d_hat_hist),
        d_true          = np.array(d_true_hist),
        solve_t         = np.array(solve_times),
        err_mm          = np.array(err_mm_list),
        collision_steps = np.array(collision_steps, dtype=int),
        solver_warns    = np.array([n_warn]),
    )
    if save_path:
        np.savez(save_path, **data)
        print(f"  Saved: {save_path}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 단독 실행 진입점
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n_steps = N_STEPS
    ts      = np.linspace(0, 2*math.pi, n_steps, endpoint=False)
    wp      = np.column_stack([
        TRAJ_CENTER[0] + TRAJ_RADIUS * np.cos(ts),
        TRAJ_CENTER[1] + TRAJ_RADIUS * np.sin(ts),
        np.full(n_steps, TRAJ_CENTER[2]),
    ])

    result = run_full_sqp(wp, save_path=os.path.join(_DIR, 'full_sqp_results.npz'))

    t = np.arange(n_steps) * DT
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'UR5 Full SQP+DOB+Soft  (horizon={HORIZON}, max_iter={MAX_ITER})')

    ax = axes[0]
    ax.plot(wp[:, 0], wp[:, 1], '--k', lw=1.5, label='ref')
    ax.plot(result['ee'][:, 0], result['ee'][:, 1], lw=1.5, label='Full SQP')
    for obs in OBSTACLES:
        cx, cy = obs['center']
        th = np.linspace(0, 2*math.pi, 64)
        ax.fill(cx + obs['r_obs']*np.cos(th), cy + obs['r_obs']*np.sin(th),
                color='red', alpha=0.4)
    ax.set_aspect('equal'); ax.grid(True); ax.legend()
    ax.set_title('EE Trajectory (XY)'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    axes[1].plot(t, result['err_mm'])
    axes[1].axhline(20, color='red', ls='--', label='20ms RT limit equiv')
    axes[1].set_title('EE Error [mm]'); axes[1].set_xlabel('t [s]'); axes[1].grid(True)

    st_ms = result['solve_t'] * 1e3
    axes[2].plot(t, st_ms, lw=1.0, alpha=0.8)
    axes[2].axhline(DT * 1e3, color='red', ls='-.', label=f'RT {DT*1e3:.0f}ms')
    axes[2].axhline(np.mean(st_ms), color='blue', ls='--',
                    label=f'mean {np.mean(st_ms):.1f}ms')
    axes[2].set_title('Solve Time [ms]'); axes[2].set_xlabel('t [s]')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.show()
