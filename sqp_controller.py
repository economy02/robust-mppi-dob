"""
sqp_controller.py — SQP + DOB UR5 6-DOF 궤적 추종 컨트롤러

ICROS 2026 논문 비교 기준 컨트롤러.
acados Full SQP (max_iter=15) + 1차 Q-필터 DOB.
단독 실행 또는 run_comparison.py에서 import하여 사용.

Solver: acados NONLINEAR_LS, ERK/RK4, SQP (multi-iteration)
DOB:    disturbance_observer.DisturbanceObserver (α=40 rad/s)
"""
import math
import sys
import os
import time
import types

# ── Path fix: replicate /usr/bin/python3 sys.path when run inside a venv ─────
for _p in [
    '/home/economy02/.local/lib/python3.10/site-packages',   # user packages (casadi, etc.)
    '/home/economy02/pytorch_mppi/src',                       # pytorch_mppi
    '/home/economy02/mppi_playground/src',
    '/usr/local/lib/python3.10/dist-packages',                # system packages (PIL, etc.)
    '/usr/lib/python3/dist-packages',
    '/home/economy02/mpc/acados/interfaces/acados_template',  # acados template
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ── UR5 DH Parameters (identical to MPPI) ────────────────────────────────────
_DH = [
    ( 0.0,      0.089159,  math.pi/2,  0.0),
    (-0.425,    0.0,       0.0,        0.0),
    (-0.39225,  0.0,       0.0,        0.0),
    ( 0.0,      0.10915,   math.pi/2,  0.0),
    ( 0.0,      0.09465,  -math.pi/2,  0.0),
    ( 0.0,      0.0823,    0.0,        0.0),
]

# ── Physical Parameters (identical to MPPI) ───────────────────────────────────
_I_EFF   = np.array([3.70, 8.40, 2.30, 1.20, 1.40, 0.30])  # 관절별 유효 관성 [kg·m²]
_B_DAMP  = np.array([0.12, 0.12, 0.10, 0.08, 0.08, 0.06])  # 관절별 점성 감쇠 계수 [N·m·s/rad]
_TAU_MAX = np.array([150., 150., 150., 28.,  28.,  28.])    # 관절별 최대 토크 한계 [N·m]
_DQ_MAX  = 6.0  # 관절 최대 각속도 [rad/s]

# ── Simulation (identical to MPPI) ───────────────────────────────────────────
DT      = 0.02   # 제어 주기 [s]
T_SIM   = 10.0   # 총 시뮬레이션 시간 [s]
N_STEPS = int(T_SIM / DT)  # 500

# ── SQP / NMPC Parameters ────────────────────────────────────────────────────
HORIZON      = 25   # NMPC 예측 호라이즌 스텝 수 (same as MPPI for fair comparison)
SQP_MAX_ITER = 15   # Full SQP outer iteration 한도

# ── Cost Weights (same scale as MPPI) ────────────────────────────────────────
W_POS   = 7500.0   # EE position error 가중치
W_VEL   = 2000.0   # EE velocity error 가중치
W_ACT   =    0.002 # torque regularization 가중치
W_TERM  = 10000.0  # terminal EE error 가중치

# ── Trajectory (identical to MPPI) ───────────────────────────────────────────
TRAJ_CENTER = np.array([0.40, 0.0, 0.35])  # 궤적 중심 좌표 [m] (x, y, z)
TRAJ_RADIUS = 0.13  # 궤적 반경 [m]

# ── Disturbance (identical to MPPI) ──────────────────────────────────────────
D_AMP  = (15.0, -12.0, 10.0, 3.0, -3.0,  1.5)  # 관절별 외란 진폭 [N·m]
D_FREQ = ( 1.0,   1.5,  0.8, 1.2,  0.9,  1.1)  # 관절별 외란 주파수 [Hz]

# ── DOB ──────────────────────────────────────────────────────────────────────
from disturbance_observer import DisturbanceObserver, ALPHA_DOB  # α=40, 통합값

# ── acados code export dir ───────────────────────────────────────────────────
_ACADOS_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'c_generated_code_ur5sqp')


# ═══════════════════════════════════════════════════════════════════════════════
# NumPy helpers (dynamics, FK, IK, trajectory)
# ═══════════════════════════════════════════════════════════════════════════════

def fk_numpy(q: np.ndarray) -> np.ndarray:
    """q: (6,) → EE XYZ (3,) — numpy FK (same formula as MPPI batch FK)."""
    T = np.eye(4)
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta = q[i] + th0
        ct, st   = math.cos(theta), math.sin(theta)
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        Ti = np.array([
            [ct, -st*cos_a,  st*sin_a, a*ct],
            [st,  ct*cos_a, -ct*sin_a, a*st],
            [ 0,     sin_a,     cos_a,    d],
            [ 0,         0,         0,    1],
        ])
        T = T @ Ti
    return T[:3, 3]


def fk_joints_np(q: np.ndarray) -> np.ndarray:
    """q: (6,) → (7,3) all joint XYZ positions (for visualization)."""
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta = q[i] + th0
        ct, st   = math.cos(theta), math.sin(theta)
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        Ti = np.array([
            [ct, -st*cos_a,  st*sin_a, a*ct],
            [st,  ct*cos_a, -ct*sin_a, a*st],
            [ 0,     sin_a,     cos_a,    d],
            [ 0,         0,         0,    1],
        ])
        T = T @ Ti
        pts.append(T[:3, 3].copy())
    return np.array(pts)


def _deriv_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    # 시뮬레이터에서 쓰는 단순 joint-space 동역학:
    #   q_dot  = dq
    #   dq_dot = ddq = (u - B*dq) / I
    # 여기서 u는 "실제로 로봇에 가해진 토크"이며,
    # DOB 보상/외란 반영 후의 u_eff가 이 함수로 들어온다.
    dq  = x[6:]
    ddq = (u - _B_DAMP * dq) / _I_EFF
    return np.concatenate([dq, ddq])


def rk4_step(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """RK4 — exact same integration as MPPI._dynamics."""
    # 위의 연속시간 동역학식을 DT 동안 적분해서 다음 상태를 만든다.
    # acados 예측모델도 같은 구조를 쓰므로, 계획 모델과 시뮬레이션 모델을 맞춘 셈이다.
    k1 = _deriv_np(x,             u)
    k2 = _deriv_np(x + .5*DT*k1, u)
    k3 = _deriv_np(x + .5*DT*k2, u)
    k4 = _deriv_np(x +    DT*k3, u)
    ns = x + (DT / 6) * (k1 + 2*k2 + 2*k3 + k4)
    ns[6:] = np.clip(ns[6:], -_DQ_MAX, _DQ_MAX)
    return ns


def get_disturbance(step: int) -> np.ndarray:
    """Sinusoidal disturbance — identical to MPPI."""
    t = step * DT
    return np.array([D_AMP[i] * math.sin(2 * math.pi * D_FREQ[i] * t)
                     for i in range(6)])


# ── Trajectory generators (identical to MPPI) ────────────────────────────────

def _circle(n, c, r):
    """n 포인트 원형 궤적 생성. 반환: (n, 3) ndarray."""
    ts = np.linspace(0, 2*math.pi, n, endpoint=False)
    pts = np.tile(c.copy(), (n, 1))
    pts[:, 0] += r * np.cos(ts)
    pts[:, 1] += r * np.sin(ts)
    return pts

def _infinity(n, c, r):
    """n 포인트 무한대(∞) 형태 리사주 궤적 생성. 반환: (n, 3) ndarray."""
    ts = np.linspace(0, 2*math.pi, n, endpoint=False)
    den = 1 + np.sin(ts)**2
    pts = np.tile(c.copy(), (n, 1))
    pts[:, 0] += r * np.cos(ts) / den
    pts[:, 1] += r * np.sin(ts) * np.cos(ts) / den
    return pts

def _rectangle(n, c, r):
    """n 포인트 직사각형 궤적 생성. 반환: (n, 3) ndarray."""
    w, h = r * 1.6, r
    corners = np.array([
        [c[0]-w, c[1]-h, c[2]],
        [c[0]+w, c[1]-h, c[2]],
        [c[0]+w, c[1]+h, c[2]],
        [c[0]-w, c[1]+h, c[2]],
    ])
    seg = n // 4
    return np.vstack([np.linspace(corners[i], corners[(i+1)%4], seg, endpoint=False)
                      for i in range(4)])

TRAJ_FN = {'circle': _circle, 'infinity': _infinity, 'rectangle': _rectangle}


def _ik_np(target, q0, n_iter=300, tol=1e-5, lam=0.01, alpha=0.5):
    """Damped-least-squares IK — identical to MPPI."""
    q = q0.copy()
    for _ in range(n_iter):
        ee  = fk_numpy(q)
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        eps = 1e-6
        J = np.zeros((3, 6))
        for j in range(6):
            dq_ = q.copy(); dq_[j] += eps
            J[:, j] = (fk_numpy(dq_) - ee) / eps
        JJT = J @ J.T + lam * np.eye(3)
        q   = np.clip(q + alpha * (J.T @ np.linalg.solve(JJT, err)), -math.pi, math.pi)
    return q


# ═══════════════════════════════════════════════════════════════════════════════
# CasADi symbolic FK (for acados NMPC cost)
# ═══════════════════════════════════════════════════════════════════════════════

def _casadi_fk(q_sym) -> ca.SX:
    """
    Symbolic FK using CasADi SX.  q_sym: (6,) SX → EE (3,) SX.
    Uses exact same DH chain as the numpy FK above.
    """
    T = ca.SX.eye(4)
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta  = q_sym[i] + th0
        ct     = ca.cos(theta)
        st     = ca.sin(theta)
        cos_a  = math.cos(alpha)   # constant
        sin_a  = math.sin(alpha)   # constant
        Ti = ca.vertcat(
            ca.horzcat(ct, -st*cos_a,  st*sin_a, a*ct),
            ca.horzcat(st,  ct*cos_a, -ct*sin_a, a*st),
            ca.horzcat(0,   sin_a,     cos_a,    d   ),
            ca.horzcat(0,   0,         0,        1   ),
        )
        T = ca.mtimes(T, Ti)
    return T[:3, 3]


# ═══════════════════════════════════════════════════════════════════════════════
# acados OCP builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_ur5_acados_ocp(x0: np.ndarray) -> AcadosOcp:
    """
    Build acados OCP for UR5 6-DOF SQP control.

    State  x = [q(6), dq(6)]  (12D)
    Control u = τ(6)           (6D)
    Dynamics: ẋ = [dq; (τ - B·dq)/I]  — same simplified model as MPPI

    Cost (NONLINEAR_LS, running):
        y = [EE_pos(3), EE_vel(3), τ(6)]   (12D)
        y_ref = [wp_ref(3), v_ref(3), 0(6)]
        W = diag([W_POS×3, W_VEL×3, W_ACT×6])

    Cost (NONLINEAR_LS, terminal):
        y_e = EE_pos(3)
        W_e = diag([W_TERM×3])
    """
    # ── Symbolic variables ────────────────────────────────────────────────────
    q   = ca.SX.sym('q',   6)
    dq  = ca.SX.sym('dq',  6)
    x   = ca.vertcat(q, dq)    # state 12D
    tau = ca.SX.sym('tau', 6)  # control 6D

    # ── Dynamics (decoupled inertia — same as MPPI) ──────────────────────────
    # acados가 horizon 전체에서 미래 상태를 예측할 때 사용하는 모델.
    # 각 관절을 독립 2차계처럼 단순화해서
    #   ddq = (tau - B*dq) / I
    # 로 두었고, 이 식이 SQP 최적화의 state transition이 된다.
    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    ddq    = (tau - B_ca * dq) / I_ca
    f_expl = ca.vertcat(dq, ddq)

    # ── FK and cost expressions ───────────────────────────────────────────────
    # 상태 x=[q,dq] 자체를 바로 tracking하지 않고,
    # q로부터 얻은 EE 위치/속도와 입력 tau를 cost output으로 사용한다.
    ee   = _casadi_fk(q)              # EE position (3,)
    J_ee = ca.jacobian(ee, q)         # geometric Jacobian (3×6)
    v_ee = ca.mtimes(J_ee, dq)        # EE velocity = J(q) dq

    # Running cost output:
    #   y = [EE_pos, EE_vel, tau]
    # acados의 NONLINEAR_LS는 ||y - y_ref||_W^2 꼴을 최소화하므로,
    # 아래 y_expr 정의가 곧 cost function에 들어갈 물리량 선택이다.
    y_expr   = ca.vertcat(ee, v_ee, tau)   # 12D
    # Terminal cost는 끝 시점 EE 위치만 강하게 맞춘다.
    y_expr_e = ee                          # 3D

    # ── Model ────────────────────────────────────────────────────────────────
    model = AcadosModel()
    model.name           = 'ur5_sqp'
    model.x              = x
    model.u              = tau
    model.xdot           = ca.SX.sym('xdot', 12)
    model.f_expl_expr    = f_expl
    model.f_impl_expr    = model.xdot - f_expl
    model.cost_y_expr    = y_expr
    model.cost_y_expr_e  = y_expr_e

    # ── OCP ──────────────────────────────────────────────────────────────────
    ocp = AcadosOcp()
    ocp.model = model

    N  = HORIZON
    Tf = N * DT

    ocp.solver_options.tf        = Tf
    ocp.solver_options.N_horizon = N

    # Cost type and weights
    # running cost:
    #   (ee - wp)^T W_POS (ee - wp)
    # + (v_ee - v_ref)^T W_VEL (v_ee - v_ref)
    # + tau^T W_ACT tau
    # terminal cost:
    #   (ee_N - wp_N)^T W_TERM (ee_N - wp_N)
    ny   = 12   # 3+3+6
    ny_e = 3    # EE only

    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W     = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e   = np.diag([W_TERM]*3)
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # Initial state equality constraint  x(0) = x0
    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()

    # Control bounds (torque limits) — same as MPPI
    ocp.constraints.lbu   = -_TAU_MAX
    ocp.constraints.ubu   =  _TAU_MAX
    ocp.constraints.idxbu = np.arange(6)

    # State bounds (velocity limits) — stages 1..N
    lbx = np.full(nx, -1e9)
    ubx = np.full(nx,  1e9)
    lbx[6:] = -_DQ_MAX
    ubx[6:] =  _DQ_MAX
    ocp.constraints.lbx   = lbx
    ocp.constraints.ubx   = ubx
    ocp.constraints.idxbx = np.arange(nx)

    # Terminal state bounds
    ocp.constraints.lbx_e   = lbx
    ocp.constraints.ubx_e   = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    # Solver options
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4   # RK4 (same as MPPI)
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.nlp_solver_max_iter   = SQP_MAX_ITER
    ocp.solver_options.tol          = 1e-4
    ocp.solver_options.qp_tol       = 1e-4
    ocp.solver_options.print_level  = 0

    # acados >= 0.5.4 uses code_gen_opts; fallback for older versions
    try:
        ocp.code_gen_opts.code_export_directory = _ACADOS_OUT
    except AttributeError:
        ocp.code_export_directory = _ACADOS_OUT

    return ocp


# ═══════════════════════════════════════════════════════════════════════════════
# SQP Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class UR5SQPTracker:
    """
    acados-based SQP controller for UR5 trajectory tracking.
    Interface mirrors UR5Tracker.run() from UR5_MPPI_DOB_Tuning.py for
    direct comparison.
    """

    def __init__(self, waypoints: np.ndarray):
        """waypoints: (N, 3) EE 목표 위치 배열. horizon 패딩 및 속도 참조 생성."""
        # Pad waypoints for horizon look-ahead
        pad = np.tile(waypoints[-1:], (HORIZON + 2, 1))
        self._wp = np.vstack([waypoints, pad])

        # EE velocity reference (finite difference — same as MPPI)
        vel = np.zeros_like(waypoints)
        vel[:-1] = (waypoints[1:] - waypoints[:-1]) / DT
        vel[-1]  = vel[-2]
        vel_pad  = np.tile(vel[-1:], (HORIZON + 2, 1))
        self._vref = np.vstack([vel, vel_pad])

    def _build_solver(self, x0: np.ndarray) -> AcadosOcpSolver:
        """x0 기준 acados OCP를 빌드하고 컴파일된 솔버를 반환한다."""
        print("Building acados OCP (C code compile, ~30-90s on first run)...")
        t0  = time.perf_counter()
        ocp = build_ur5_acados_ocp(x0)
        solver = AcadosOcpSolver(
            ocp,
            json_file=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'ur5_sqp_acados_ocp.json'),
            verbose=False,
        )
        print(f"  → acados solver ready in {time.perf_counter()-t0:.1f}s")
        return solver

    def run(self, q0: np.ndarray,
            has_disturbance: bool = False,
            use_dob: bool = False):
        """
        Run SQP (±DOB) simulation.

        Returns
        -------
        q_hist      : (N+1, 6)
        ee_hist     : (N+1, 3)
        u_hist      : (N,   6)  applied torque
        err_mm      : (N,)      EE tracking error [mm]
        d_hat_hist  : (N+1, 6) DOB estimate
        d_true_hist : (N+1, 6) true disturbance
        solve_times : (N,)      per-step solver wall time [s]
        """
        x0_np  = np.concatenate([q0, np.zeros(6)])
        solver = self._build_solver(x0_np)

        x_cur  = x0_np.copy()
        dob    = DisturbanceObserver(_I_EFF, _B_DAMP, DT)

        q_hist      = [q0.copy()]
        ee_hist     = [fk_numpy(q0)]
        u_hist      = []
        err_mm      = []
        d_hat_hist  = [dob.d_hat.copy()]
        d_true_hist = [np.zeros(6)]
        solve_times = []

        mode_str = ("SQP+DOB+외란" if use_dob and has_disturbance
                    else "SQP+외란"  if has_disturbance
                    else "SQP 기본")
        print(f"SQP simulation [{mode_str}]: {N_STEPS} steps  "
              f"(horizon={HORIZON}, SQP max_iter={SQP_MAX_ITER})")

        for step in range(N_STEPS):
            # ── Update initial state equality constraint ───────────────────
            solver.set(0, "lbx", x_cur)
            solver.set(0, "ubx", x_cur)

            # ── Update reference for each horizon stage ───────────────────
            for k in range(HORIZON):
                wp   = self._wp  [step + k]
                vref = self._vref[step + k]
                # yref는 위에서 정의한 y=[ee, v_ee, tau]와 같은 순서로 넣는다.
                # 즉 "원하는 EE 위치", "원하는 EE 속도", "0에 가까운 토크"를
                # 동시에 추종하도록 문제를 세팅하는 부분이다.
                yref = np.concatenate([wp, vref, np.zeros(6)])  # 12D
                solver.set(k, "yref", yref)
            # Terminal (3D: EE only)
            solver.set(HORIZON, "yref", self._wp[step + HORIZON])

            # ── Solve ─────────────────────────────────────────────────────
            t0     = time.perf_counter()
            status = solver.solve()
            dt_s   = time.perf_counter() - t0
            solve_times.append(dt_s)

            if status not in (0, 2):
                print(f"  [WARN] acados status={status} at step {step}")

            u_sqp = solver.get(0, "u")

            # ── DOB feedforward compensation ──────────────────────────────
            u_app = np.clip(u_sqp + dob.d_hat if use_dob else u_sqp,
                            -_TAU_MAX, _TAU_MAX)

            # ── True disturbance (same model as MPPI) ─────────────────────
            d_true = get_disturbance(step) if has_disturbance else np.zeros(6)
            # 실제 플랜트에는 외란이 반대 방향으로 작용하므로 유효 토크는
            #   u_eff = u_app - d_true
            # 가 된다. 이 u_eff가 위의 동역학식 ddq=(u-B*dq)/I로 들어간다.
            u_eff  = np.clip(u_app - d_true, -_TAU_MAX, _TAU_MAX)

            # ── RK4 step — same integration as MPPI._dynamics ─────────────
            x_prev = x_cur.copy()
            x_cur  = rk4_step(x_cur, u_eff)

            # ── DOB update ─────────────────────────────────────────────────
            if use_dob:
                dob.update(u_app, x_prev[6:], x_cur[6:])

            # ── Log ───────────────────────────────────────────────────────
            q    = x_cur[:6]
            ee   = fk_numpy(q)
            wp   = self._wp[step + 1]
            e_mm = np.linalg.norm(ee - wp) * 1e3

            q_hist     .append(q.copy())
            ee_hist    .append(ee.copy())
            u_hist     .append(u_app.copy())
            err_mm     .append(e_mm)
            d_hat_hist .append(dob.d_hat.copy())
            d_true_hist.append(d_true.copy())

            if (step + 1) % 50 == 0:
                print(f"  step {step+1:>3}/{N_STEPS}  "
                      f"EE err={e_mm:.1f} mm  solve={dt_s*1e3:.1f} ms")

        print(f"Mean EE err : {np.mean(err_mm):.1f} mm   "
              f"Max: {np.max(err_mm):.1f} mm")
        print(f"Solve time  : mean={np.mean(solve_times)*1e3:.2f} ms/step  "
              f"max={np.max(solve_times)*1e3:.2f} ms/step")

        return (np.array(q_hist), np.array(ee_hist),
                np.array(u_hist), np.array(err_mm),
                np.array(d_hat_hist), np.array(d_true_hist),
                np.array(solve_times))


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison plot (SQP vs SQP+DOB)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sqp_comparison(res_no: dict, res_dob: dict, waypoints: np.ndarray):
    """SQP vs SQP+DOB 결과를 6-panel 2D 비교 플롯으로 출력한다."""
    n_steps = len(res_no["ee"]) - 1
    tg  = np.arange(n_steps + 1) * DT
    tgu = np.arange(n_steps)     * DT

    wp_ref  = waypoints[:n_steps]
    err_no  = np.linalg.norm(res_no ["ee"][1:] - wp_ref, axis=1)
    err_dob = np.linalg.norm(res_dob["ee"][1:] - wp_ref, axis=1)
    print(f"[SQP     ] mean err={err_no.mean()*1e3:.1f} mm  "
          f"final={err_no[-1]*1e3:.1f} mm")
    print(f"[SQP+DOB ] mean err={err_dob.mean()*1e3:.1f} mm  "
          f"final={err_dob[-1]*1e3:.1f} mm")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("UR5 6-DOF SQP (acados) — Disturbance: No-DOB vs With-DOB",
                 fontsize=13)

    # EE trajectory XY
    ax = axes[0, 0]
    ax.plot(wp_ref[:, 0], wp_ref[:, 1], '--k', lw=1.5, label='reference')
    ax.plot(res_no ["ee"][:, 0], res_no ["ee"][:, 1], label='SQP',      alpha=0.8)
    ax.plot(res_dob["ee"][:, 0], res_dob["ee"][:, 1], label='SQP+DOB', lw=2)
    ax.set_aspect('equal'); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title('EE Trajectory (XY)')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    # Tracking error
    ax = axes[0, 1]
    ax.plot(tgu, err_no  * 1e3, label='SQP')
    ax.plot(tgu, err_dob * 1e3, label='SQP+DOB')
    ax.grid(True); ax.legend(); ax.set_title('EE Tracking Error')
    ax.set_xlabel('t [s]'); ax.set_ylabel('mm')

    # Joint angles
    ax = axes[0, 2]
    for i, c in enumerate(['C0', 'C1', 'C2']):
        ax.plot(tg, res_no ["q"][:, i], color=c, ls='--', alpha=0.6,
                label=f'q{i+1} SQP')
        ax.plot(tg, res_dob["q"][:, i], color=c, label=f'q{i+1} DOB')
    ax.grid(True); ax.legend(fontsize=7); ax.set_title('Joint Angles (q1-q3)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('rad')

    # Applied torques
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(tgu, res_no ["u"][:, i], ls='--', alpha=0.6, label=f'u{i+1} SQP')
        ax.plot(tgu, res_dob["u"][:, i], label=f'u{i+1} DOB')
    ax.grid(True); ax.legend(fontsize=7); ax.set_title('Applied Torques (u1-u3) [Nm]')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm')

    # DOB estimates vs true
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(tg, res_dob["d_true"][:, i], ls='--', label=f'd{i+1} true')
        ax.plot(tg, res_dob["d_hat"] [:, i],           label=f'd{i+1} hat')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_title(f'DOB Estimates  (α={ALPHA_DOB})')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm')

    # Solver computation time
    ax = axes[1, 2]
    ax.plot(tgu, res_no ["solve_t"] * 1e3, label='SQP',     alpha=0.8)
    ax.plot(tgu, res_dob["solve_t"] * 1e3, label='SQP+DOB', alpha=0.8)
    ax.axhline(np.mean(res_no ["solve_t"]) * 1e3, color='C0', ls=':', lw=1.2,
               label=f'SQP mean {np.mean(res_no["solve_t"])*1e3:.1f} ms')
    ax.axhline(np.mean(res_dob["solve_t"]) * 1e3, color='C1', ls=':', lw=1.2,
               label=f'DOB mean {np.mean(res_dob["solve_t"])*1e3:.1f} ms')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_title('Solver Computation Time per Step')
    ax.set_xlabel('t [s]'); ax.set_ylabel('ms')

    plt.tight_layout()


def animate_results(q_hist, ee_hist, u_hist, err_mm,
                    waypoints, traj_type,
                    q_hist_compare=None, pos_err_compare=None,
                    label_main='SQP+DOB', label_cmp='SQP'):
    """
    3D animated visualization — same layout as UR5_MPPI_DOB_Tuning.py.
      · 3D panel : robot arm (main=color, compare=gray dashed), EE trace, ref
      · top-right : joint angles
      · bot-right : tracking error (main vs compare)
    """
    import matplotlib.animation as animation

    _LINK_CLR  = ['#003f8a', '#1560bd', '#2878d8', '#4a94e8', '#70b0f0', '#98ccff']
    _JOINT_CLR = '#cc2200'
    _EE_CLR    = '#e06000'
    _REF_CLR   = '#008844'
    _TR_CMAP   = plt.get_cmap('plasma')

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.suptitle(f"UR5 SQP+DOB (acados) — {traj_type.capitalize()}  "
                 f"horizon={HORIZON}  SQP max_iter={SQP_MAX_ITER}",
                 fontsize=14, color='#111111', fontweight='bold')

    gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.38)
    ax3 = fig.add_subplot(gs[:, :2], projection='3d')
    axq = fig.add_subplot(gs[0, 2])
    axe = fig.add_subplot(gs[1, 2])

    # ── 3D panel setup ────────────────────────────────────────────────────────
    ax3.set_facecolor('white')
    ax3.set_xlim(-0.15, 0.70); ax3.set_ylim(-0.45, 0.45); ax3.set_zlim(0, 0.80)
    ax3.set_xlabel('X [m]', fontsize=8); ax3.set_ylabel('Y [m]', fontsize=8)
    ax3.set_zlabel('Z [m]', fontsize=8)
    ax3.set_title('3D Robot Arm (SQP+DOB)', fontsize=10)
    ax3.grid(True, color='#cccccc', linewidth=0.5)
    ax3.plot(*waypoints.T, '--', color=_REF_CLR, lw=1.5, alpha=0.5, label='Reference')

    def _style2d(ax, title):
        ax.set_facecolor('white')
        ax.tick_params(colors='#333', labelsize=7)
        ax.set_title(title, color='#222', fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#bbbbbb')

    # ── Joint angle panel ─────────────────────────────────────────────────────
    _style2d(axq, 'Joint Angles [rad]')
    axq.set_xlim(0, N_STEPS); axq.set_ylim(-math.pi - 0.2, math.pi + 0.2)
    axq.set_xlabel('Step', fontsize=7); axq.grid(True, color='#e0e0e0')
    q_lines = [axq.plot([], [], lw=1.5, label=f'q{i+1}')[0] for i in range(6)]
    axq.legend(fontsize=6, ncol=2, loc='upper right')

    # ── Error panel ───────────────────────────────────────────────────────────
    _style2d(axe, f'EE Tracking Error [mm]  ({label_main} vs {label_cmp})')
    _ymax = max(err_mm) * 1.15 + 1
    if pos_err_compare is not None:
        _ymax = max(_ymax, max(pos_err_compare) * 1.15 + 1)
    axe.set_xlim(0, N_STEPS); axe.set_ylim(0, _ymax)
    axe.set_xlabel('Step', fontsize=7); axe.grid(True, color='#e0e0e0')
    if pos_err_compare is not None:
        axe.plot(np.arange(len(pos_err_compare)), pos_err_compare,
                 color='#888888', lw=1.2, alpha=0.55, ls='--',
                 label=f'{label_cmp} (mean {np.mean(pos_err_compare):.1f} mm)')
    axe.axhline(np.mean(err_mm), color='#cc3333', lw=1, ls=':',
                label=f'{label_main} mean {np.mean(err_mm):.1f} mm')
    axe.legend(fontsize=7)
    err_line, = axe.plot([], [], color=_EE_CLR, lw=1.8)

    # ── 3D artists ────────────────────────────────────────────────────────────
    arm_segs = [ax3.plot([], [], [], '-', color=_LINK_CLR[i], lw=5)[0]
                for i in range(6)]
    cmp_segs = ([ax3.plot([], [], [], '--', color='#999999', lw=2, alpha=0.5)[0]
                 for _ in range(6)]
                if q_hist_compare is not None else [])
    jt_dots,  = ax3.plot([], [], [], 'o', color=_JOINT_CLR, ms=8, zorder=9)
    ee_dot,   = ax3.plot([], [], [], 'D', color=_EE_CLR,    ms=10, zorder=10)
    ref_dot,  = ax3.plot([], [], [], '*', color=_REF_CLR,   ms=14, zorder=10,
                         label='target wp')
    ax3.legend(loc='upper right', fontsize=7)
    step_txt = ax3.text2D(0.02, 0.96, '', transform=ax3.transAxes,
                          color='#111', fontsize=9)

    trace_xs, trace_ys, trace_zs = [], [], []
    ee_trace_segs = []

    def init():
        for seg in arm_segs + cmp_segs:
            seg.set_data([], []); seg.set_3d_properties([])
        jt_dots.set_data([], []); jt_dots.set_3d_properties([])
        ee_dot .set_data([], []); ee_dot .set_3d_properties([])
        ref_dot.set_data([], []); ref_dot.set_3d_properties([])
        for ln in q_lines: ln.set_data([], [])
        err_line.set_data([], [])
        step_txt.set_text('')
        return (*arm_segs, *cmp_segs, jt_dots, ee_dot, ref_dot,
                *q_lines, err_line, step_txt)

    def update(i):
        # ── main arm ─────────────────────────────────────────────────────────
        pos = fk_joints_np(q_hist[i])
        for k in range(6):
            p0, p1 = pos[k], pos[k+1]
            arm_segs[k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            arm_segs[k].set_3d_properties([p0[2], p1[2]])

        # ── compare arm (gray) ───────────────────────────────────────────────
        if q_hist_compare is not None and i < len(q_hist_compare):
            pos_c = fk_joints_np(q_hist_compare[i])
            for k in range(6):
                p0, p1 = pos_c[k], pos_c[k+1]
                cmp_segs[k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
                cmp_segs[k].set_3d_properties([p0[2], p1[2]])

        # ── joints and EE ─────────────────────────────────────────────────────
        jt_dots.set_data(pos[:-1, 0], pos[:-1, 1])
        jt_dots.set_3d_properties(pos[:-1, 2])
        ee_dot.set_data([pos[-1, 0]], [pos[-1, 1]])
        ee_dot.set_3d_properties([pos[-1, 2]])

        # ── EE trace ─────────────────────────────────────────────────────────
        trace_xs.append(pos[-1, 0])
        trace_ys.append(pos[-1, 1])
        trace_zs.append(pos[-1, 2])
        if len(trace_xs) >= 2:
            seg, = ax3.plot(trace_xs[-2:], trace_ys[-2:], trace_zs[-2:],
                            '-', color=_TR_CMAP(i / max(N_STEPS, 1)),
                            lw=2.2, alpha=0.9, zorder=6)
            ee_trace_segs.append(seg)

        # ── reference dot ────────────────────────────────────────────────────
        wp = waypoints[i % len(waypoints)]
        ref_dot.set_data([wp[0]], [wp[1]])
        ref_dot.set_3d_properties([wp[2]])

        # ── joint angles panel ───────────────────────────────────────────────
        steps = np.arange(i + 1)
        for j, ln in enumerate(q_lines):
            ln.set_data(steps, q_hist[:i+1, j])

        # ── error panel ──────────────────────────────────────────────────────
        if i > 0:
            err_line.set_data(np.arange(i), err_mm[:i])

        step_txt.set_text(
            f"Step {i+1}/{N_STEPS} | EE err {err_mm[i-1] if i > 0 else 0:.1f} mm"
        )
        return (*arm_segs, *cmp_segs, jt_dots, ee_dot, ref_dot,
                *q_lines, err_line, step_txt)

    anim = animation.FuncAnimation(
        fig, update, frames=N_STEPS + 1, init_func=init,
        interval=int(DT * 1000), blit=False, repeat=True,
    )
    plt.tight_layout()
    return anim


def save_results(res: dict, prefix: str):
    """Save results to npz for cross-controller comparison."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f'{prefix}_results.npz')
    np.savez(path,
             q=res['q'], ee=res['ee'], u=res['u'],
             d_hat=res['d_hat'], d_true=res['d_true'],
             solve_t=res.get('solve_t', np.array([])))
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    valid = list(TRAJ_FN.keys())
    if len(sys.argv) > 1 and sys.argv[1] in valid:
        traj = sys.argv[1]
    else:
        print("Select trajectory:")
        for k, v in enumerate(valid, 1):
            print(f"  [{k}] {v}")
        ch = input("Enter number (default=1): ").strip()
        traj = valid[int(ch)-1] if ch.isdigit() and 1 <= int(ch) <= len(valid) else valid[0]

    print(f"\nTrajectory : {traj}")
    print(f"DOB        : alpha={ALPHA_DOB}")
    print(f"Disturbance: amp={D_AMP}")
    print(f"Solver     : acados SQP (max_iter={SQP_MAX_ITER})  "
          f"horizon={HORIZON}")

    waypoints = TRAJ_FN[traj](N_STEPS, TRAJ_CENTER, TRAJ_RADIUS)

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    print("\nComputing initial q0 via IK ...")
    q0 = _ik_np(waypoints[0], q_hint)
    q0_ee = fk_numpy(q0)
    print(f"  q0 EE = {np.round(q0_ee, 3)}  (target: {np.round(waypoints[0], 3)})")

    # ── SQP (no DOB) — disturbance present ──────────────────────────────────
    print("\n[1/2] SQP without DOB ...")
    tracker_no = UR5SQPTracker(waypoints)
    (q_no, ee_no, u_no, err_no,
     dhat_no, dtrue_no, st_no) = tracker_no.run(
        q0, has_disturbance=True, use_dob=False)

    # ── SQP + DOB — disturbance present ─────────────────────────────────────
    print("\n[2/2] SQP with DOB ...")
    tracker_dob = UR5SQPTracker(waypoints)
    (q_dob, ee_dob, u_dob, err_dob,
     dhat_dob, dtrue_dob, st_dob) = tracker_dob.run(
        q0, has_disturbance=True, use_dob=True)

    res_no  = dict(q=q_no,  ee=ee_no,  u=u_no,
                   d_hat=dhat_no,  d_true=dtrue_no, solve_t=st_no)
    res_dob = dict(q=q_dob, ee=ee_dob, u=u_dob,
                   d_hat=dhat_dob, d_true=dtrue_dob, solve_t=st_dob)

    save_results(res_no,  f'SQP_{traj}')
    save_results(res_dob, f'SQP_DOB_{traj}')

    # ── 2D 비교 플롯 ─────────────────────────────────────────────────────────
    plot_sqp_comparison(res_no, res_dob, waypoints)

    # ── 3D 애니메이션 (DOB 버전 main, No-DOB 버전 gray 비교) ─────────────────
    anim = animate_results(
        q_dob, ee_dob, u_dob, err_dob,
        waypoints, traj,
        q_hist_compare=q_no,
        pos_err_compare=err_no,
        label_main='SQP+DOB',
        label_cmp='SQP',
    )
    plt.show()
