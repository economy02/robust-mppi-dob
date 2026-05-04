"""
sqp_soft_controller.py — SQP + Soft Obstacle Constraint + DOB

sqp_controller.py 의 UR5SQPTracker를 상속하여
- nlp_solver_type = 'SQP'  (full multi-iteration, max_iter=15)
- soft obstacle constraint  (slack 변수, 항상 feasible)
두 가지를 추가한 버전.

MPPI 대비 연산 시간 급증 시연 목적 (장애물 구간 worst-case 비교).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ── 원본에서 모든 공통 코드 재사용 (불변) ────────────────────────────────────
from sqp_controller import (
    _casadi_fk, _I_EFF, _B_DAMP, _TAU_MAX, _DQ_MAX,
    DT, HORIZON, SQP_MAX_ITER, W_POS, W_VEL, W_ACT, W_TERM,
    UR5SQPTracker,
)

# ── 장애물 정의 (MPPI 파일과 동기화) ─────────────────────────────────────────
OBSTACLES = [
    {'center': (0.492,  0.092), 'r_obs': 0.008},   # θ=45°
    {'center': (0.308, -0.092), 'r_obs': 0.008},   # θ=225°
]
SOFT_OBS_MARGIN = 0.045  # [m] 장애물 안전 마진

# ── Soft constraint 페널티 ────────────────────────────────────────────────────
# 페널티 스케일 분석:
#   s_l (장애물 중심) = r² = (0.008)² = 6.4e-5
#   penalty @ center  = 0.5 × W_SLACK × s_l² ≈ W_SLACK × 2e-9
#   tracking cost (1mm error) = 0.5 × W_POS × 3 × (0.001)² ≈ 1.1e-2
#
#   W_SLACK = 5e8 → penalty ≈ 1.0  (90× 1mm 추적 오차) → 회피 시도
#   → 그래도 얇은 장애물(r=0.008)은 선형화 오차로 통과 가능
W_SLACK = 5e11  # 극도로 높은 페널티 → SQP가 장애물 회피에 전력 투구
                # 장애물 중심 페널티 ≈ 5e11 × 2e-9 = 1000 (1mm 추적 오차의 9만 배)
                # → 솔버가 constraint 수렴에 전체 이터레이션 소모 → 연산 시간 급증

_ACADOS_OUT_SOFT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'c_generated_code_ur5sqp_soft')


# ═══════════════════════════════════════════════════════════════════════════════
# Soft 장애물 제약이 추가된 OCP 빌더
# ═══════════════════════════════════════════════════════════════════════════════

def build_ur5_acados_ocp_soft(x0: np.ndarray) -> AcadosOcp:
    """
    원본 OCP에 soft 장애물 제약 추가.
    model.name = 'ur5_sqp_soft' → 원본과 별도 C 코드 컴파일.
    """
    q   = ca.SX.sym('q',   6)
    dq  = ca.SX.sym('dq',  6)
    x   = ca.vertcat(q, dq)
    tau = ca.SX.sym('tau', 6)

    # 동역학 (원본과 동일)
    I_ca   = ca.DM(_I_EFF.tolist())
    B_ca   = ca.DM(_B_DAMP.tolist())
    f_expl = ca.vertcat(dq, (tau - B_ca * dq) / I_ca)

    # 비용 출력식 (원본과 동일)
    ee      = _casadi_fk(q)
    J_ee    = ca.jacobian(ee, q)
    v_ee    = ca.mtimes(J_ee, dq)
    y_expr  = ca.vertcat(ee, v_ee, tau)   # 12D
    y_expr_e = ee                          # 3D

    # ── Soft 장애물 제약: h(q) = dist² - r_safe² ≥ 0 ─────────────────────
    n_obs  = len(OBSTACLES)
    h_list = []
    for obs in OBSTACLES:
        cx, cy = obs['center']
        r_safe = obs['r_obs'] + SOFT_OBS_MARGIN
        dist_sq = (ee[0] - cx)**2 + (ee[1] - cy)**2
        h_list.append(dist_sq - r_safe**2)   # 양수 = 안전, 음수 = 안전 마진 침범
    h_expr = ca.vertcat(*h_list)

    # ── 모델 ─────────────────────────────────────────────────────────────
    model = AcadosModel()
    model.name           = 'ur5_sqp_soft'   # 원본('ur5_sqp')과 다른 이름
    model.x              = x
    model.u              = tau
    model.xdot           = ca.SX.sym('xdot', 12)
    model.f_expl_expr    = f_expl
    model.f_impl_expr    = model.xdot - f_expl
    model.cost_y_expr    = y_expr
    model.cost_y_expr_e  = y_expr_e
    model.con_h_expr     = h_expr   # running stages
    model.con_h_expr_e   = h_expr   # terminal stage

    # ── OCP ──────────────────────────────────────────────────────────────
    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.tf        = HORIZON * DT
    ocp.solver_options.N_horizon = HORIZON

    # 비용 가중치 (원본과 동일)
    ny, ny_e = 12, 3
    ocp.cost.cost_type   = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.W     = np.diag([W_POS]*3 + [W_VEL]*3 + [W_ACT]*6)
    ocp.cost.W_e   = np.diag([W_TERM]*3)
    ocp.cost.yref  = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # 초기 상태 제약
    nx = 12
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = x0.copy()
    ocp.constraints.ubx_0   = x0.copy()

    # 제어·상태 한계 (원본과 동일)
    ocp.constraints.lbu   = -_TAU_MAX
    ocp.constraints.ubu   =  _TAU_MAX
    ocp.constraints.idxbu = np.arange(6)

    lbx, ubx = np.full(nx, -1e9), np.full(nx, 1e9)
    lbx[6:] = -_DQ_MAX; ubx[6:] = _DQ_MAX
    ocp.constraints.lbx   = lbx; ocp.constraints.ubx   = ubx
    ocp.constraints.idxbx = np.arange(nx)
    ocp.constraints.lbx_e = lbx; ocp.constraints.ubx_e = ubx
    ocp.constraints.idxbx_e = np.arange(nx)

    # ── Soft constraint 설정 ─────────────────────────────────────────────
    # h(x) ≥ 0 (dist ≥ r_obs 강제), 위반 시 슬랙 페널티
    ocp.constraints.lh   = np.zeros(n_obs)      # 하한: 0
    ocp.constraints.uh   = np.full(n_obs, 1e9)  # 상한: 없음
    ocp.constraints.lh_e = np.zeros(n_obs)
    ocp.constraints.uh_e = np.full(n_obs, 1e9)

    # 모든 h 제약을 soft로 (hard로 하면 infeasible 위험)
    ocp.constraints.idxsh   = np.arange(n_obs)
    ocp.constraints.idxsh_e = np.arange(n_obs)

    # 슬랙 페널티: L2(Zl) — 낮게 설정해서 추적 비용이 우세하도록
    ocp.cost.Zl   = W_SLACK * np.ones(n_obs)
    ocp.cost.Zu   = np.zeros(n_obs)
    ocp.cost.zl   = np.zeros(n_obs)
    ocp.cost.zu   = np.zeros(n_obs)
    ocp.cost.Zl_e = W_SLACK * np.ones(n_obs)
    ocp.cost.Zu_e = np.zeros(n_obs)
    ocp.cost.zl_e = np.zeros(n_obs)
    ocp.cost.zu_e = np.zeros(n_obs)

    # ── 솔버 옵션 ────────────────────────────────────────────────────────
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type       = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.nlp_solver_max_iter   = SQP_MAX_ITER
    ocp.solver_options.tol         = 1e-4
    ocp.solver_options.qp_tol     = 1e-4
    ocp.solver_options.print_level = 0

    try:
        ocp.code_gen_opts.code_export_directory = _ACADOS_OUT_SOFT
    except AttributeError:
        ocp.code_export_directory = _ACADOS_OUT_SOFT

    return ocp


# ═══════════════════════════════════════════════════════════════════════════════
# Soft 트래커 — 원본 run() 재사용, _build_solver만 override
# ═══════════════════════════════════════════════════════════════════════════════

class UR5SQPSoftTracker(UR5SQPTracker):
    """Soft 장애물 제약 SQP 트래커."""

    def _build_solver(self, x0: np.ndarray) -> AcadosOcpSolver:
        """soft 장애물 제약 OCP를 빌드하고 컴파일된 솔버를 반환한다."""
        print(f"Building soft-constraint OCP "
              f"(W_SLACK={W_SLACK:.0e}, r_safe=r_obs+{SOFT_OBS_MARGIN:.3f}m, "
              f"SQP max_iter={SQP_MAX_ITER}, "
              f"{len(OBSTACLES)} obstacles)...")
        t0  = time.perf_counter()
        ocp = build_ur5_acados_ocp_soft(x0)
        solver = AcadosOcpSolver(
            ocp,
            json_file=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'ur5_sqp_soft_acados_ocp.json'),
            verbose=False,
        )
        print(f"  → soft solver ready in {time.perf_counter()-t0:.1f}s")
        return solver
