"""
mppi_dob_controller.py — MPPI + DOB UR5 6-DOF 궤적 추종 컨트롤러

ICROS 2026: "DOB-Based MPPI for Robust Trajectory Tracking of Manipulators
            under Disturbance and Dynamic Obstacle Environments"
국민대학교 전자공학과

Architecture:
    MPPI (pytorch_mppi) ─── 샘플링 기반 최적 토크 계산
        └── DOB (TorchDisturbanceObserver) ─── 외란 feedforward 보상

Components:
    UR5Tracker   — MPPI 컨트롤러 + DOB + 비용함수 정의
    fk_batch     — GPU 배치 순방향 기구학 (torch.compile)
    fk_joints_np — 시각화용 전 관절 위치 계산 (numpy)
"""
import math
import sys
import os
import types

# ── Path fix: allow running from the pytorch_mppi venv while reusing user/system packages.
for _p in [
    '/home/economy02/.local/lib/python3.10/site-packages',
    '/home/economy02/pytorch_mppi/src',
    '/home/economy02/mppi_playground/src',
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/lib/python3/dist-packages',
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
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# TF32 활성화: RTX 5060에서 float32 matmul 전용 텐서 코어 사용 → 추가 속도 향상
torch.set_float32_matmul_precision('high')

from pytorch_mppi import MPPI

# ── UR5 DH Parameters (Modified DH) ──────────────────────────────────────────
_DH = [
    ( 0.0,      0.089159,  math.pi/2,  0.0),
    (-0.425,    0.0,       0.0,        0.0),
    (-0.39225,  0.0,       0.0,        0.0),
    ( 0.0,      0.10915,   math.pi/2,  0.0),
    ( 0.0,      0.09465,  -math.pi/2,  0.0),
    ( 0.0,      0.0823,    0.0,        0.0),
]

# ── UR5 Physical Parameters ───────────────────────────────────────────────────
_I_EFF   = (3.70, 8.40, 2.30, 1.20, 1.40, 0.30)  # 관절별 유효 관성 [kg·m²]
_B_DAMP  = (0.12, 0.12, 0.10, 0.08, 0.08, 0.06)  # 관절별 점성 감쇠 계수 [N·m·s/rad]
_TAU_MAX = (150., 150., 150., 28.,  28.,  28.)    # 관절별 최대 토크 한계 [N·m]
_DQ_MAX  = 6.0                                     # 관절속도 포화 한계 [rad/s]

# ── Simulation ────────────────────────────────────────────────────────────────
DT      = 0.02           # 제어 주기 [s]
T_SIM   = 10.0           # 총 시뮬레이션 시간 [s]
N_STEPS = int(T_SIM / DT)  # 총 스텝 수

# ── MPPI Hyperparameters ──────────────────────────────────────────────────────
HORIZON   = 12       # H=10: 0.20s 예측 (장애물모드: HORIZON_OBS=22, 0.44s)
N_SAMPLES = 2000                               # 몬테카를로 샘플 수
LAMBDA    = 0.45                                # 온도 파라미터 (낮을수록 최적 샘플 집중)
NOISE_VAR = (4.0, 7.0, 4.0, 0.8, 0.8, 0.25)  # 관절별 탐색 노이즈 분산 [N·m²]

# ── Cost Weights ──────────────────────────────────────────────────────────────
W_POS    = 80000.0        # 위치 오차 가중치
W_VEL    =     500        # 속도 오차 가중치
W_ACT    =     0.05      # 제어 입력 정규화 가중치
W_TERM   = 120000.0   # 120000→140000: terminal 강화로 circle 추적 SQP 수준 근접
W_SMOOTH =     0.25       # 토크 변화율 평활화 가중치 (기본 모드)

# ── Obstacle Avoidance ────────────────────────────────────────────────────────
# 원통 장애물: 궤적 중간에 배치 (시작점 θ=0° 제외)
#   θ=45°:  [0.492,  0.092]  → 시뮬 1.25s 지점
#   θ=225°: [0.308, -0.092]  → 시뮬 6.25s 지점
OBSTACLES = [
    {'center': (0.492,  0.092), 'r_obs': 0.008},   # θ=45°,  ~1.6cm 직경 얇은 봉
    {'center': (0.308, -0.092), 'r_obs': 0.008},   # θ=225°, ~1.6cm 직경 얇은 봉
]
# Exponential barrier 페널티:
#   dist ≥ r_outer : 0
#   dist < r_outer : W_EXP × (exp(K_EXP × (r_outer − dist)) − 1)
#
#   relu² 대비 개선:
#     · relu²: 경계 기울기=0 → MPPI가 경계 근방 샘플 구분 불가
#     · exp  : 경계 기울기 = −K_EXP × W_EXP (즉시 방향 신호)
#     · pen=SOFT_ZONE(0.03m): W_EXP × (e^1.2 − 1) ≈ 1160
#     · pen=SOFT_ZONE+OBS_MARGIN(0.08m): W_EXP × (e^3.2 − 1) ≈ 11750
W_EXP        = 5e3 # exponential 페널티 진폭
K_EXP        = 1.5   # 지수 기울기 (높을수록 심부에서 급격히 상승)
OBS_MARGIN   = 0.05   # 안전 마진: r_safe = r_obs + 0.05 = 0.058m
SOFT_ZONE    = 0.03  # 검출 구역: r_outer = r_safe + 0.03 = 0.088m
W_SMOOTH_OBS = 0.025   # 장애물 모드 전용 smooth weight (circle 모드보다 크게)

# 동적 crossing 장애물은 analysis.py가 경로를 주입하고, 비용 계산은 여기서 수행한다.
USE_CROSS_OBSTACLE = False  # run_comparison.py가 set_cross_obstacles()로 활성화
CROSS_PATHS = None          # run_comparison.py가 주입하는 장애물 경로 배열
CROSS_SAFE_RADIUS = 0.056   # crossing 장애물 안전 반경 [m]
CROSS_EXP_W = 160.0         # crossing 장애물 exponential 페널티 진폭
CROSS_EXP_K = 38.0          # crossing 장애물 지수 기울기
CROSS_EXP_ZONE = 0.012      # crossing 장애물 검출 구역 두께 [m]

USE_OBSTACLE    = False     # run_comparison.py가 use_obstacle=True로 활성화
# 장애물 모드: 긴 horizon으로 조기 감지 및 smooth 우회 경로 계획
HORIZON_OBS     = 12   # 0.20s 선행 예측, 연산속도 우선
N_SAMPLES_OBS   = 2000 # 더 많은 샘플로 bypass 경로 품질 향상

# ── Trajectory ────────────────────────────────────────────────────────────────
TRAJ_CENTER = np.array([0.40, 0.0, 0.35])  # 작업공간 기준 궤적 중심 [m]
TRAJ_RADIUS = 0.13                          # 궤적 반경 [m]

# ── External Disturbance (6관절) ──────────────────────────────────────────────
D_AMP  = (15.0, -12.0, 10.0, 3.0, -3.0,  1.5)  # 사인파 진폭 [Nm]
D_FREQ = ( 1.0,   1.5,  0.8, 1.2,  0.9,  1.1)  # 주파수 [Hz]

# ── DOB ──────────────────────────────────────────────────────────────────────
from disturbance_observer import TorchDisturbanceObserver, ALPHA_DOB  # α=40, 통합값


# ── Torch Batched FK ──────────────────────────────────────────────────────────

def _dh_mat_batch(a, d, alpha, theta):
    K   = theta.shape[0]
    dev = theta.device; dt = theta.dtype
    ct = torch.cos(theta); st = torch.sin(theta)
    ca = math.cos(alpha);  sa = math.sin(alpha)
    zeros = torch.zeros(K, device=dev, dtype=dt)
    ones  = torch.ones (K, device=dev, dtype=dt)
    row0 = torch.stack([ ct,        -st*ca,   st*sa,  a*ct], dim=1)
    row1 = torch.stack([ st,         ct*ca,  -ct*sa,  a*st], dim=1)
    row2 = torch.stack([zeros, sa*ones, ca*ones, d*ones], dim=1)
    row3 = torch.stack([zeros,  zeros,   zeros,   ones], dim=1)
    return torch.stack([row0, row1, row2, row3], dim=1)


@torch.compile(mode='reduce-overhead')
def fk_batch(q):
    """q: (K,6) → (K,3) EE XYZ.
    @torch.compile: 6번 bmm Python 루프를 단일 fused 커널로 합쳐 CUDA 런치 오버헤드 제거.
    """
    K = q.shape[0]
    T = torch.eye(4, dtype=q.dtype, device=q.device).unsqueeze(0).expand(K, -1, -1).clone()
    for i, (a, d, alpha, th0) in enumerate(_DH):
        T = torch.bmm(T, _dh_mat_batch(a, d, alpha, q[:, i] + th0))
    return T[:, :3, 3]


def fk_and_jacobian_batch(q):
    """q: (K,6) → ee:(K,3), J:(K,3,6)."""
    K = q.shape[0]
    T = torch.eye(4, dtype=q.dtype, device=q.device).unsqueeze(0).expand(K, -1, -1).clone()
    z_axes, origins = [], []
    for i, (a, d, alpha, th0) in enumerate(_DH):
        z_axes .append(T[:, :3, 2].clone())
        origins.append(T[:, :3, 3].clone())
        T = torch.bmm(T, _dh_mat_batch(a, d, alpha, q[:, i] + th0))
    ee = T[:, :3, 3]
    J  = torch.zeros(K, 3, 6, dtype=q.dtype, device=q.device)
    for i in range(6):
        z  = z_axes[i]
        dp = ee - origins[i]
        J[:, 0, i] = z[:, 1]*dp[:, 2] - z[:, 2]*dp[:, 1]
        J[:, 1, i] = z[:, 2]*dp[:, 0] - z[:, 0]*dp[:, 2]
        J[:, 2, i] = z[:, 0]*dp[:, 1] - z[:, 1]*dp[:, 0]
    return ee, J


def fk_joints_np(q):
    """q: (6,) numpy → (7,3) 모든 관절 위치."""
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (a, d, alpha, th0) in enumerate(_DH):
        theta = q[i] + th0
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        Ti = np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1],
        ])
        T = T @ Ti
        pts.append(T[:3, 3].copy())
    return np.array(pts)


# ── Disturbance ───────────────────────────────────────────────────────────────

def get_disturbance(step: int, device, dtype) -> torch.Tensor:
    """사인파 외란 토크 (6관절).
    d(t) = D_AMP * sin(2π * D_FREQ * t)
    """
    t = step * DT
    return torch.tensor(
        [D_AMP[i] * math.sin(2 * math.pi * D_FREQ[i] * t) for i in range(6)],
        dtype=dtype, device=device)


def set_cross_obstacles(paths: np.ndarray = None,
                        safe_radius: float = None,
                        exp_w: float = None,
                        exp_k: float = None,
                        exp_zone: float = None) -> None:
    """analysis.py가 동적 crossing 장애물 경로/가중치를 MPPI 비용함수에 주입한다."""
    global USE_CROSS_OBSTACLE, CROSS_PATHS
    global CROSS_SAFE_RADIUS, CROSS_EXP_W, CROSS_EXP_K, CROSS_EXP_ZONE

    if paths is None:
        USE_CROSS_OBSTACLE = False
        CROSS_PATHS = None
        return

    USE_CROSS_OBSTACLE = True
    CROSS_PATHS = np.asarray(paths, dtype=np.float32)
    if safe_radius is not None:
        CROSS_SAFE_RADIUS = float(safe_radius)
    if exp_w is not None:
        CROSS_EXP_W = float(exp_w)
    if exp_k is not None:
        CROSS_EXP_K = float(exp_k)
    if exp_zone is not None:
        CROSS_EXP_ZONE = float(exp_zone)


# ── Trajectory Generators ─────────────────────────────────────────────────────

def _circle(n, c, r):
    ts  = np.linspace(0, 2*math.pi, n, endpoint=False)
    pts = np.tile(c.copy(), (n, 1))
    pts[:, 0] += r * np.cos(ts)
    pts[:, 1] += r * np.sin(ts)
    return pts

def _infinity(n, c, r):
    ts  = np.linspace(0, 2*math.pi, n, endpoint=False)
    den = 1 + np.sin(ts)**2
    pts = np.tile(c.copy(), (n, 1))
    pts[:, 0] += r * np.cos(ts) / den
    pts[:, 1] += r * np.sin(ts) * np.cos(ts) / den
    return pts

def _rectangle(n, c, r):
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


# ── MPPI Controller ───────────────────────────────────────────────────────────

class UR5Tracker:
    """UR5 6-DOF MPPI + DOB 컨트롤러.

    Attributes
    ----------
    _dob   : TorchDisturbanceObserver — 외란 추정기 (GPU 상주)
    _d_hat : torch.Tensor             — 현재 외란 추정값 (_dob.d_hat 참조)
    ctrl   : MPPI                     — pytorch_mppi MPPI 인스턴스
    """

    def __init__(self, waypoints: np.ndarray,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype  = dtype
        self._step  = 0

        # 장애물 모드: 더 긴 horizon 사용
        _horizon    = HORIZON_OBS  if USE_OBSTACLE else HORIZON
        _n_samples  = N_SAMPLES_OBS if USE_OBSTACLE else N_SAMPLES
        self._horizon = _horizon
        self._n_samples = _n_samples

        wp = torch.tensor(waypoints, dtype=dtype, device=device)
        pad = wp[-1:].expand(_horizon + 2, -1)
        self.waypoints = torch.cat([wp, pad], dim=0)

        vel_np = np.zeros_like(waypoints)
        vel_np[:-1] = (waypoints[1:] - waypoints[:-1]) / DT
        vel_np[-1]  = vel_np[-2]
        rv = torch.tensor(vel_np, dtype=dtype, device=device)
        rv_pad = rv[-1:].expand(_horizon + 2, -1)
        self.ref_vel = torch.cat([rv, rv_pad], dim=0)

        self._I    = torch.tensor(_I_EFF,   dtype=dtype, device=device)
        self._b    = torch.tensor(_B_DAMP,  dtype=dtype, device=device)
        self.u_max = torch.tensor(_TAU_MAX, dtype=dtype, device=device)
        self.u_min = -self.u_max

        self._tau_prev = torch.zeros(6, dtype=dtype, device=device)
        self._dob      = TorchDisturbanceObserver(self._I, self._b, DT)
        self._d_hat    = self._dob.d_hat   # GPU 텐서 참조 (dob와 동기화)
        self._cross_paths = (
            torch.tensor(CROSS_PATHS, dtype=dtype, device=device)
            if USE_CROSS_OBSTACLE and CROSS_PATHS is not None
            else None
        )

        noise_sigma = torch.diag(torch.tensor(NOISE_VAR, dtype=dtype, device=device))
        self.ctrl = MPPI(
            dynamics              = self._dynamics,
            running_cost          = self._running_cost,
            nx                    = 12,
            noise_sigma           = noise_sigma,
            num_samples           = _n_samples,
            horizon               = _horizon,
            lambda_               = LAMBDA,
            device                = device,
            terminal_state_cost   = self._terminal_cost,
            u_min                 = self.u_min,
            u_max                 = self.u_max,
            step_dependent_dynamics = True,
        )

    # ── Dynamics ──────────────────────────────────────────────────────────────

    def _deriv(self, state, tau):
        dq  = state[:, 6:]
        ddq = (tau - self._b * dq) / self._I
        return torch.cat([dq, ddq], dim=1)

    def _dynamics(self, state, action, t):
        tau = action.clamp(self.u_min, self.u_max)
        k1  = self._deriv(state,             tau)
        k2  = self._deriv(state + .5*DT*k1,  tau)
        k3  = self._deriv(state + .5*DT*k2,  tau)
        k4  = self._deriv(state +    DT*k3,  tau)
        ns  = state + (DT / 6) * (k1 + 2*k2 + 2*k3 + k4)
        ns[:, 6:] = ns[:, 6:].clamp(-_DQ_MAX, _DQ_MAX)
        return ns

    # ── Cost ──────────────────────────────────────────────────────────────────

    def _obstacle_cost(self, ee: torch.Tensor, t_idx: int = 0) -> torch.Tensor:
        """Exponential barrier 장애물 페널티.

        dist ≥ r_outer : 0
        dist < r_outer : W_EXP × (exp(K_EXP × (r_outer − dist)) − 1)

        경계에서 즉시 기울기 −K_EXP×W_EXP → relu² 대비 조기 방향 신호.
        장애물 심부 접근 시 지수 급증 → 강한 반발 보장.
        """
        cost = torch.zeros(ee.shape[0], dtype=ee.dtype, device=ee.device)

        if self._cross_paths is not None:
            path_idx = min(self._step + int(t_idx), self._cross_paths.shape[1] - 1)
            centers = self._cross_paths[:, path_idx, :]
            r_outer = CROSS_SAFE_RADIUS + CROSS_EXP_ZONE
            for center in centers:
                diff = ee - center.unsqueeze(0)
                dist = torch.sqrt((diff * diff).sum(dim=1) + 1e-8)
                pen = torch.relu(r_outer - dist)
                cost += CROSS_EXP_W * (torch.exp(CROSS_EXP_K * pen) - 1.0)
            return cost

        for obs in OBSTACLES:
            cx, cy  = obs['center']
            r_outer = obs['r_obs'] + OBS_MARGIN + SOFT_ZONE   # 0.088m
            dist    = torch.sqrt((ee[:, 0] - cx) ** 2 +
                                 (ee[:, 1] - cy) ** 2 + 1e-8)
            pen     = torch.relu(r_outer - dist)
            cost   += W_EXP * (torch.exp(K_EXP * pen) - 1.0)
        return cost

    def _running_cost(self, state, action, t):
        q   = state[:, :6]
        tau = action.clamp(self.u_min, self.u_max)
        wp  = self.waypoints[self._step + t]

        ee = fk_batch(q)

        pos_err = (ee - wp).square().sum(dim=1)
        act_reg = tau.square().sum(dim=1)
        # 장애물 모드에서는 W_SMOOTH_OBS(더 큰 값)를 사용해 급격한 토크 변화를 억제,
        # 자연스러운 우회 경로를 유도한다.
        w_sm   = W_SMOOTH_OBS if USE_OBSTACLE else W_SMOOTH
        smooth = (w_sm * (tau - self._tau_prev).square().sum(dim=1)
                  if t == 0 else 0.0)

        cost = W_POS * pos_err + W_ACT * act_reg + smooth
        if USE_OBSTACLE:
            cost = cost + self._obstacle_cost(ee, t)
        return cost

    def _terminal_cost(self, states, actions):
        last_q  = states[..., -1, :6].reshape(-1, 6)
        wp      = self.waypoints[self._step + self._horizon]
        ee_last = fk_batch(last_q)
        cost    = W_TERM * (ee_last - wp).square().sum(dim=1)
        if USE_OBSTACLE:
            # terminal에 더 강한 장애물 패널티 → horizon 끝점에서 확실한 회피 보장
            cost = cost + self._obstacle_cost(ee_last, self._horizon) * 4.0
        return cost.reshape(states.shape[:2])

    # ── Simulation Loop ───────────────────────────────────────────────────────

    def run(self, q0: torch.Tensor,
            capture_samples: bool = False, n_show: int = 50,
            has_disturbance: bool = False, use_dob: bool = False):
        """한 에피소드 시뮬레이션 실행.

        Returns
          q_hist      : (N+1, 6)
          ee_hist     : (N+1, 3)
          u_hist      : (N,   6)   applied torque (delta_u + d_hat if DOB)
          pos_err_mm  : (N,)
          sample_hist : list or None
          d_hat_hist  : (N+1, 6)  DOB 추정 외란
          d_true_hist : (N+1, 6)  실제 외란
        """
        dq0 = torch.zeros(6, dtype=self.dtype, device=self.device)
        xk  = torch.cat([q0, dq0])

        q_hist      = [q0.cpu().numpy()]
        ee_hist     = [fk_batch(q0.unsqueeze(0))[0].cpu().numpy()]
        u_hist      = []
        pos_err_mm  = []
        sample_hist = [] if capture_samples else None
        d_hat_hist  = [self._d_hat.cpu().numpy().copy()]
        d_true_hist = [np.zeros(6)]

        mode_str = ("DOB+외란" if use_dob and has_disturbance
                    else "외란(No-DOB)" if has_disturbance
                    else "기본")
        print(f"MPPI simulation [{mode_str}]: {N_STEPS} steps "
              f"(K={self._n_samples}, T={self._horizon})")

        for step in range(N_STEPS):
            delta_u = self.ctrl.command(xk).clamp(self.u_min, self.u_max)

            if capture_samples and self.ctrl.states is not None:
                q_all  = self.ctrl.states[0, :, :, :6]
                ee_all = fk_batch(q_all.reshape(-1, 6)).reshape(
                    self._n_samples, self._horizon, 3
                )
                omega  = self.ctrl.omega
                top    = torch.argsort(omega, descending=True)[:n_show]
                w_norm = omega / (omega.sum() + 1e-8)
                ee_mean = (w_norm[:, None, None] * ee_all).sum(dim=0)
                sample_hist.append({
                    'ee':      ee_all[top].cpu().numpy(),
                    'omega':   omega[top].cpu().numpy(),
                    'ee_mean': ee_mean.cpu().numpy(),
                })

            d_true = (get_disturbance(self._step, self.device, self.dtype)
                      if has_disturbance
                      else torch.zeros(6, dtype=self.dtype, device=self.device))

            # DOB 보상 적용
            u_app = delta_u + self._d_hat if use_dob else delta_u
            xk_prev = xk.clone()

            # 외란이 포함된 실제 dynamics
            u_eff = (u_app - d_true).clamp(self.u_min, self.u_max)
            xk    = self._dynamics(xk.unsqueeze(0), u_eff.unsqueeze(0), 0).squeeze(0)

            # DOB 업데이트
            if use_dob:
                self._dob.update(u_app, xk_prev[6:], xk[6:])
                self._d_hat = self._dob.d_hat

            self._tau_prev = delta_u.detach().clone()
            self._step    += 1

            q  = xk[:6]
            ee = fk_batch(q.unsqueeze(0))[0]
            wp = self.waypoints[self._step]
            err_mm = (ee - wp).norm().item() * 1e3

            q_hist    .append(q.cpu().numpy())
            ee_hist   .append(ee.cpu().numpy())
            u_hist    .append(u_app.cpu().numpy())
            pos_err_mm.append(err_mm)
            d_hat_hist .append(self._d_hat.cpu().numpy().copy())
            d_true_hist.append(d_true.cpu().numpy().copy())

            if (step + 1) % 50 == 0:
                print(f"  step {step+1:>3}/{N_STEPS}  EE err={err_mm:.1f} mm")

        print(f"Mean EE err: {np.mean(pos_err_mm):.1f} mm  "
              f"Max: {np.max(pos_err_mm):.1f} mm")

        return (np.array(q_hist), np.array(ee_hist),
                np.array(u_hist), np.array(pos_err_mm), sample_hist,
                np.array(d_hat_hist), np.array(d_true_hist))



# ── 비교 플롯 (No-DOB vs With-DOB) ───────────────────────────────────────────

def plot_comparison(res_no, res_dob, waypoints):
    n_steps = len(res_no["ee"]) - 1   # 실제 스텝 수 (초기값 제외)
    tg  = np.arange(n_steps + 1) * DT
    tgu = np.arange(n_steps)     * DT

    # ee_hist: (N+1, 3) — index 0은 초기 상태, index 1~N이 step 1~N의 결과
    # waypoints: (N, 3) — step 1~N의 목표점
    # → ee_hist[1:] vs waypoints[:n_steps] 로 비교
    wp_ref = waypoints[:n_steps]
    err_no  = np.linalg.norm(res_no["ee"][1:]  - wp_ref, axis=1)
    err_dob = np.linalg.norm(res_dob["ee"][1:] - wp_ref, axis=1)
    print(f"[No-DOB ] mean err={err_no.mean()*1e3:.1f} mm  final={err_no[-1]*1e3:.1f} mm")
    print(f"[With-DOB] mean err={err_dob.mean()*1e3:.1f} mm  final={err_dob[-1]*1e3:.1f} mm")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("UR5 6-DOF MPPI — Disturbance: No-DOB vs With-DOB", fontsize=13)

    # EE 궤적 (XY 투영)
    ax = axes[0, 0]
    ax.plot(wp_ref[:, 0], wp_ref[:, 1], '--k', lw=1.5, label='reference')
    ax.plot(res_no["ee"][:, 0],  res_no["ee"][:, 1],  label='No-DOB',   alpha=0.8)
    ax.plot(res_dob["ee"][:, 0], res_dob["ee"][:, 1], label='With-DOB', lw=2)
    ax.set_aspect('equal'); ax.grid(True); ax.legend(fontsize=8)
    ax.set_title('EE Trajectory (XY)'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')

    # 추적 오차 (step 1~N: ee[1:] vs waypoints)
    ax = axes[0, 1]
    ax.plot(tgu, err_no  * 1e3, label='No-DOB')
    ax.plot(tgu, err_dob * 1e3, label='With-DOB')
    ax.grid(True); ax.legend(); ax.set_title('EE Tracking Error')
    ax.set_xlabel('t [s]'); ax.set_ylabel('mm')

    # 관절각 (q1~q3 대표, 전체 N+1 포인트)
    ax = axes[0, 2]
    for i, c in enumerate(['C0', 'C1', 'C2']):
        ax.plot(tg, res_no["q"][:,i],  color=c, ls='--', alpha=0.6, label=f'q{i+1} NoDOB')
        ax.plot(tg, res_dob["q"][:,i], color=c,           label=f'q{i+1} DOB')
    ax.grid(True); ax.legend(fontsize=7); ax.set_title('Joint Angles (q1-q3)')
    ax.set_xlabel('t [s]'); ax.set_ylabel('rad')

    # 제어 입력 (u1~u3, N 포인트)
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(tgu, res_no["u"][:,i],  ls='--', alpha=0.6, label=f'u{i+1} NoDOB')
        ax.plot(tgu, res_dob["u"][:,i],            label=f'u{i+1} DOB')
    ax.grid(True); ax.legend(fontsize=7); ax.set_title('Applied Torques (u1-u3) [Nm]')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm')

    # DOB 추정 (N+1 포인트: d_hat_hist/d_true_hist)
    ax = axes[1, 1]
    for i in range(3):
        ax.plot(tg, res_dob["d_true"][:, i], ls='--', label=f'd{i+1} true')
        ax.plot(tg, res_dob["d_hat"][:,  i],           label=f'd{i+1} hat')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_title(f'DOB Estimates  (α={ALPHA_DOB})')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm')

    # DOB 추정 오차
    ax = axes[1, 2]
    d_err = res_dob["d_hat"] - res_dob["d_true"]
    for i in range(3):
        ax.plot(tg, d_err[:, i], label=f'err d{i+1}')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.grid(True); ax.legend(fontsize=7)
    ax.set_title('DOB Estimation Error')
    ax.set_xlabel('t [s]'); ax.set_ylabel('Nm')

    plt.tight_layout()


# ── Animation ─────────────────────────────────────────────────────────────────

_LINK_CLR   = ['#003f8a', '#1560bd', '#2878d8', '#4a94e8', '#70b0f0', '#98ccff']
_JOINT_CLR  = '#cc2200'
_EE_CLR     = '#e06000'
_REF_CLR    = '#008844'
_TRACE_CLR  = '#d44000'


def animate_results(q_hist, ee_hist, u_hist, pos_err_mm,
                    waypoints, sample_hist, traj_type,
                    q_hist_compare=None, ee_hist_compare=None,
                    pos_err_no=None):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.suptitle(f"UR5 MPPI — {traj_type.capitalize()} + DOB  "
                 f"(K={N_SAMPLES}, T={HORIZON})",
                 fontsize=14, color='#111111', fontweight='bold')

    gs  = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.38)
    ax3 = fig.add_subplot(gs[:, :2], projection='3d')
    axq = fig.add_subplot(gs[0, 2])
    axe = fig.add_subplot(gs[1, 2])

    ax3.set_facecolor('white')
    ax3.set_xlim(-0.15, 0.70); ax3.set_ylim(-0.45, 0.45); ax3.set_zlim(0, 0.80)
    ax3.set_xlabel('X [m]', fontsize=8); ax3.set_ylabel('Y [m]', fontsize=8)
    ax3.set_zlabel('Z [m]', fontsize=8)
    ax3.set_title('3D Robot Arm', fontsize=10)
    ax3.grid(True, color='#cccccc', linewidth=0.5)
    ax3.plot(*waypoints.T, '--', color=_REF_CLR, lw=1.5, alpha=0.5, label='Reference')

    def _style2d(ax, title):
        ax.set_facecolor('white')
        ax.tick_params(colors='#333', labelsize=7)
        ax.set_title(title, color='#222', fontsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor('#bbbbbb')

    _style2d(axq, 'Joint Angles [rad]')
    axq.set_xlim(0, N_STEPS); axq.set_ylim(-math.pi - 0.2, math.pi + 0.2)
    axq.set_xlabel('Step', fontsize=7); axq.grid(True, color='#e0e0e0')
    q_lines = [axq.plot([], [], lw=1.5, label=f'q{i+1}')[0] for i in range(6)]
    axq.legend(fontsize=6, ncol=2, loc='upper right')

    _style2d(axe, 'EE Tracking Error [mm]  (DOB vs No-DOB)')
    _ymax = max(pos_err_mm) * 1.15 + 1
    if pos_err_no is not None:
        _ymax = max(_ymax, max(pos_err_no) * 1.15 + 1)
    axe.set_xlim(0, N_STEPS); axe.set_ylim(0, _ymax)
    axe.set_xlabel('Step', fontsize=7)
    # No-DOB 전체 곡선을 배경에 정적으로 표시
    if pos_err_no is not None:
        axe.plot(np.arange(len(pos_err_no)), pos_err_no,
                 color='#888888', lw=1.2, alpha=0.55, ls='--',
                 label=f'No-DOB (mean {np.mean(pos_err_no):.1f} mm)')
    axe.axhline(np.mean(pos_err_mm), color='#cc3333', lw=1, ls=':',
                label=f'DOB mean {np.mean(pos_err_mm):.1f} mm')
    axe.legend(fontsize=7); axe.grid(True, color='#e0e0e0')
    err_line, = axe.plot([], [], color=_EE_CLR, lw=1.8, label='DOB (live)')

    n_show    = len(sample_hist[0]['ee']) if sample_hist else 0
    _tr_cmap  = plt.get_cmap('plasma')

    # 나머지 샘플: 얕은 회색  /  상위 3개: 노란색 (별도 Line3D)
    samp_lines = [ax3.plot([], [], [], '-', lw=0.7, alpha=0.18, color='#aaaaaa')[0]
                  for _ in range(n_show)]
    top3_lines = [ax3.plot([], [], [], '-', lw=2.0, alpha=0.92, color='#ffdd00',
                           zorder=5)[0] for _ in range(3)]
    mean_line, = ax3.plot([], [], [], '-', color='#00e5ff', lw=3.0, alpha=0.97,
                          zorder=8, label='weighted mean')
    arm_segs   = [ax3.plot([], [], [], '-', color=_LINK_CLR[i], lw=5)[0]
                  for i in range(6)]
    # No-DOB 비교 팔 (회색 점선)
    cmp_segs   = ([ax3.plot([], [], [], '--', color='#999999', lw=2, alpha=0.55)[0]
                   for _ in range(6)] if q_hist_compare is not None else [])
    jt_dots,   = ax3.plot([], [], [], 'o', color=_JOINT_CLR, ms=8,  zorder=9)
    ee_dot,    = ax3.plot([], [], [], 'D', color=_EE_CLR,    ms=10, zorder=10)
    ee_trace_segs = []
    ref_dot,   = ax3.plot([], [], [], '*', color=_REF_CLR, ms=14, zorder=10,
                          label='target wp')
    ax3.legend(loc='upper right', fontsize=7)
    step_txt = ax3.text2D(0.02, 0.96, '', transform=ax3.transAxes,
                          color='#111', fontsize=9)

    trace_xs, trace_ys, trace_zs = [], [], []

    def init():
        for seg in arm_segs:
            seg.set_data([], []); seg.set_3d_properties([])
        for seg in cmp_segs:
            seg.set_data([], []); seg.set_3d_properties([])
        for sl in samp_lines:
            sl.set_data([], []); sl.set_3d_properties([])
        for tl in top3_lines:
            tl.set_data([], []); tl.set_3d_properties([])
        mean_line.set_data([], []); mean_line.set_3d_properties([])
        jt_dots.set_data([], []);   jt_dots.set_3d_properties([])
        ee_dot.set_data([], []);    ee_dot.set_3d_properties([])
        ref_dot.set_data([], []);   ref_dot.set_3d_properties([])
        for ln in q_lines: ln.set_data([], [])
        err_line.set_data([], [])
        step_txt.set_text('')
        return (*arm_segs, *cmp_segs, *samp_lines, *top3_lines, mean_line,
                jt_dots, ee_dot, ref_dot, *q_lines, err_line, step_txt)

    def update(i):
        pos = fk_joints_np(q_hist[i])
        for k in range(6):
            p0, p1 = pos[k], pos[k+1]
            arm_segs[k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
            arm_segs[k].set_3d_properties([p0[2], p1[2]])

        # No-DOB 비교 팔
        if q_hist_compare is not None and i < len(q_hist_compare):
            pos_c = fk_joints_np(q_hist_compare[i])
            for k in range(6):
                p0, p1 = pos_c[k], pos_c[k+1]
                cmp_segs[k].set_data([p0[0], p1[0]], [p0[1], p1[1]])
                cmp_segs[k].set_3d_properties([p0[2], p1[2]])

        jt_dots.set_data(pos[:-1, 0], pos[:-1, 1])
        jt_dots.set_3d_properties(pos[:-1, 2])
        ee_dot.set_data([pos[-1, 0]], [pos[-1, 1]])
        ee_dot.set_3d_properties([pos[-1, 2]])

        trace_xs.append(pos[-1, 0])
        trace_ys.append(pos[-1, 1])
        trace_zs.append(pos[-1, 2])
        if len(trace_xs) >= 2:
            seg, = ax3.plot(trace_xs[-2:], trace_ys[-2:], trace_zs[-2:],
                            '-', color=_tr_cmap(i / max(N_STEPS, 1)),
                            lw=2.2, alpha=0.9, zorder=6)
            ee_trace_segs.append(seg)

        if sample_hist and i < len(sample_hist):
            sh = sample_hist[i]
            # 나머지 샘플: 얕은 회색
            for j, sl in enumerate(samp_lines):
                sl.set_data(sh['ee'][j, :, 0], sh['ee'][j, :, 1])
                sl.set_3d_properties(sh['ee'][j, :, 2])
            # 상위 3개: 노란색
            for rank, tl in enumerate(top3_lines):
                tl.set_data(sh['ee'][rank, :, 0], sh['ee'][rank, :, 1])
                tl.set_3d_properties(sh['ee'][rank, :, 2])
            em = sh['ee_mean']
            mean_line.set_data(em[:, 0], em[:, 1])
            mean_line.set_3d_properties(em[:, 2])

        wp = waypoints[i % len(waypoints)]
        ref_dot.set_data([wp[0]], [wp[1]])
        ref_dot.set_3d_properties([wp[2]])

        steps = np.arange(i + 1)
        for j, ln in enumerate(q_lines):
            ln.set_data(steps, q_hist[:i+1, j])
        if i > 0:
            err_line.set_data(np.arange(i), pos_err_mm[:i])

        step_txt.set_text(
            f"Step {i+1}/{N_STEPS} | EE err {pos_err_mm[i-1] if i>0 else 0:.1f} mm"
        )
        return (*arm_segs, *cmp_segs, *samp_lines, *top3_lines, mean_line,
                jt_dots, ee_dot, ref_dot, *q_lines, err_line, step_txt)

    anim = animation.FuncAnimation(
        fig, update, frames=N_STEPS + 1, init_func=init,
        interval=int(DT * 1000), blit=False, repeat=True,
    )
    plt.tight_layout()
    return anim


# ── IK ────────────────────────────────────────────────────────────────────────

def _ik_np(target, q0, n_iter=300, tol=1e-5, lam=0.01, alpha=0.5):
    q = q0.copy()
    for _ in range(n_iter):
        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        ee = fk_batch(qt)[0].numpy()
        err = target - ee
        if np.linalg.norm(err) < tol:
            break
        eps = 1e-6
        J = np.zeros((3, 6))
        for j in range(6):
            dq = q.copy(); dq[j] += eps
            ee2 = fk_batch(torch.tensor(dq, dtype=torch.float64).unsqueeze(0))[0].numpy()
            J[:, j] = (ee2 - ee) / eps
        JJT = J @ J.T + lam * np.eye(3)
        q   = np.clip(q + alpha * (J.T @ np.linalg.solve(JJT, err)), -math.pi, math.pi)
    return q


# ── Entry Point ───────────────────────────────────────────────────────────────

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  Trajectory: {traj}")
    print(f"DOB: alpha={ALPHA_DOB}  disturbance_amp={D_AMP}")

    waypoints = TRAJ_FN[traj](N_STEPS, TRAJ_CENTER, TRAJ_RADIUS)

    q_hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    print("Computing initial q0 via IK ...")
    q0_np = _ik_np(waypoints[0], q_hint)
    q0_ee = fk_batch(torch.tensor(q0_np, dtype=torch.float64).unsqueeze(0))[0].numpy()
    print(f"  q0 EE = {np.round(q0_ee, 3)}  (target: {np.round(waypoints[0], 3)})")
    q0 = torch.tensor(q0_np, dtype=torch.float64, device=device)

    # ── No-DOB 실행 ──────────────────────────────────────────────────────────
    print("\n[1/2] No-DOB run ...")
    tracker_no = UR5Tracker(waypoints, device)
    (q_no, ee_no, u_no, err_no, _, dhat_no, dtrue_no) = tracker_no.run(
        q0, has_disturbance=True, use_dob=False
    )

    # ── With-DOB 실행 ─────────────────────────────────────────────────────────
    print("\n[2/2] With-DOB run ...")
    tracker_dob = UR5Tracker(waypoints, device)
    (q_dob, ee_dob, u_dob, err_dob, samples, dhat_dob, dtrue_dob) = tracker_dob.run(
        q0, capture_samples=True, n_show=80, has_disturbance=True, use_dob=True
    )

    res_no  = dict(q=q_no,  ee=ee_no,  u=u_no,  d_hat=dhat_no,  d_true=dtrue_no)
    res_dob = dict(q=q_dob, ee=ee_dob, u=u_dob, d_hat=dhat_dob, d_true=dtrue_dob)

    plot_comparison(res_no, res_dob, waypoints)

    anim = animate_results(
        q_dob, ee_dob, u_dob, err_dob,
        waypoints, samples, traj,
        q_hist_compare=q_no,
        ee_hist_compare=ee_no,
        pos_err_no=err_no,
    )
    plt.show()
