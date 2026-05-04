"""
disturbance_observer.py — 1차 Q-필터 기반 Disturbance Observer (DOB)

이산 방정식:
    d_hat += dt · (−α · d_hat + α · residual)
    residual  = u_app − (I · ddq_meas + B · dq_prev)
    ddq_meas ≈ (dq_next − dq_prev) / dt

연속 전달함수:  Q(s) = ωc / (s + ωc)   (1차 저역통과)
이산 근사:      d_hat_k+1 = (1 − α·dt)·d_hat_k + α·dt·residual_k

논문 실험 통합 파라미터: α = ωc = 40 rad/s  (analysis.py 기준)
"""

import numpy as np

# ── 통합 observer bandwidth ────────────────────────────────────────────────────
# 안정 조건: α < 1/dt = 50  →  (1 − dt·α) > 0
# α = 40: 시상수 τ = 25 ms (~1.25 스텝), 안정 마진 충분 (1 − 40·0.02 = 0.20)
ALPHA_DOB: float = 40.0


class DisturbanceObserver:
    """Numpy 기반 DOB — SQP 컨트롤러 및 analysis.py SQP 러너에서 사용.

    Parameters
    ----------
    I_eff  : array-like (n,) — 유효 관절 관성 [kg·m²]
    B_damp : array-like (n,) — 점성 감쇠 계수 [Nm·s/rad]
    dt     : float            — 시뮬레이션 스텝 [s]
    alpha  : float            — observer bandwidth [rad/s]  (기본값 ALPHA_DOB)
    """

    def __init__(self, I_eff, B_damp, dt: float, alpha: float = ALPHA_DOB):
        self._I     = np.asarray(I_eff,  dtype=np.float64)
        self._B     = np.asarray(B_damp, dtype=np.float64)
        self._dt    = dt
        self._alpha = alpha
        self.d_hat  = np.zeros(len(self._I))

    def reset(self) -> None:
        """외란 추정값 초기화."""
        self.d_hat[:] = 0.0

    def update(self,
               u_app: np.ndarray,
               dq_prev: np.ndarray,
               dq_curr: np.ndarray) -> np.ndarray:
        """DOB 추정값 한 스텝 업데이트.

        Parameters
        ----------
        u_app   : 인가 토크 (외란 보상 후, 외란 미포함) [Nm], shape (n,)
        dq_prev : 이전 스텝 관절 속도 [rad/s], shape (n,)
        dq_curr : 현재 스텝 관절 속도 [rad/s], shape (n,)

        Returns
        -------
        d_hat : 업데이트된 외란 추정값 (view, copy 불필요)
        """
        ddq_meas   = (dq_curr - dq_prev) / self._dt
        residual   = u_app - (self._I * ddq_meas + self._B * dq_prev)
        self.d_hat = (self.d_hat
                      + self._dt * (-self._alpha * self.d_hat
                                    + self._alpha * residual))
        return self.d_hat


class TorchDisturbanceObserver:
    """Torch 기반 DOB — MPPI 컨트롤러 GPU 연산용 (CPU↔GPU 변환 없음).

    Parameters
    ----------
    I_t : torch.Tensor (n,) — 유효 관절 관성, 올바른 device에 이미 위치해야 함
    B_t : torch.Tensor (n,) — 점성 감쇠 계수, 동일 device
    dt  : float
    alpha : float  (기본값 ALPHA_DOB)
    """

    def __init__(self, I_t, B_t, dt: float, alpha: float = ALPHA_DOB):
        self._I     = I_t   # 기존 torch.Tensor 참조 (device/dtype 유지)
        self._B     = B_t
        self._dt    = dt
        self._alpha = alpha
        self.d_hat  = I_t.new_zeros(I_t.shape)

    def reset(self) -> None:
        """외란 추정값 초기화."""
        self.d_hat = self._I.new_zeros(self._I.shape)

    def update(self, u_app, dq_prev, dq_curr):
        """DOB 추정값 한 스텝 업데이트.

        Parameters
        ----------
        u_app, dq_prev, dq_curr : torch.Tensor (n,)  — 동일 device/dtype

        Returns
        -------
        d_hat : torch.Tensor (n,)
        """
        ddq      = (dq_curr - dq_prev) / self._dt
        residual = u_app - (self._I * ddq + self._B * dq_prev)
        self.d_hat = (self.d_hat
                      + self._dt * (-self._alpha * self.d_hat
                                    + self._alpha * residual))
        return self.d_hat
