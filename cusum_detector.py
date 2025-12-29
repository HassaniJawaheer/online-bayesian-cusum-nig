from dataclasses import dataclass
from typing import Optional
from math import isfinite
import numpy as np

from nig_model import NormalInverseGammaState


@dataclass
class StepResult:
    s: float
    S: float
    log: tuple[float, float]
    alert: bool


@dataclass
class DetectionResult:
    alert_index: int | None
    scores: list[float]
    increments: list[float]
    alerts: list[bool]
    logs: Optional[list[tuple[float, float]]]
    stopped_early: bool


class BayesianCUSUMDetector:
    def __init__(
        self,
        model0: NormalInverseGammaState,
        model1: NormalInverseGammaState,
        threshold_h: float,
        store_logs: bool = True,
        stop_on_alert: bool = True,
    ):
        assert threshold_h > 0.0, "threshold_h must be > 0"
        self.model0 = model0
        self.model1 = model1
        self.threshold_h = float(threshold_h)
        self.S = 0.0
        self.store_logs = store_logs
        self.stop_on_alert = stop_on_alert

        self.reset()

    def _sanity_S(self):
        assert self.S >= 0.0 and isfinite(self.S)

    def _sanity_log(self, logp0: float, logp1: float):
        assert isfinite(logp0)
        assert isfinite(logp1)

    def reset(self) -> None:
        self.S = 0.0
        self.model1.reset()

    def step(self, x: float) -> StepResult:
        x = float(x)

        logp0 = float(self.model0.log_predictive(x))
        logp1 = float(self.model1.log_predictive(x))
        self._sanity_log(logp0, logp1)

        s = logp1 - logp0
        assert isfinite(s)

        self.S = max(0.0, self.S + s)
        self._sanity_S()

        # reset du modèle post si on retombe à zéro
        if self.S <= 0.0:
            self.model1.reset()
        else:
            self.model1.update(x)

        alert = (self.S >= self.threshold_h)
        return StepResult(s=s, S=self.S, log=(logp0, logp1), alert=alert)

    def run(self, samples: np.ndarray) -> DetectionResult:
        self.reset()

        logs_init = [] if self.store_logs else None
        detection_result = DetectionResult(
            alert_index=None,
            scores=[],
            increments=[],
            alerts=[],
            logs=logs_init,
            stopped_early=False,
        )

        for index in range(len(samples)):
            step_result = self.step(samples[index])

            detection_result.scores.append(step_result.S)
            detection_result.increments.append(step_result.s)
            detection_result.alerts.append(step_result.alert)

            if self.store_logs:
                detection_result.logs.append(step_result.log)

            if step_result.alert and self.stop_on_alert:
                detection_result.alert_index = index
                detection_result.stopped_early = True
                return detection_result

        return detection_result





