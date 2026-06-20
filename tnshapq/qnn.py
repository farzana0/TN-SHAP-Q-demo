"""Single-R_Y QNN, the feature-coalition game, and the white-box mixed-state contraction.

The QNN encodes feature x_i by ONE R_Y(x_i) rotation, applies an input-independent
variational block U (R_Y(w) layers + CX rings), and reads P(class 1) = (1 - <Z0>)/2.
This is the finite-lift encoding: rho_i(x_i) = 1/2 (I + cos x_i Z + sin x_i X), lift
phi(x_i) = [1, cos x_i, sin x_i].  The circuit output is therefore multilinear in the
coalition coordinates z, so it *is* the multilinear extension (MLE) of the feature game
(Proposition: Exact realization, features).

Ported verbatim (semantics preserved) from the TN-SHAP-Q verification core
(experiments/e3_tn_qlime_qnn.py :: QNN/FeatureGame/apply_mask, verify/common.py ::
rho/kron_q/layers_operator).  No tensor-network surrogate, no torch dependency.
"""
from __future__ import annotations

from functools import reduce
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

# single-qubit Paulis (qubit 0 = rightmost in qiskit kron order)
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


class QNN:
    """RY(x_i) encoding, L layers of [RY(w) on every qubit + CX ring], Z0 readout."""

    def __init__(self, n_features: int, n_layers: int = 2, seed: int = 0):
        self.nq = n_features
        self.L = n_layers
        self.x = ParameterVector("x", n_features)
        self.w = ParameterVector("w", n_layers * n_features)
        self.n_weights = n_layers * n_features
        self.Z0 = SparsePauliOp.from_list([("Z" + "I" * (n_features - 1), 1.0)])
        self._template = self._build()
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0, 2 * np.pi, size=self.n_weights)

    def _build(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.nq)
        for q in range(self.nq):
            qc.ry(self.x[q], q)
        k = 0
        for _ in range(self.L):
            for q in range(self.nq):
                qc.ry(self.w[k], q)
                k += 1
            for q in range(self.nq):
                qc.cx(q, (q + 1) % self.nq)
        return qc

    def _bind(self, x: np.ndarray, weights: np.ndarray) -> QuantumCircuit:
        amap = {self.x[i]: float(x[i]) for i in range(self.nq)}
        amap.update({self.w[i]: float(weights[i]) for i in range(self.n_weights)})
        return self._template.assign_parameters(amap)

    def prob1(self, x, weights=None, *, shots: Optional[int] = None, rng=None) -> float:
        """P(class 1) = (1 - <Z0>)/2; exact statevector when shots is None."""
        weights = self.weights if weights is None else weights
        qc = self._bind(np.asarray(x, float), weights)
        sv = Statevector(qc)
        if shots is None:
            z0 = float(np.real(sv.expectation_value(self.Z0)))
        else:
            rng = rng or np.random.default_rng()
            counts = sv.sample_counts(shots, qargs=range(self.nq))
            tot = sum(counts.values())
            z0 = sum((1 if b[0] == "0" else -1) * c for b, c in counts.items()) / tot  # b[0] = highest qubit = self.Z0 support
        return float((1 - z0) / 2)

    def predict(self, X, *, shots: Optional[int] = None) -> np.ndarray:
        return np.array([self.prob1(x, shots=shots) for x in X], dtype=float)

    def fit(self, X, y, *, epochs: int = 80, lr: float = 0.15, seed: int = 0,
            batch: Optional[int] = None) -> "QNN":
        """Parameter-shift gradient + Adam on BCE loss (tiny, deterministic for fixed seed)."""
        rng = np.random.default_rng(seed)
        m1, v1, t = np.zeros(self.n_weights), np.zeros(self.n_weights), 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n = len(X)
        for _ep in range(epochs):
            idx = rng.permutation(n)[: (batch or n)]
            grad = np.zeros(self.n_weights)
            for i in idx:
                xi, yi = X[i], y[i]
                p = np.clip(self.prob1(xi), 1e-6, 1 - 1e-6)
                dL_dp = (p - yi) / (p * (1 - p))
                for j in range(self.n_weights):
                    wp = self.weights.copy(); wp[j] += np.pi / 2
                    wm = self.weights.copy(); wm[j] -= np.pi / 2
                    dp_dw = (self.prob1(xi, wp) - self.prob1(xi, wm)) / 2.0
                    grad[j] += dL_dp * dp_dw
            grad /= len(idx)
            t += 1
            m1 = b1 * m1 + (1 - b1) * grad
            v1 = b2 * v1 + (1 - b2) * grad ** 2
            mhat = m1 / (1 - b1 ** t)
            vhat = v1 / (1 - b2 ** t)
            self.weights -= lr * mhat / (np.sqrt(vhat) + eps)
        return self


# --------------------------------------------------------------------------- #
# Feature-coalition game
# --------------------------------------------------------------------------- #
def apply_mask(x, baseline, z) -> np.ndarray:
    """x_S: keep feature i (x_i) where z_i=1, else replace by baseline_i."""
    z = np.asarray(z).astype(int).ravel()
    x = np.asarray(x, float).ravel()
    baseline = np.asarray(baseline, float).ravel()
    return np.where(z == 1, x, baseline)


class FeatureGame:
    """nu(S) = P(target | x_S), x_S keeps S at x and replaces the rest by baseline.

    target='class1' makes nu(S) = P(class 1) directly (the natural multilinear object).
    """

    def __init__(self, qnn: QNN, x, baseline, *, target: str = "class1"):
        self.qnn = qnn
        self.x = np.asarray(x, float)
        self.baseline = np.asarray(baseline, float)
        self.n = int(self.x.size)
        self.n_eval = 0
        self.shots_used = 0
        self.p_full = float(self.qnn.prob1(self.x, shots=None))
        self.pred_label = 1 if self.p_full >= 0.5 else 0
        self.target_label = {"pred": self.pred_label, "class1": 1, "class0": 0}[target]

    @property
    def reliability(self) -> float:
        return abs(self.p_full - 0.5)

    def value(self, z, *, shots: Optional[int] = None, rng=None) -> float:
        self.n_eval += 1
        if shots is not None:
            self.shots_used += int(shots)
        p1 = float(self.qnn.prob1(apply_mask(self.x, self.baseline, z), shots=shots, rng=rng))
        return p1 if self.target_label == 1 else 1.0 - p1


# --------------------------------------------------------------------------- #
# White-box mixed-state contraction of the feature MLE (no 2^d enumeration)
# --------------------------------------------------------------------------- #
def rho(theta) -> np.ndarray:
    """Input density operator for RY(theta)|0>:  1/2 (I + cos(theta) Z + sin(theta) X)."""
    return 0.5 * (I2 + np.cos(theta) * Z + np.sin(theta) * X)


def kron_q(mats) -> np.ndarray:
    """Full-space operator from single-qubit factors, qiskit order (qubit 0 = LSB)."""
    return reduce(np.kron, [mats[q] for q in range(len(mats) - 1, -1, -1)])


def layers_operator(qnn: QNN) -> np.ndarray:
    """Unitary of the variational block ONLY (RY(w) layers + CX rings), no encoding."""
    n = qnn.nq
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(qnn.L):
        for q in range(n):
            qc.ry(float(qnn.weights[k]), q); k += 1
        for q in range(n):
            qc.cx(q, (q + 1) % n)
    return Operator(qc).data


def readout_observable(qnn: QNN) -> np.ndarray:
    """O = U^dagger Z0 U, so <Z0> = Tr(O * prod_i rho_i) and P1 = (1 - <Z0>)/2."""
    U = layers_operator(qnn)
    Z0 = Operator(qnn.Z0).data
    return U.conj().T @ Z0 @ U
