"""Gate-coalition game (players = circuit gates) and the qshaptools QAOA example.

A coalition S keeps the locked gates plus the player gates in S (gate REMOVAL masking,
in original order) and reads v_g(S) = <psi_S| H |psi_S> (energy) or P(target). This is the
SVQX / qshaptools setting.  Ported verbatim (semantics preserved) from the TN-SHAP-Q
verification core (svqx_core.py :: CircuitGame).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


class CircuitGame:
    """Gate-coalition value function over a decomposed, parameter-bound circuit."""

    def __init__(self, qc: QuantumCircuit, value_fun: str = "energy", H=None,
                 target: Optional[str] = None, locked=None):
        if qc.num_parameters > 0:
            raise ValueError("Bind all circuit parameters before constructing CircuitGame.")
        self.qc = qc
        self.value_fun = value_fun
        self.H = H
        self.target = target
        self.locked = set(locked or [])
        self.players = [i for i in range(len(qc.data)) if i not in self.locked]
        self.n = len(self.players)
        if value_fun == "energy" and H is None:
            raise ValueError("value_fun='energy' requires H (SparsePauliOp).")
        if value_fun == "target_prob" and target is None:
            raise ValueError("value_fun='target_prob' requires target bitstring.")
        self._groups = H.group_commuting(qubit_wise=True) if (H is not None) else None
        self.n_eval = 0
        self.shots_used = 0

    # -- masking -------------------------------------------------------------
    def kept_indices(self, z):
        z = np.asarray(z).astype(int).ravel()
        assert z.size == self.n, f"z has {z.size} entries, expected {self.n}"
        keep = set(self.locked)
        for i, bit in enumerate(z):
            if bit:
                keep.add(self.players[i])
        return keep

    def masked_circuit(self, z) -> QuantumCircuit:
        keep = self.kept_indices(z)
        new = self.qc.copy_empty_like()
        for i, ci in enumerate(self.qc.data):
            if i in keep:
                new.append(ci.operation, ci.qubits, ci.clbits)
        return new

    # -- value functions -----------------------------------------------------
    def value(self, z, shots: Optional[int] = None, rng=None) -> float:
        qc = self.masked_circuit(z)
        self.n_eval += 1
        if self.value_fun == "energy":
            return self._value_energy(qc, shots, rng)
        return self._value_target_prob(qc, shots, rng)

    def _value_energy(self, qc, shots, rng) -> float:
        sv = Statevector(qc)
        if shots is None:
            return float(np.real(sv.expectation_value(self.H)))
        rng = rng or np.random.default_rng()
        groups = self._groups
        per = max(1, shots // len(groups))
        total = 0.0
        nq = qc.num_qubits
        for g in groups:
            basis = ["I"] * nq
            for p in g.paulis:
                lbl = p.to_label()[::-1]
                for q in range(nq):
                    if lbl[q] != "I":
                        basis[q] = lbl[q]
            mqc = qc.copy()
            for q in range(nq):
                if basis[q] == "X":
                    mqc.h(q)
                elif basis[q] == "Y":
                    mqc.sdg(q); mqc.h(q)
            counts = Statevector(mqc).sample_counts(per, qargs=range(nq))
            self.shots_used += per
            for coeff, p in zip(g.coeffs, g.paulis):
                lbl = p.to_label()[::-1]
                acc = 0.0
                for bitstr, c in counts.items():
                    b = bitstr[::-1]
                    parity = sum(1 for q in range(nq) if lbl[q] != "I" and b[q] == "1")
                    acc += c * ((-1) ** parity)
                total += float(np.real(coeff)) * acc / per
        return total

    def _value_target_prob(self, qc, shots, rng) -> float:
        sv = Statevector(qc)
        if shots is None:
            return float(sv.probabilities_dict().get(self.target, 0.0))
        rng = rng or np.random.default_rng()
        counts = sv.sample_counts(shots, qargs=range(qc.num_qubits))
        self.shots_used += shots
        return counts.get(self.target, 0) / shots


# --------------------------------------------------------------------------- #
# The qshaptools README example: QAOA cost H = Z0Z1 + 2 Z0 - 3 Z2
# --------------------------------------------------------------------------- #
def build_qaoa_example(gamma: float = 0.3):
    """Reproduce the qshaptools README circuit (QAOAAnsatz, reps=1, decompose x3).

    Returns (qc, H, game) with m = 13 gate players on 3 qubits (energy value function).
    H is the little-endian SparsePauliOp Z0Z1 + 2 Z0 - 3 Z2 used throughout the repo.
    """
    H = SparsePauliOp.from_list([("ZZI", 1.0), ("ZII", 2.0), ("ZIZ", -3.0)])
    qc = QAOAAnsatz(cost_operator=H, reps=1)
    for _ in range(3):
        qc = qc.decompose()
    qc = qc.assign_parameters([gamma] * len(qc.parameters))
    game = CircuitGame(qc, value_fun="energy", H=H)
    return qc, H, game


def gate_unitaries(game: CircuitGame):
    """Full-space (2^nq x 2^nq) unitary for each instruction in game.qc (qiskit order)."""
    nq = game.qc.num_qubits
    Us = []
    for ci in game.qc.data:
        qc1 = QuantumCircuit(nq)
        qc1.append(ci.operation, ci.qubits)
        Us.append(Operator(qc1).data)
    return Us, nq
