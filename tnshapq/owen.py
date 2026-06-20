"""Direct Owen-integral Shapley and interaction indices for finite-lift QNNs.

For a game with multilinear extension F(z), z in [0,1]^P:

  * first-order Shapley value          Phi_i   = int_0^1 d_i F(t 1) dt
  * order-k interaction index (GR/STI) I(S)    = int_0^1 d_S F(t 1) dt,   d_S = prod_{i in S} d_i

Because F is multilinear, every diagonal (mixed) partial is an exact finite difference and a
polynomial in t of degree <= P - |S|.  M-point Gauss-Legendre quadrature is exact for degree
<= 2M-1, so:

  * first order   exact once  M >= ceil(P/2)        ( 2 P M  extension evaluations )
  * order k       exact once  M >= ceil((P-k+1)/2)  ( 2^k M  evaluations per coalition )

The required quadrature order DECREASES with interaction order.

Two interpolation regimes implement F(z):
  * features: per-feature input is the convex mixture (1-z_i) rho(b_i) + z_i rho(x_i),
              contracted white-box through the variational block (no 2^d enumeration);
  * gates:    gate i acts via the mixed-unitary channel z_i U_i . U_i^dag + (1-z_i) (.),
              composing to the exact Bernoulli mixture over kept gates.

Both are convex mixtures of physical states (hardware-realizable by classical randomization);
finite-shot estimates make F_hat unbiased and the quadrature a fixed linear combination of them.

Ported / extended from the TN-SHAP-Q verification core
(verify/owen_integral_shapley.py); the order-2 interaction estimators and the gate
finite-shot energy estimator are the shared helpers added for the demo.
"""
from __future__ import annotations

from itertools import combinations
from math import ceil

import numpy as np

from qiskit.quantum_info import Operator

from .qnn import kron_q, layers_operator, rho
from .gate_game import gate_unitaries


# --------------------------------------------------------------------------- #
# Gauss-Legendre quadrature on [0, 1] and the exactness thresholds
# --------------------------------------------------------------------------- #
def gl_nodes_weights(M):
    """M-point Gauss-Legendre nodes t_m in (0,1) and weights w_m with sum w_m = 1.

    Exact for polynomials in t of degree <= 2M-1."""
    nodes, wts = np.polynomial.legendre.leggauss(int(M))   # on [-1, 1]
    return 0.5 * (nodes + 1.0), 0.5 * wts                   # map to [0, 1]


def threshold_M(P, order=1):
    """Smallest M for which GL quadrature is exact for an order-`order` index on P players."""
    return int(ceil((P - order + 1) / 2))


def query_counts(P, M, order=1):
    """Per-instance evaluation counts for the routes (order = interaction order)."""
    return {
        "owen": int((2 ** order) * M),            # 2^k M per coalition of size k (k=order)
        "owen_first_order_total": int(2 * P * M),
        "exact_enumeration": int(2 ** P),
    }


# --------------------------------------------------------------------------- #
# Feature game: white-box mixed-state evaluation of the MLE
# --------------------------------------------------------------------------- #
def _feature_evaluator(qnn, n, x, baseline):
    """Return P1(ops) closure and the per-feature baseline/delta density matrices.

    The readout observable O = U^dagger Z0 U is formed ONCE (cost 8^d), so each interior
    evaluation is P1 = (1 - Re Tr(O R))/2 at cost 4^d (build the product state R = (x)rho_i
    and trace), not a fresh U R U^dagger contraction.  This keeps Owen evaluation polynomial
    in the number of nodes and feasible up to d = 12.
    """
    U = layers_operator(qnn)
    Z0 = Operator(qnn.Z0).data
    O = U.conj().T @ Z0 @ U                               # readout observable, formed once
    base = np.asarray(baseline, float)
    x = np.asarray(x, float)
    w = [rho(base[j]) for j in range(n)]                 # baseline input state
    dl = [rho(x[j]) - rho(base[j]) for j in range(n)]    # instance - baseline

    def P1(ops):
        R = kron_q(ops)
        return 0.5 * (1.0 - float(np.real(np.sum(O * R.T))))   # Tr(O R) = sum(O * R^T)

    return P1, w, dl


def precompute_marginal_probs(qnn, n, x, baseline, M):
    """Exact P(class 1) at the 2 d M diagonal evaluation points.

    Returns (t, w, on, off) with on[i,m] = F(z_i=1, z_{-i}=t_m), off[i,m] = F(z_i=0, z_{-i}=t_m).
    """
    n = int(n)
    t, wt = gl_nodes_weights(M)
    P1, w, dl = _feature_evaluator(qnn, n, x, baseline)
    on = np.empty((n, len(t)))
    off = np.empty((n, len(t)))
    for i in range(n):
        for m, tm in enumerate(t):
            others = [w[j] + tm * dl[j] for j in range(n)]
            on[i, m] = P1([(w[j] + dl[j]) if j == i else others[j] for j in range(n)])
            off[i, m] = P1([w[j] if j == i else others[j] for j in range(n)])
    return t, wt, on, off


def shapley_from_probs(on, off, w, *, shots=None, rng=None):
    """phi from precomputed on/off probabilities and GL weights (binomial shots optional)."""
    if shots is None:
        G = on - off
    else:
        rng = rng or np.random.default_rng()
        on_hat = rng.binomial(shots, np.clip(on, 0.0, 1.0)) / float(shots)
        off_hat = rng.binomial(shots, np.clip(off, 0.0, 1.0)) / float(shots)
        G = on_hat - off_hat
    return G @ np.asarray(w, float)


def owen_integral_shapley(qnn, n, x, baseline, M):
    """Exact-statevector Owen-integral feature Shapley.  Returns (phi, n_queries=2 d M)."""
    t, w, on, off = precompute_marginal_probs(qnn, n, x, baseline, M)
    return shapley_from_probs(on, off, w), int(2 * n * M)


def owen_integral_shapley_shots(qnn, n, x, baseline, M, shots, rng=None):
    """Finite-shot Owen feature Shapley (Binomial(shots, p)/shots per query).

    Returns (phi, n_queries=2 d M, total_shots)."""
    t, w, on, off = precompute_marginal_probs(qnn, n, x, baseline, M)
    phi = shapley_from_probs(on, off, w, shots=shots, rng=rng)
    n_q = int(2 * n * M)
    return phi, n_q, int(n_q * shots)


def owen_integral_interactions(qnn, n, x, baseline, M):
    """Owen-integral order-2 feature interaction indices (Grabisch-Roubens).

    I_{ij} = int_0^1 d^2 F / dz_i dz_j (t 1) dt, the diagonal integral of the second mixed
    finite difference.  Exact once M >= ceil((d-1)/2); 4 M extension evaluations per pair.
    Returns (I, n_per_pair) with I symmetric [n,n], diag 0.
    """
    t, wt = gl_nodes_weights(M)
    P1, w, dl = _feature_evaluator(qnn, n, x, baseline)
    I = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        acc = 0.0
        for tm, wm in zip(t, wt):
            base_ops = [w[k] + tm * dl[k] for k in range(n)]

            def setij(bi, bj):
                ops = list(base_ops)
                ops[i] = (w[i] + dl[i]) if bi else w[i]
                ops[j] = (w[j] + dl[j]) if bj else w[j]
                return P1(ops)

            acc += wm * (setij(1, 1) - setij(1, 0) - setij(0, 1) + setij(0, 0))
        I[i, j] = I[j, i] = acc
    return I, int(4 * M)


# --------------------------------------------------------------------------- #
# Gate game: exact mixed-unitary channel evaluation of the MLE
# --------------------------------------------------------------------------- #
def gate_extension_value(game, z, *, Us=None, nq=None, H_mat=None):
    """Exact gate MLE value F_g(z), z over the m gate players in [0,1]^m.

    rho evolves through Phi_i(rho) = p_i U_i rho U_i^dag + (1-p_i) rho, p_i = 1 for locked
    gates and p_i = z[player index] for player gates.  Returns the energy / target prob.
    """
    if Us is None:
        Us, nq = gate_unitaries(game)
    pos = {gi: j for j, gi in enumerate(game.players)}
    dim = 1 << nq
    rho_m = np.zeros((dim, dim), complex); rho_m[0, 0] = 1.0
    for gi, U in enumerate(Us):
        p = 1.0 if gi in game.locked else float(z[pos[gi]])
        if p == 1.0:
            rho_m = U @ rho_m @ U.conj().T
        elif p != 0.0:
            rho_m = p * (U @ rho_m @ U.conj().T) + (1.0 - p) * rho_m
    if game.value_fun == "energy":
        if H_mat is None:
            H_mat = Operator(game.H).data
        return float(np.real(np.trace(H_mat @ rho_m)))
    return float(np.real(rho_m[int(game.target, 2), int(game.target, 2)]))


def _gate_rho(game, z, Us, nq):
    """The mixed-unitary channel density matrix at gate-coordinate z (for shot sampling)."""
    pos = {gi: j for j, gi in enumerate(game.players)}
    dim = 1 << nq
    rho_m = np.zeros((dim, dim), complex); rho_m[0, 0] = 1.0
    for gi, U in enumerate(Us):
        p = 1.0 if gi in game.locked else float(z[pos[gi]])
        if p == 1.0:
            rho_m = U @ rho_m @ U.conj().T
        elif p != 0.0:
            rho_m = p * (U @ rho_m @ U.conj().T) + (1.0 - p) * rho_m
    return rho_m


def gate_owen_shapley(game, M):
    """Owen-integral gate Shapley via the exact channel-mixture extension.

    Returns (phi, n_evals = 2 m M).  Exact once M >= ceil(m/2)."""
    Us, nq = gate_unitaries(game)
    H_mat = Operator(game.H).data if game.value_fun == "energy" else None
    m = game.n
    t, w = gl_nodes_weights(M)
    phi = np.zeros(m)
    for j in range(m):
        acc = 0.0
        for tm, wm in zip(t, w):
            z_on = np.full(m, tm); z_on[j] = 1.0
            z_off = np.full(m, tm); z_off[j] = 0.0
            f_on = gate_extension_value(game, z_on, Us=Us, nq=nq, H_mat=H_mat)
            f_off = gate_extension_value(game, z_off, Us=Us, nq=nq, H_mat=H_mat)
            acc += wm * (f_on - f_off)
        phi[j] = acc
    return phi, int(2 * m * M)


def gate_owen_interactions(game, M):
    """Owen-integral order-2 gate interaction indices (Grabisch-Roubens).

    Exact once M >= ceil((m-1)/2); 4 M extension evaluations per pair.  Returns
    (I, pairs, n_per_pair) with I symmetric [m,m], diag 0 and pairs the list of (i,j)."""
    Us, nq = gate_unitaries(game)
    H_mat = Operator(game.H).data if game.value_fun == "energy" else None
    m = game.n
    t, wt = gl_nodes_weights(M)
    I = np.zeros((m, m))
    pairs = list(combinations(range(m), 2))
    for i, j in pairs:
        acc = 0.0
        for tm, wm in zip(t, wt):
            def setij(bi, bj):
                z = np.full(m, tm)
                z[i] = 1.0 if bi else 0.0
                z[j] = 1.0 if bj else 0.0
                return gate_extension_value(game, z, Us=Us, nq=nq, H_mat=H_mat)
            acc += wm * (setij(1, 1) - setij(1, 0) - setij(0, 1) + setij(0, 0))
        I[i, j] = I[j, i] = acc
    return I, pairs, int(4 * M)


# --------------------------------------------------------------------------- #
# Finite-shot energy estimation for the gate game (measurement shot noise)
# --------------------------------------------------------------------------- #
def sample_energy_from_rho(rho_m, H, shots, rng):
    """Unbiased N-shot estimate of Tr(H rho_m) by qubit-wise-commuting group sampling.

    Density-matrix generalization of the statevector energy estimator: for each group,
    rotate rho into the measurement basis, draw `shots` outcomes from its diagonal, and
    average Pauli parities.  Error ~ 1/sqrt(shots)."""
    nq = int(np.log2(rho_m.shape[0]))
    groups = H.group_commuting(qubit_wise=True)
    per = max(1, shots // len(groups))
    h_mat = {"X": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
             "Y": np.array([[1, -1j], [1, 1j]]) / np.sqrt(2),   # H . Sdg
             "I": np.eye(2), "Z": np.eye(2)}
    total = 0.0
    for g in groups:
        basis = ["I"] * nq
        for p in g.paulis:
            lbl = p.to_label()[::-1]
            for q in range(nq):
                if lbl[q] != "I":
                    basis[q] = lbl[q]
        R = kron_q([h_mat[basis[q]] for q in range(nq)])
        rho_rot = R @ rho_m @ R.conj().T
        probs = np.real(np.diag(rho_rot))
        probs = np.clip(probs, 0.0, None)
        probs = probs / probs.sum()
        counts = rng.multinomial(per, probs)
        for coeff, p in zip(g.coeffs, g.paulis):
            lbl = p.to_label()[::-1]
            acc = 0.0
            for outcome, c in enumerate(counts):
                if c == 0:
                    continue
                b = [(outcome >> q) & 1 for q in range(nq)]
                parity = sum(1 for q in range(nq) if lbl[q] != "I" and b[q] == 1)
                acc += c * ((-1) ** parity)
            total += float(np.real(coeff)) * acc / per
    return total


def gate_owen_shapley_shots(game, M, shots, rng=None):
    """Finite-shot Owen gate Shapley: each channel-mixture energy is estimated from `shots`.

    Returns (phi, n_evals = 2 m M, total_shots)."""
    rng = rng or np.random.default_rng()
    Us, nq = gate_unitaries(game)
    m = game.n
    t, w = gl_nodes_weights(M)
    phi = np.zeros(m)
    for j in range(m):
        acc = 0.0
        for tm, wm in zip(t, w):
            z_on = np.full(m, tm); z_on[j] = 1.0
            z_off = np.full(m, tm); z_off[j] = 0.0
            f_on = sample_energy_from_rho(_gate_rho(game, z_on, Us, nq), game.H, shots, rng)
            f_off = sample_energy_from_rho(_gate_rho(game, z_off, Us, nq), game.H, shots, rng)
            acc += wm * (f_on - f_off)
        phi[j] = acc
    n_q = int(2 * m * M)
    return phi, n_q, int(n_q * shots)
