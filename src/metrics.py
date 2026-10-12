# src/metrics.py
import math

def _safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0

def js_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence between two distributions.
    Lower = more similar (better emotional fidelity).
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    P = [max(p.get(k, 0.0), 0.0) + eps for k in keys]
    Q = [max(q.get(k, 0.0), 0.0) + eps for k in keys]

    # normalize (in case they don't sum perfectly to 1)
    sp, sq = sum(P), sum(Q)
    P = [x / sp for x in P]
    Q = [x / sq for x in Q]

    M = [(P[i] + Q[i]) / 2 for i in range(len(keys))]

    def kl(a, b):
        return sum(a[i] * (_safe_log(a[i]) - _safe_log(b[i])) for i in range(len(keys)))

    return (kl(P, M) + kl(Q, M)) / 2

#issuecommit*/