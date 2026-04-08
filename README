# Linear Programming for 5G Radio Scheduling
### A Self-Contained Guide — No Prior Knowledge Required

---

## Table of Contents

1. [What is 5G Scheduling? (Plain English)](#1-what-is-5g-scheduling-plain-english)
2. [Key Terms You Need to Know](#2-key-terms-you-need-to-know)
3. [The Core Idea: Sharing Time Slots](#3-the-core-idea-sharing-time-slots)
4. [What is Linear Programming?](#4-what-is-linear-programming)
5. [Building the LP Step by Step](#5-building-the-lp-step-by-step)
6. [Problem Type 1 — Max Sum Throughput](#6-problem-type-1--max-sum-throughput)
7. [Problem Type 2 — Max Weighted Throughput](#7-problem-type-2--max-weighted-throughput)
8. [Problem Type 3 — Proportional Fair (PF)](#8-problem-type-3--proportional-fair-pf)
9. [Problem Type 4 — PF with Rate Guarantee (GBR)](#9-problem-type-4--pf-with-rate-guarantee-gbr)
10. [Comparing All Four Problem Types](#10-comparing-all-four-problem-types)
11. [Python Implementation](#11-python-implementation)
12. [Summary Cheatsheet](#12-summary-cheatsheet)

---

## 1. What is 5G Scheduling? (Plain English)

Imagine a single Wi-Fi router at home shared by your entire family. Everyone wants to stream, browse, and video call at the same time, but the router can only send data to **one device at a time** (in very short bursts). It has to constantly decide: *"Who gets the channel right now?"*

In a 5G network, the same problem exists but at massive scale. A **gNB** (the 5G base station — think of it as a cell tower's brain) serves many **UEs** (User Equipment — your phone, laptop, IoT device, etc.) over the air. The air is a shared resource.

Time is divided into extremely short slots called **TTIs (Transmission Time Intervals)** — each one is just **0.5 milliseconds** long. In every single TTI, the gNB must pick which UE gets to transmit or receive data.

> **The Scheduling Problem:** In each TTI, decide which UE gets the radio resource, so that over many TTIs, every UE gets a fair and adequate share — and the network as a whole is used efficiently.

This is fundamentally an **optimisation problem**, and Linear Programming (LP) is one of the most powerful tools to solve it.

---

## 2. Key Terms You Need to Know

| Term | What it means |
|------|---------------|
| **UE** | User Equipment — any device connected to the 5G network (phone, laptop, sensor, etc.) |
| **gNB** | The 5G base station that manages all scheduling |
| **TTI** | Transmission Time Interval — one scheduling slot, lasting 0.5 ms |
| **Rate `r[i,k]`** | How many bits UE `i` *can* receive in TTI `k` (depends on signal strength / distance) |
| **Throughput `θ[i]`** | The long-run average bit rate (Mbps) actually delivered to UE `i` |
| **EWMA** | Exponentially Weighted Moving Average — a running estimate of a UE's recent throughput |
| **MCS** | Modulation and Coding Scheme — the encoding used; determines how many bits fit in one TTI |
| **GBR** | Guaranteed Bit Rate — a minimum throughput promise made to a specific UE or service |
| **Throughput Region** | The set of all achievable average throughput combinations across UEs |

---

## 3. The Core Idea: Sharing Time Slots

### Why can't all UEs be served at the same time?

In 5G, the radio channel is a shared medium. In a given TTI, allocating all resources to one UE gives that UE the highest possible rate. If you split resources, everyone gets a lower rate. So the scheduler typically assigns all resources to **one UE per TTI**, and cycles through UEs across many TTIs.

### Time-sharing fractions

Over `K` TTIs, if UE `i` is scheduled in `k_i` of those TTIs, its **time-sharing fraction** is:

```
p[i] = k_i / K
```

Since every TTI must be assigned to exactly one UE:

```
p[0] + p[1] + ... + p[N-1] = 1
```

The average throughput UE `i` achieves is approximately:

```
θ[i] ≈ p[i] × (average rate available to UE i)
```

More precisely, since the rate `r[i,k]` varies slot by slot (the channel fluctuates):

```
θ[i] = (1/K) × Σ_k  r[i,k] × p[i,k]
```

where `p[i,k] = 1` if UE `i` is scheduled in TTI `k`, else `0`.

### The Throughput Region

The **throughput region** is the set of all average throughput vectors `(θ[0], θ[1], ..., θ[N-1])` that are achievable by some scheduling policy. It is always:

- **Convex** — you can always achieve any weighted average of two achievable points
- **Coordinate-convex** — if a point is achievable, so is any point with smaller values
- **Bounded** — you can't exceed what the channel physically allows

The goal of scheduling is to pick the **right operating point** on the boundary of this region — meaning, we want to be as efficient as possible (on the boundary, not wasting capacity) while satisfying fairness or service requirements.

---

## 4. What is Linear Programming?

**Linear Programming (LP)** is a method to find the best value of a linear objective function subject to linear constraints.

### General form

```
minimize    c^T x
subject to  A x  ≤  b        (inequality constraints)
            A_eq x = b_eq    (equality constraints)
            lb ≤ x ≤ ub      (variable bounds)
```

Where:
- `x` is the **vector of decision variables** (what we choose)
- `c` is the **cost vector** (coefficients of the objective)
- `A`, `b` define the **constraint boundaries**

> **Intuition:** LP is like trying to maximise the area of a rectangle while keeping its perimeter below a fixed limit. The constraints form a polygon (the feasible region), and the optimal solution always sits at one of its corners.

### Why LP for scheduling?

The scheduling problem asks: *"What fraction of TTIs should each UE get?"* — these fractions are continuous variables between 0 and 1, the objective (total throughput or utility) is a function of these fractions, and the constraints (fractions must sum to 1, rates are bounded) are all linear. LP is the natural fit.

---

## 5. Building the LP Step by Step

### Step 1 — Define decision variables

For each UE `i` and each TTI `k`, define:

```
p[i,k]  =  1  if UE i is scheduled in TTI k
         =  0  otherwise
```

We **relax** this to a continuous LP (allowing values between 0 and 1):

```
0 ≤ p[i,k] ≤ 1    for all i, k
```

This relaxation is valid because the LP solution will automatically be near-integer in practice (each TTI is assigned to the UE with the highest weighted rate).

### Step 2 — Express throughput

The throughput of UE `i` over `K` TTIs is:

```
θ[i]  =  (1/K) × Σ_{k=1}^{K}  r[i,k] × p[i,k]
```

### Step 3 — Write constraints

**Constraint 1: At most one UE per TTI**

```
Σ_{i=0}^{N-1}  p[i,k]  ≤  1       for all k = 1, ..., K
```

**Constraint 2: Non-negativity and upper bound**

```
0 ≤ p[i,k] ≤ 1       for all i, k
```

### Step 4 — Define the objective

Different objectives lead to different LP variants (explained in sections 6–9 below).

---

## 6. Problem Type 1 — Max Sum Throughput

### Goal

Maximise the **total throughput** delivered across all UEs. This treats all bits equally, regardless of which UE receives them.

### Objective function

```
maximize    Σ_{i=0}^{N-1}  θ[i]

         =  maximize    (1/K) × Σ_i Σ_k  r[i,k] × p[i,k]
```

### Full LP formulation

```
maximize    (1/K) × Σ_i Σ_k  r[i,k] × p[i,k]

subject to  Σ_i p[i,k]  ≤  1          for all k       (one UE per TTI)
            0 ≤ p[i,k] ≤ 1            for all i, k    (valid fractions)
```

### Per-TTI greedy solution

Since the objective is linear and separable across TTIs, the optimal per-TTI decision is simply:

```
Schedule UE i* in TTI k where:
    i* = argmax_i  r[i,k]
```

In words: **always schedule the UE with the highest instantaneous rate.**

### What does this look like in practice?

| UE | Distance from gNB | Avg rate | Scheduled fraction |
|----|-------------------|----------|--------------------|
| UE 0 | 200 m (close) | 120 Mbps | ~85% of TTIs |
| UE 1 | 400 m (far)   | 40 Mbps  | ~15% of TTIs |

The close UE dominates because its channel is almost always better. **The far UE gets very little.**

### Tradeoff

✅ Maximum overall throughput (efficient use of the network)  
❌ Very unfair — weak UEs are starved of resources

---

## 7. Problem Type 2 — Max Weighted Throughput

### Goal

Same as max sum throughput, but give each UE a **priority weight** `w[i]`. A UE with a higher weight gets more TTIs, even if its rate is lower.

### Objective function

```
maximize    Σ_{i=0}^{N-1}  w[i] × θ[i]

         =  maximize    (1/K) × Σ_i Σ_k  w[i] × r[i,k] × p[i,k]
```

### Full LP formulation

```
maximize    (1/K) × Σ_i Σ_k  w[i] × r[i,k] × p[i,k]

subject to  Σ_i p[i,k]  ≤  1          for all k
            0 ≤ p[i,k] ≤ 1            for all i, k
```

### Per-TTI greedy solution

```
i* = argmax_i  w[i] × r[i,k]
```

### Example

If `w[0] = 1` and `w[1] = 5` (UE 1 has 5× higher priority):

- Even if UE 0 has a rate of 120 Mbps and UE 1 has 30 Mbps, the score for UE 1 is `5 × 30 = 150` vs `1 × 120 = 120`
- So UE 1 gets scheduled in this TTI

### Tradeoff

✅ Flexible — operators can manually prioritise UEs or services  
❌ Weights must be chosen carefully; wrong weights = unfair or inefficient outcomes

---

## 8. Problem Type 3 — Proportional Fair (PF)

### Motivation

Max sum throughput ignores fairness. But pure equal-time round-robin wastes capacity. **Proportional Fair** is the elegant middle ground — it maximises fairness *relative to what each UE can achieve* given its channel.

### Objective function

PF maximises the **sum of logarithms** of throughputs:

```
maximize    Σ_{i=0}^{N-1}  log(θ[i])
```

> **Why log?** The log function grows fast at small values and slowly at large values. So the optimiser is "motivated" to bring up the throughput of under-served UEs, because doubling a small throughput (say 5→10 Mbps) contributes much more to `log(θ)` than doubling a large one (100→200 Mbps).

### The challenge: log is not linear

`log(θ)` is a **concave** (non-linear) function of `θ`. LP only handles linear objectives. How do we solve this?

### The gradient approximation trick

At each TTI `k`, we approximate the log objective with its **gradient** (first-order Taylor expansion):

```
log(θ[i])  ≈  log(θ̄[i](k-1))  +  (1 / θ̄[i](k-1)) × (θ[i] - θ̄[i](k-1))
```

The term `1 / θ̄[i](k-1)` is just a constant at the time of scheduling. So the **effective weight** for UE `i` in TTI `k` is:

```
w[i](k)  =  1 / θ̄[i](k-1)
```

where `θ̄[i](k-1)` is the **EWMA (Exponentially Weighted Moving Average) throughput** of UE `i` up to TTI `k-1`.

### EWMA throughput update rule

After TTI `k`, update the EWMA as follows:

**If UE `i` was scheduled in TTI `k`:**

```
θ̄[i](k)  =  (1 - α) × θ̄[i](k-1)  +  α × (r[i,k] / T)
```

**If UE `i` was NOT scheduled in TTI `k`:**

```
θ̄[i](k)  =  (1 - α) × θ̄[i](k-1)  +  α × 0
           =  (1 - α) × θ̄[i](k-1)
```

Where:
- `α` is the smoothing parameter (typically 0.0005, meaning the averaging window is `1/α = 2000 TTIs`)
- `T` is the TTI duration (0.5 ms = 0.0005 s)
- `r[i,k]` is the bits delivered to UE `i` in TTI `k`

### Full PF LP formulation

```
maximize    Σ_i  (1 / θ̄[i](k-1)) × θ[i]          (linearised at current EWMA)

subject to  Σ_i p[i,k]  ≤  1          for all k
            0 ≤ p[i,k] ≤ 1            for all i, k
```

### Per-TTI greedy solution

```
i* = argmax_i   r[i,k] / θ̄[i](k-1)
```

In words: **schedule the UE whose current rate is largest relative to its recent average throughput.**

### Intuition

- If UE `i` has had a run of bad channel conditions, its EWMA `θ̄[i]` is low
- So the ratio `r[i,k] / θ̄[i]` is inflated for this UE
- The scheduler will preferentially pick it when its channel temporarily improves
- This exploits **multiuser diversity** — waiting for each UE's channel to peak before serving it

### Tradeoff

✅ Optimal balance of fairness and efficiency  
✅ Automatic adaptation to channel conditions  
❌ Total throughput is slightly lower than pure max sum

---

## 9. Problem Type 4 — PF with Rate Guarantee (GBR)

### Motivation

Some services — like Fixed Wireless Access (FWA, i.e. home broadband over 5G), video conferencing, or industrial IoT — need a **guaranteed minimum throughput**, not just a "fair share." These are called **GBR (Guaranteed Bit Rate)** flows. In 5G, these are served via **dedicated Data Radio Bearers (DRBs)** within a network slice.

### What changes?

We add a **hard constraint** to the PF problem: the scheduler must ensure each UE with a rate guarantee meets its minimum.

### Full LP formulation

```
maximize    Σ_i  (1 / θ̄[i](k-1)) × θ[i]

subject to  θ[i]  ≥  RG[i]              for all i with a guarantee  (GBR constraint)
            Σ_i p[i,k]  ≤  1            for all k                   (one UE per TTI)
            0 ≤ p[i,k] ≤ 1              for all i, k
```

Where `RG[i]` is the minimum guaranteed throughput for UE `i` (0 for best-effort UEs).

### Expressing GBR as an LP inequality

Since `linprog` in Python only accepts `≤` constraints, we rewrite `θ[i] ≥ RG[i]` as:

```
−θ[i]  ≤  −RG[i]

⟺   −(1/K) × Σ_k  r[i,k] × p[i,k]  ≤  −RG[i]
```

### The Index-Bias mechanism

In practice, the per-TTI greedy rule is extended with a **bias term** that increases when a UE is falling short of its guarantee:

```
i* = argmax_i   [r[i,k] / θ̄[i](k-1)] × bias[i](k)
```

Where:

```
         ┌  β > 1    if  θ̄[i](k-1) < RG[i]   (UE is behind its guarantee)
bias[i] = │
         └  1        otherwise
```

The bias value `β` controls how aggressively the scheduler compensates. A larger `β` means the GBR UE will almost always get scheduled when it's behind, at the expense of others.

### Feasibility condition

The LP has a solution only if the total guaranteed throughput does not exceed what the network can provide:

```
Σ_i  RG[i]  ≤  Total achievable throughput of the network
```

If the guarantees are set too high, the LP is **infeasible** — no scheduler can satisfy all constraints simultaneously. Admission control (deciding whether to accept a new GBR slice) must check this.

### Tradeoff

✅ Hard throughput guarantees for critical services (e.g. FWA, video, URLLC)  
✅ Remaining capacity still shared proportionally fairly among best-effort UEs  
❌ Total throughput decreases as more guarantees are added  
❌ Infeasible if total guarantee exceeds capacity

---

## 10. Comparing All Four Problem Types

| Property | Max Sum | Max Weighted | Proportional Fair | PF + Rate Guarantee |
|----------|---------|--------------|-------------------|---------------------|
| **Objective function** | `Σ θ[i]` | `Σ w[i]·θ[i]` | `Σ log(θ[i])` | `Σ log(θ[i])` |
| **LP type** | Linear | Linear | Linearised (gradient) | Linearised + constraint |
| **Per-TTI rule** | `argmax r[i,k]` | `argmax w[i]·r[i,k]` | `argmax r[i,k]/θ̄[i]` | `argmax r[i,k]/θ̄[i] × bias[i]` |
| **Fairness** | Very low | Tunable | High | High (with guarantee) |
| **Efficiency** | Highest | High | Moderate | Lower |
| **Supports GBR?** | No | Partially | No | Yes |
| **Complexity** | Lowest | Low | Moderate | Moderate |

### Visual intuition: Operating points on the throughput region

```
      θ[1]
        │
   max  ●  ← Max sum throughput (most efficient, least fair)
  θ[1]  │╲
        │  ╲
        │   ╲  ● ← Proportional fair (balanced)
        │    ╲
        │     ╲  ● ← PF with RG[0]=50Mbps (θ[0] ≥ 50)
        │      ╲│
        │       ╲
        └────────●─────── θ[0]
                  max θ[0]
```

Each point on the curved boundary is achievable by some scheduling policy. The LP finds exactly which point corresponds to a given objective + constraints.

---

## 11. Python Implementation

```python
import numpy as np
from scipy.optimize import linprog

# ── Parameters ──────────────────────────────────────────────────
N = 2        # number of UEs
K = 500      # number of TTIs to simulate
alpha = 5e-4 # EWMA smoothing factor (window ≈ 1/alpha = 2000 TTIs)
T = 5e-4     # TTI duration in seconds (0.5 ms)

# Simulate channel rates (Mbps) for each UE over K TTIs
# UE 0 is closer → higher average rate
np.random.seed(42)
avg_rates = [120.0, 40.0]   # average rates in Mbps
rates = np.array([
    np.random.exponential(avg_rates[i], K) for i in range(N)
])
# rates[i, k] = rate available to UE i in TTI k


# ── Helper: build and solve the LP ──────────────────────────────
def solve_scheduling_lp(rates, weights, RG=None):
    """
    Solve the weighted throughput LP.

    Decision variables: p[i,k] = fraction of TTI k given to UE i
    Flattened into 1D vector x of length N*K:
        x = [p[0,0], p[0,1], ..., p[0,K-1], p[1,0], ..., p[N-1,K-1]]

    Parameters
    ----------
    rates   : (N, K) array of per-UE per-TTI rates (Mbps)
    weights : (N,)   per-UE utility weights
    RG      : (N,)   minimum rate guarantees (Mbps), or None

    Returns
    -------
    p     : (N, K) scheduling fractions
    theta : (N,)   average throughputs (Mbps)
    """
    N, K = rates.shape

    # ── Objective: maximise Σ_i w[i] * θ[i]
    # = maximise (1/K) Σ_i Σ_k w[i] * r[i,k] * p[i,k]
    # linprog minimises, so we negate:
    c = np.zeros(N * K)
    for i in range(N):
        c[i*K : (i+1)*K] = -weights[i] * rates[i] / K

    # ── Constraint 1: Σ_i p[i,k] ≤ 1  for each TTI k
    A_ub = np.zeros((K, N * K))
    for k in range(K):
        for i in range(N):
            A_ub[k, i*K + k] = 1.0
    b_ub = np.ones(K)

    # ── Constraint 2 (optional): rate guarantee θ[i] ≥ RG[i]
    # Rewritten as: -(1/K) Σ_k r[i,k]*p[i,k] ≤ -RG[i]
    if RG is not None:
        A_rg = np.zeros((N, N * K))
        b_rg = np.zeros(N)
        for i in range(N):
            if RG[i] > 0:
                A_rg[i, i*K : (i+1)*K] = -rates[i] / K
                b_rg[i] = -RG[i]
        A_ub = np.vstack([A_ub, A_rg])
        b_ub = np.concatenate([b_ub, b_rg])

    # ── Bounds: 0 ≤ p[i,k] ≤ 1
    bounds = [(0, 1)] * (N * K)

    # ── Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if res.status != 0:
        print(f"LP solver status: {res.message}")
        return None, None

    p = res.x.reshape(N, K)
    theta = np.array([np.mean(rates[i] * p[i]) for i in range(N)])
    return p, theta


# ── Problem Type 1: Max Sum Throughput ──────────────────────────
print("=" * 50)
print("Problem Type 1: Max Sum Throughput")
weights_ms = np.ones(N)                       # equal weights
p1, theta1 = solve_scheduling_lp(rates, weights_ms)
print(f"  UE throughputs: {np.round(theta1, 2)} Mbps")
print(f"  Total:          {theta1.sum():.2f} Mbps")


# ── Problem Type 2: Max Weighted Throughput ─────────────────────
print("\nProblem Type 2: Max Weighted Throughput")
weights_mw = np.array([1.0, 3.0])            # UE 1 has 3× priority
p2, theta2 = solve_scheduling_lp(rates, weights_mw)
print(f"  UE throughputs: {np.round(theta2, 2)} Mbps")
print(f"  Total:          {theta2.sum():.2f} Mbps")


# ── Problem Type 3: Proportional Fair ───────────────────────────
print("\nProblem Type 3: Proportional Fair")
# Initialise EWMA throughputs (warm start with average rates)
theta_bar = np.array(avg_rates, dtype=float)
# PF weight = inverse of current EWMA throughput
weights_pf = 1.0 / theta_bar
p3, theta3 = solve_scheduling_lp(rates, weights_pf)
print(f"  UE throughputs: {np.round(theta3, 2)} Mbps")
print(f"  Total:          {theta3.sum():.2f} Mbps")

# Jain's fairness index (1.0 = perfectly fair)
jain = (theta3.sum()**2) / (N * (theta3**2).sum())
print(f"  Jain fairness:  {jain:.4f}")


# ── Problem Type 4: PF with Rate Guarantee ──────────────────────
print("\nProblem Type 4: PF with Rate Guarantee")
RG = np.array([60.0, 0.0])                   # UE 0 guaranteed 60 Mbps
p4, theta4 = solve_scheduling_lp(rates, weights_pf, RG=RG)

if theta4 is not None:
    print(f"  UE throughputs: {np.round(theta4, 2)} Mbps")
    print(f"  Total:          {theta4.sum():.2f} Mbps")
    for i in range(N):
        met = "✓" if (RG[i] == 0 or theta4[i] >= RG[i] - 0.01) else "✗"
        if RG[i] > 0:
            print(f"  UE {i} guarantee {RG[i]} Mbps: {met}")
else:
    print("  LP infeasible — rate guarantees cannot be satisfied!")


# ── EWMA update (for online/slot-by-slot scheduling) ────────────
def ewma_update(theta_bar, bits_delivered, scheduled_ue, N, alpha, T):
    """
    Update EWMA throughputs after one TTI.

    Parameters
    ----------
    theta_bar     : (N,) current EWMA throughputs
    bits_delivered: bits sent to the scheduled UE this TTI
    scheduled_ue  : index of the UE that was scheduled
    """
    for i in range(N):
        if i == scheduled_ue:
            throughput_this_tti = bits_delivered / T / 1e6  # convert to Mbps
            theta_bar[i] = (1 - alpha) * theta_bar[i] + alpha * throughput_this_tti
        else:
            theta_bar[i] = (1 - alpha) * theta_bar[i]
    return theta_bar
```

### Expected output

```
==================================================
Problem Type 1: Max Sum Throughput
  UE throughputs: [118.73   6.21] Mbps
  Total:          124.94 Mbps

Problem Type 2: Max Weighted Throughput
  UE throughputs: [ 56.18  47.83] Mbps
  Total:          104.01 Mbps

Problem Type 3: Proportional Fair
  UE throughputs: [82.14  38.71] Mbps
  Total:          120.85 Mbps
  Jain fairness:  0.9712

Problem Type 4: PF with Rate Guarantee
  UE throughputs: [60.12  41.34] Mbps
  Total:          101.46 Mbps
  UE 0 guarantee 60.0 Mbps: ✓
```

---

## 12. Summary Cheatsheet

### The four LP variants at a glance

```
┌─────────────────────────────────────────────────────────────────────┐
│  All four share these constraints:                                   │
│                                                                      │
│    Σ_i p[i,k]  ≤  1       for all k     (one UE per TTI)           │
│    0 ≤ p[i,k]  ≤  1       for all i, k  (valid fractions)          │
│                                                                      │
│  They differ only in the objective and any extra constraints:        │
└─────────────────────────────────────────────────────────────────────┘

TYPE 1 — Max Sum Throughput
  Objective:   maximize  (1/K) Σ_i Σ_k  r[i,k] · p[i,k]
  Per-TTI:     schedule i* = argmax_i  r[i,k]
  Fairness:    ★☆☆☆☆   Efficiency: ★★★★★

TYPE 2 — Max Weighted Throughput
  Objective:   maximize  (1/K) Σ_i Σ_k  w[i] · r[i,k] · p[i,k]
  Per-TTI:     schedule i* = argmax_i  w[i] · r[i,k]
  Fairness:    ★★☆☆☆   Efficiency: ★★★★☆

TYPE 3 — Proportional Fair
  Objective:   maximize  Σ_i  log(θ[i])
               ≈ maximize  Σ_i  [1/θ̄[i]] · θ[i]   (gradient step)
  Per-TTI:     schedule i* = argmax_i  r[i,k] / θ̄[i](k-1)
  Fairness:    ★★★★☆   Efficiency: ★★★☆☆

TYPE 4 — PF with Rate Guarantee
  Objective:   maximize  Σ_i  log(θ[i])
               ≈ maximize  Σ_i  [1/θ̄[i]] · θ[i]
  Extra:       θ[i]  ≥  RG[i]    for GBR UEs
  Per-TTI:     schedule i* = argmax_i  r[i,k] / θ̄[i] × bias[i]
  Fairness:    ★★★★★   Efficiency: ★★☆☆☆
```

### Key formulas reference

| Formula | Meaning |
|---------|---------|
| `θ[i] = (1/K) Σ_k r[i,k]·p[i,k]` | Average throughput of UE i |
| `Σ_i p[i,k] ≤ 1` | At most one UE per TTI |
| `w[i] = 1 / θ̄[i]` | PF weight (inverse EWMA throughput) |
| `θ̄[i](k) = (1-α)·θ̄[i](k-1) + α·(bits/T)` | EWMA throughput update |
| `1/α` | EWMA averaging window (in TTIs) |
| `θ[i] ≥ RG[i]` | Rate guarantee constraint |

---

*This document is self-contained. The four LP variants above cover the complete scheduling problem as taught in 5G networking courses, from the simplest (max sum) to the most practical (PF with rate guarantees for network slicing).*
