"""
5G Radio Scheduling — LP Solver
================================
Solves all four scheduling LP problem types with structured output.

Results are saved to:
    results/ms/        Max Sum Throughput
    results/mw/        Max Weighted Throughput
    results/pf/        Proportional Fair
    results/rg/        PF with Rate Guarantee
    results/all/       All four compared
    results/compare/   Rate-Guarantee sweep

Each subfolder gets:
    experiment_<ts>.log     Full console output
    data_<ts>.npz           Raw numpy arrays (theta, p, rates)
    plot_<ts>.png           Plots (if --plot is passed)

Usage examples:
    python 5g_scheduling_lp.py --mode ms
    python 5g_scheduling_lp.py --mode pf --N 4 --K 1000 --plot
    python 5g_scheduling_lp.py --mode rg --N 3 --rg 80 50 0 --plot
    python 5g_scheduling_lp.py --mode all --N 3 --K 1000 --rg 60 0 0 --plot
    python 5g_scheduling_lp.py --mode compare --N 2 --K 500 --plot

Requirements:
    pip install numpy scipy matplotlib
"""

import argparse
import os
import sys
import datetime
import numpy as np
from scipy.optimize import linprog

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — works on all machines
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ── Colour palettes ──────────────────────────────────────────────
# One fixed colour per scheduler label (used in scatter / legend)
SCHED_COLORS = {
    "MS":    "#7F77DD",
    "MWT":   "#D85A30",
    "PF":    "#1D9E75",
    "PF-RG": "#378ADD",
}
# Per-UE colours (bars, lines, histograms)
UE_COLORS = ["#7F77DD", "#1D9E75", "#D85A30", "#378ADD",
             "#BA7517", "#BA3560", "#6B8E23", "#CC8844"]


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — Output directory & logging
# ═══════════════════════════════════════════════════════════════════

def make_outdir(mode):
    """Create results/<mode>/ beside the script; return (path, timestamp)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(script_dir, "results", mode)
    os.makedirs(outdir, exist_ok=True)
    return outdir, ts


class Tee:
    """Write to both terminal and a log file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def save_arrays(outdir, ts, **arrays):
    path = os.path.join(outdir, f"data_{ts}.npz")
    np.savez(path, **arrays)
    print(f"[SAVED] Raw data   → {path}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — Rate generation
# ═══════════════════════════════════════════════════════════════════

def generate_rates(N, K, seed=42, model="exponential"):
    """
    Generate per-TTI channel rates (Mbps) for N UEs over K TTIs.

    UE 0 is closest to the gNB (120 Mbps avg); last UE is farthest (30 Mbps).
    Exponential distribution approximates Rayleigh fading.
    """
    np.random.seed(seed)
    avg_rates = np.linspace(120, 30, N)
    rates = np.zeros((N, K))
    for i in range(N):
        if model == "exponential":
            rates[i] = np.random.exponential(avg_rates[i], K)
        elif model == "uniform":
            rates[i] = np.random.uniform(avg_rates[i] * 0.5,
                                          avg_rates[i] * 1.5, K)
    return rates, avg_rates


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — Core LP solver
# ═══════════════════════════════════════════════════════════════════

def solve_lp(rates, weights, RG=None, verbose=False):
    """
    Solve the weighted scheduling LP.

    Formulation
    -----------
    maximize   (1/K) * Σ_i Σ_k  weights[i] * rates[i,k] * p[i,k]

    subject to  Σ_i p[i,k]  ≤  1           ∀ k   (one UE per TTI)
                θ[i]         ≥  RG[i]       ∀ GBR UEs  (optional)
                0 ≤ p[i,k]  ≤  1           ∀ i, k

    Decision vector x = [p[0,0],...,p[0,K-1], p[1,0],...,p[N-1,K-1]]
    """
    N, K = rates.shape

    # Objective — negated for minimisation
    c = np.zeros(N * K)
    for i in range(N):
        c[i*K:(i+1)*K] = -weights[i] * rates[i] / K

    # Constraint 1: Σ_i p[i,k] ≤ 1  for each TTI k
    A_ub = np.zeros((K, N * K))
    for k in range(K):
        for i in range(N):
            A_ub[k, i*K + k] = 1.0
    b_ub = np.ones(K)

    # Constraint 2 (optional): θ[i] ≥ RG[i]
    #   → -(1/K) Σ_k rates[i,k]*p[i,k]  ≤  -RG[i]
    if RG is not None:
        RG_arr = np.asarray(RG, dtype=float)
        active = np.where(RG_arr > 0)[0]
        if len(active):
            A_rg = np.zeros((len(active), N * K))
            b_rg = np.zeros(len(active))
            for row, i in enumerate(active):
                A_rg[row, i*K:(i+1)*K] = -rates[i] / K
                b_rg[row] = -RG_arr[i]
            A_ub = np.vstack([A_ub, A_rg])
            b_ub = np.concatenate([b_ub, b_rg])

    bounds = [(0.0, 1.0)] * (N * K)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if verbose:
        print(f"  [LP] {res.message}  (status={res.status})")

    if res.status != 0:
        return {"p": None, "theta": None, "total": None,
                "jain": None, "status": res.message, "feasible": False}

    p     = res.x.reshape(N, K)
    theta = np.array([np.mean(rates[i] * p[i]) for i in range(N)])
    total = float(theta.sum())
    jain  = float((total**2) / (N * np.sum(theta**2))) if total > 0 else 0.0
    return {"p": p, "theta": theta, "total": total,
            "jain": jain, "status": res.message, "feasible": True}


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — Online slot-by-slot simulation
# ═══════════════════════════════════════════════════════════════════

def run_online_simulation(rates, mode, RG=None, weights=None,
                          alpha=5e-4, T=5e-4):
    """Greedy per-TTI scheduling — mirrors real gNB behaviour."""
    N, K = rates.shape
    theta_bar     = np.array([np.mean(rates[i]) for i in range(N)], dtype=float)
    theta_history = np.zeros((N, K))
    scheduled     = np.zeros(K, dtype=int)
    if weights is None:
        weights = np.ones(N)

    for k in range(K):
        r_k = rates[:, k]
        if mode == "ms":
            scores = r_k.copy()
        elif mode == "mw":
            scores = weights * r_k
        elif mode in ("pf", "rg"):
            scores = r_k / np.maximum(theta_bar, 1e-9)
            if RG is not None and mode == "rg":
                RG_arr = np.asarray(RG, dtype=float)
                for i in range(N):
                    if RG_arr[i] > 0 and theta_bar[i] < RG_arr[i]:
                        scores[i] *= 3.0

        i_star = int(np.argmax(scores))
        scheduled[k] = i_star
        for i in range(N):
            bits = rates[i, k] if i == i_star else 0.0
            theta_bar[i] = ((1 - alpha) * theta_bar[i]
                            + alpha * (bits / T / 1e6))
        theta_history[:, k] = theta_bar.copy()

    return theta_history, scheduled


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — Print helpers
# ═══════════════════════════════════════════════════════════════════

def print_result(label, result, RG=None, avg_rates=None):
    N   = len(result["theta"]) if result["feasible"] else 0
    bar = "─" * 56

    print(f"\n{'═'*56}")
    print(f"  {label}")
    print(f"{'═'*56}")

    if not result["feasible"]:
        print(f"  ✗  LP INFEASIBLE: {result['status']}")
        if RG is not None:
            print(f"  Guarantees requested : {np.asarray(RG)} Mbps")
        if avg_rates is not None:
            print(f"  Max achievable rates : {np.round(avg_rates, 1)} Mbps")
        return

    RG_arr = np.asarray(RG, dtype=float) if RG is not None else None
    print(f"  {'UE':<5} {'Avg Rate':>11} {'Throughput':>12} "
          f"{'Share':>7}  {'RG':>9}  {'Met':>4}")
    print(f"  {bar}")
    for i in range(N):
        avg_str = f"{avg_rates[i]:.1f} Mbps" if avg_rates is not None else "  —"
        rg_str  = (f"{RG_arr[i]:.0f} Mbps"
                   if RG_arr is not None and RG_arr[i] > 0 else "  —")
        met = ""
        if RG_arr is not None and RG_arr[i] > 0:
            met = "✓" if result["theta"][i] >= RG_arr[i] - 0.5 else "✗"
        share = result["theta"][i] / result["total"] * 100
        print(f"  UE {i:<2}  {avg_str:>11}  {result['theta'][i]:>9.2f} Mbps"
              f"  {share:>6.1f}%  {rg_str:>9}  {met:>4}")
    print(f"  {bar}")
    print(f"  {'Total':>23}  {result['total']:>9.2f} Mbps")
    print(f"  {'Jain fairness index':>23}  {result['jain']:>9.4f}"
          f"  (1.0 = perfectly fair)")


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — Plot functions (one per mode)
# ═══════════════════════════════════════════════════════════════════

def _uc(N):
    return UE_COLORS[:N]


def plot_single(result, rates, avg_rates, outdir, ts, title):
    """Three-panel summary for a single mode."""
    if not MATPLOTLIB_AVAILABLE:
        print("[INFO] matplotlib unavailable — skipping plots.")
        return
    N  = rates.shape[0]
    uc = _uc(N)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"5G Scheduling — {title}", fontsize=12, fontweight="bold")

    # Panel A: throughput bar
    ax = axes[0]
    bars = ax.bar(range(N), result["theta"], color=uc,
                  edgecolor="white", linewidth=0.8)
    ax.plot(range(N), avg_rates, "k--o", markersize=4,
            linewidth=1.2, label="Avg channel rate")
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"UE {i}" for i in range(N)])
    ax.set_ylabel("Mbps")
    ax.set_title("Throughput per UE")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, result["theta"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    # Panel B: channel histograms
    ax = axes[1]
    for i in range(N):
        ax.hist(rates[i], bins=40, alpha=0.6, color=uc[i],
                label=f"UE {i}  μ={avg_rates[i]:.0f}", density=True)
    ax.set_xlabel("Instantaneous Rate (Mbps)")
    ax.set_ylabel("Density")
    ax.set_title("Channel Distributions")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel C: scheduling fraction pie
    ax = axes[2]
    if result["p"] is not None:
        fracs = result["p"].sum(axis=1)
        fracs = fracs / fracs.sum()
        ax.pie(fracs,
               labels=[f"UE {i}\n{fracs[i]*100:.1f}%" for i in range(N)],
               colors=uc, startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Scheduling Share")

    plt.tight_layout()
    path = os.path.join(outdir, f"plot_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot       → {path}")


def plot_all(results_dict, rates, avg_rates, online_histories, outdir, ts):
    """Five-panel comparison for --mode all."""
    if not MATPLOTLIB_AVAILABLE:
        print("[INFO] matplotlib unavailable — skipping plots.")
        return
    N  = rates.shape[0]
    uc = _uc(N)

    # Feasible entries only
    feasible_items  = [(l, r) for l, r in results_dict.items() if r["feasible"]]
    feasible_labels = [l for l, _ in feasible_items]
    n_s = len(feasible_items)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("5G LP Scheduling — Full Comparison",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # Panel 1: grouped bar — per-UE throughput
    ax1 = fig.add_subplot(gs[0, 0])
    width = 0.8 / n_s
    x = np.arange(N)
    for j, (lbl, res) in enumerate(feasible_items):
        offset = (j - n_s / 2 + 0.5) * width
        ax1.bar(x + offset, res["theta"], width,
                label=lbl,
                color=SCHED_COLORS.get(lbl, uc[j % len(uc)]),
                alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"UE {i}" for i in range(N)])
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title("Per-UE Throughput by Scheduler")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: scatter — efficiency vs fairness
    # FIX: build colour list with one entry per scheduler (length = n_s)
    ax2 = fig.add_subplot(gs[0, 1])
    sc_totals = [r["total"] for _, r in feasible_items]
    sc_jains  = [r["jain"]  for _, r in feasible_items]
    sc_colors = [SCHED_COLORS.get(l, "#888888") for l in feasible_labels]

    ax2.scatter(sc_jains, sc_totals,
                s=140, zorder=5,
                c=sc_colors,        # one colour per point — length matches x/y
                edgecolors="k", linewidths=0.6)
    for lbl, xv, yv in zip(feasible_labels, sc_jains, sc_totals):
        ax2.annotate(lbl, (xv, yv),
                     textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax2.set_xlabel("Jain Fairness Index →")
    ax2.set_ylabel("Total Throughput (Mbps) →")
    ax2.set_title("Efficiency vs Fairness Tradeoff")
    ax2.set_xlim(0, 1.08)
    ax2.grid(alpha=0.3)

    # Panel 3: channel rate histograms
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(N):
        ax3.hist(rates[i], bins=40, alpha=0.6, color=uc[i],
                 label=f"UE {i}  μ={avg_rates[i]:.0f}", density=True)
    ax3.set_xlabel("Instantaneous Rate (Mbps)")
    ax3.set_ylabel("Density")
    ax3.set_title("Channel Rate Distributions")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # Panel 4: EWMA convergence
    if online_histories:
        ax4 = fig.add_subplot(gs[1, :2])
        t = np.arange(rates.shape[1])
        linestyles = ["-", "--", ":", "-."]
        for j, (sched_lbl, history) in enumerate(online_histories.items()):
            for i in range(N):
                ax4.plot(t, history[i],
                         color=uc[i],
                         linestyle=linestyles[j % 4],
                         alpha=0.75, linewidth=1.2,
                         label=f"{sched_lbl} UE{i}")
        ax4.set_xlabel("TTI index (k)")
        ax4.set_ylabel("EWMA Throughput (Mbps)")
        ax4.set_title("EWMA Throughput Convergence (Online Simulation)")
        ax4.legend(fontsize=7, ncol=min(4, n_s * N))
        ax4.grid(alpha=0.3)

    # Panel 5: PF scheduling share pie
    ax5 = fig.add_subplot(gs[1, 2])
    pf_res = results_dict.get("PF") or results_dict.get("pf")
    if pf_res and pf_res["feasible"] and pf_res["p"] is not None:
        fracs = pf_res["p"].sum(axis=1)
        fracs = fracs / fracs.sum()
        ax5.pie(fracs,
                labels=[f"UE {i}\n{fracs[i]*100:.1f}%" for i in range(N)],
                colors=uc, startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax5.set_title("PF Scheduling Share")

    path = os.path.join(outdir, f"comparison_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot       → {path}")


def plot_compare(rg_list, totals, jains, feasibles, outdir, ts):
    """Two-panel sweep plot for --mode compare."""
    if not MATPLOTLIB_AVAILABLE:
        print("[INFO] matplotlib unavailable — skipping plots.")
        return
    valid = [(rg, t, j) for rg, t, j, f in
             zip(rg_list, totals, jains, feasibles) if f]
    if not valid:
        print("[WARN] No feasible points to plot.")
        return
    rgs, tots, jns = zip(*valid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Rate Guarantee Sweep — Efficiency vs Fairness",
                 fontsize=12, fontweight="bold")
    axes[0].plot(rgs, tots, "o-", color="#7F77DD", linewidth=2)
    axes[0].set_xlabel("Rate Guarantee for UE0 (Mbps)")
    axes[0].set_ylabel("Total Network Throughput (Mbps)")
    axes[0].set_title("Guarantee ↑  →  Total Throughput ↓")
    axes[0].grid(alpha=0.3)
    axes[1].plot(rgs, jns, "o-", color="#1D9E75", linewidth=2)
    axes[1].set_xlabel("Rate Guarantee for UE0 (Mbps)")
    axes[1].set_ylabel("Jain Fairness Index")
    axes[1].set_title("Guarantee ↑  →  Fairness shifts")
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, f"rg_sweep_{ts}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] Plot       → {path}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — Experiment runners
# ═══════════════════════════════════════════════════════════════════

def run_max_sum(rates, avg_rates, outdir, ts, verbose=False, do_plot=False):
    weights = np.ones(rates.shape[0])
    result  = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 1 — Max Sum Throughput  [U(θ) = θ]",
                 result, avg_rates=avg_rates)
    if result["feasible"]:
        save_arrays(outdir, ts, theta=result["theta"],
                    rates=rates, avg_rates=avg_rates)
        if do_plot:
            plot_single(result, rates, avg_rates, outdir, ts,
                        "Max Sum Throughput")
    return result


def run_max_weighted(rates, avg_rates, outdir, ts,
                     weights=None, verbose=False, do_plot=False):
    N = rates.shape[0]
    if weights is None:
        weights = np.arange(N, 0, -1, dtype=float)
    print(f"\n  [Config] Weights : {np.round(weights, 3)}")
    result = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 2 — Max Weighted Throughput  [U(θ) = w·θ]",
                 result, avg_rates=avg_rates)
    if result["feasible"]:
        save_arrays(outdir, ts, theta=result["theta"], rates=rates,
                    avg_rates=avg_rates, weights=weights)
        if do_plot:
            plot_single(result, rates, avg_rates, outdir, ts,
                        "Max Weighted Throughput")
    return result


def run_proportional_fair(rates, avg_rates, outdir, ts,
                          alpha=5e-4, verbose=False, do_plot=False):
    N         = rates.shape[0]
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)])
    weights   = 1.0 / np.maximum(theta_bar, 1e-9)
    print(f"\n  [Config] EWMA init (avg rates) : {np.round(theta_bar, 2)} Mbps")
    print(f"  [Config] PF weights (1/θ̄)      : {np.round(weights, 5)}")
    result = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 3 — Proportional Fair  [U(θ) = log(θ)]",
                 result, avg_rates=avg_rates)
    if result["feasible"]:
        save_arrays(outdir, ts, theta=result["theta"], rates=rates,
                    avg_rates=avg_rates, weights=weights)
        if do_plot:
            plot_single(result, rates, avg_rates, outdir, ts,
                        "Proportional Fair")
    return result


def run_rate_guarantee(rates, avg_rates, RG, outdir, ts,
                       alpha=5e-4, verbose=False, do_plot=False):
    N      = rates.shape[0]
    RG_arr = np.asarray(RG, dtype=float)
    if len(RG_arr) != N:
        print(f"[ERROR] --rg needs exactly {N} values, got {len(RG_arr)}")
        sys.exit(1)
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)])
    weights   = 1.0 / np.maximum(theta_bar, 1e-9)
    print(f"\n  [Config] Rate Guarantees : {RG_arr} Mbps")
    print(f"  [Config] PF weights      : {np.round(weights, 5)}")
    result = solve_lp(rates, weights, RG=RG_arr, verbose=verbose)
    print_result("Problem Type 4 — PF with Rate Guarantee (GBR)",
                 result, RG=RG_arr, avg_rates=avg_rates)
    if result["feasible"]:
        save_arrays(outdir, ts, theta=result["theta"], rates=rates,
                    avg_rates=avg_rates, RG=RG_arr)
        if do_plot:
            plot_single(result, rates, avg_rates, outdir, ts,
                        "PF with Rate Guarantee")
    return result


def run_all(rates, avg_rates, outdir, ts,
            RG=None, alpha=5e-4, verbose=False, do_plot=False):
    results   = {}
    histories = {}

    results["MS"]  = run_max_sum(rates, avg_rates, outdir, ts, verbose)
    results["MWT"] = run_max_weighted(rates, avg_rates, outdir, ts,
                                      verbose=verbose)
    results["PF"]  = run_proportional_fair(rates, avg_rates, outdir, ts,
                                           alpha, verbose)
    if RG is not None:
        results["PF-RG"] = run_rate_guarantee(rates, avg_rates, RG,
                                              outdir, ts, alpha, verbose)

    histories["MS"], _ = run_online_simulation(rates, "ms", alpha=alpha)
    histories["PF"], _ = run_online_simulation(rates, "pf", alpha=alpha)
    if RG is not None:
        histories["PF-RG"], _ = run_online_simulation(
            rates, "rg", RG=np.asarray(RG), alpha=alpha)

    # Summary table
    print(f"\n{'═'*62}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*62}")
    print(f"  {'Scheduler':<12} {'Total (Mbps)':>14} "
          f"{'Jain Index':>12} {'Feasible':>12}")
    print(f"  {'─'*54}")
    for lbl, res in results.items():
        if res["feasible"]:
            print(f"  {lbl:<12} {res['total']:>14.2f} "
                  f"{res['jain']:>12.4f} {'Yes':>12}")
        else:
            print(f"  {lbl:<12} {'—':>14} {'—':>12} "
                  f"{'No (infeasible)':>12}")
    print(f"  {'─'*54}")

    if do_plot:
        plot_all(results, rates, avg_rates, histories, outdir, ts)
    return results


def run_compare(rates, avg_rates, outdir, ts,
                alpha=5e-4, do_plot=False):
    N = rates.shape[0]
    if N < 2:
        print("[ERROR] --mode compare requires N ≥ 2")
        sys.exit(1)

    max_rate   = np.mean(rates[0])
    rg_values  = np.linspace(0, max_rate * 0.9, 15)
    theta_bar  = np.array([np.mean(rates[i]) for i in range(N)])
    weights_pf = 1.0 / np.maximum(theta_bar, 1e-9)

    print(f"\n{'═'*62}")
    print(f"  EXPERIMENT: Rate Guarantee Sweep for UE0")
    print(f"  UE0 average channel rate = {max_rate:.1f} Mbps")
    print(f"{'═'*62}")
    print(f"  {'RG[0] (Mbps)':>14} {'Total (Mbps)':>14} "
          f"{'Jain':>10} {'Feasible':>10}")
    print(f"  {'─'*52}")

    totals, jains, feasibles = [], [], []
    for rg0 in rg_values:
        RG = np.zeros(N); RG[0] = rg0
        res = solve_lp(rates, weights_pf, RG=RG)
        feasibles.append(res["feasible"])
        if res["feasible"]:
            totals.append(res["total"])
            jains.append(res["jain"])
            print(f"  {rg0:>14.1f} {res['total']:>14.2f} "
                  f"{res['jain']:>10.4f} {'✓':>10}")
        else:
            totals.append(None)
            jains.append(None)
            print(f"  {rg0:>14.1f} {'—':>14} {'—':>10} "
                  f"{'✗ infeasible':>10}")

    save_arrays(outdir, ts,
                rg_values=rg_values,
                totals=np.array([t if t else np.nan for t in totals]),
                jains=np.array([j if j else np.nan  for j in jains]))
    if do_plot:
        plot_compare(rg_values, totals, jains, feasibles, outdir, ts)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="5G Scheduling LP Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  ms       Max Sum Throughput
  mw       Max Weighted Throughput  (use --weights to set per-UE priority)
  pf       Proportional Fair
  rg       PF with Rate Guarantee   (requires --rg)
  all      All four modes compared
  compare  Sweep rate guarantee → tradeoff curve

Output structure
----------------
  results/
    ms/       experiment_<ts>.log  data_<ts>.npz  plot_<ts>.png
    mw/       ...
    pf/       ...
    rg/       ...
    all/      ...
    compare/  ...

Examples
--------
  python 5g_scheduling_lp.py --mode ms --N 2 --K 500 --plot
  python 5g_scheduling_lp.py --mode mw --N 3 --weights 1 3 2 --plot
  python 5g_scheduling_lp.py --mode rg --N 3 --rg 80 50 0 --plot
  python 5g_scheduling_lp.py --mode all --N 3 --K 1000 --rg 60 0 0 --plot
  python 5g_scheduling_lp.py --mode compare --N 2 --K 500 --plot
        """
    )
    p.add_argument("--mode",    default="all",
                   choices=["ms", "mw", "pf", "rg", "all", "compare"])
    p.add_argument("--N",       type=int,   default=2)
    p.add_argument("--K",       type=int,   default=500)
    p.add_argument("--alpha",   type=float, default=5e-4)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--rg",      type=float, nargs="+",
                   help="Rate guarantees in Mbps, one per UE")
    p.add_argument("--weights", type=float, nargs="+",
                   help="Weights for --mode mw, one per UE")
    p.add_argument("--channel", default="exponential",
                   choices=["exponential", "uniform"])
    p.add_argument("--plot",    action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — Entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    outdir, ts = make_outdir(args.mode)
    log_path   = os.path.join(outdir, f"experiment_{ts}.log")
    tee        = Tee(log_path)
    sys.stdout = tee

    try:
        print(f"\n{'═'*62}")
        print("  5G SCHEDULING LP SOLVER")
        print(f"{'═'*62}")
        print(f"  Mode            : {args.mode}")
        print(f"  UEs (N)         : {args.N}")
        print(f"  TTIs (K)        : {args.K}")
        print(f"  EWMA alpha      : {args.alpha}"
              f"  (window ≈ {int(1/args.alpha)} TTIs)")
        print(f"  Channel model   : {args.channel}")
        print(f"  Random seed     : {args.seed}")
        if args.rg:
            print(f"  Rate guarantees : {args.rg} Mbps")
        if args.weights:
            print(f"  UE weights      : {args.weights}")
        print(f"  Output dir      : {outdir}")

        rates, avg_rates = generate_rates(args.N, args.K,
                                          seed=args.seed,
                                          model=args.channel)
        print(f"\n  Average rates per UE : {np.round(avg_rates, 1)} Mbps")

        if args.mode == "ms":
            run_max_sum(rates, avg_rates, outdir, ts,
                        args.verbose, args.plot)

        elif args.mode == "mw":
            w = np.array(args.weights) if args.weights else None
            run_max_weighted(rates, avg_rates, outdir, ts,
                             weights=w, verbose=args.verbose,
                             do_plot=args.plot)

        elif args.mode == "pf":
            run_proportional_fair(rates, avg_rates, outdir, ts,
                                  args.alpha, args.verbose, args.plot)

        elif args.mode == "rg":
            if args.rg is None:
                print("[ERROR] --mode rg requires --rg. E.g. --rg 80 40 0")
                sys.exit(1)
            run_rate_guarantee(rates, avg_rates, args.rg,
                               outdir, ts, args.alpha,
                               args.verbose, args.plot)

        elif args.mode == "all":
            run_all(rates, avg_rates, outdir, ts,
                    RG=args.rg, alpha=args.alpha,
                    verbose=args.verbose, do_plot=args.plot)

        elif args.mode == "compare":
            run_compare(rates, avg_rates, outdir, ts,
                        alpha=args.alpha, do_plot=args.plot)

        print(f"\n[DONE] All outputs in : {outdir}\n")

    finally:
        sys.stdout = tee.terminal
        tee.close()
        print(f"[LOG]  Saved → {log_path}")


if __name__ == "__main__":
    main()