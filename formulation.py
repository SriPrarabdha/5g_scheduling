"""
5G Radio Scheduling — LP Solver
================================
Run all four scheduling problem types from the command line.

Usage examples:
    python 5g_scheduling_lp.py --mode ms
    python 5g_scheduling_lp.py --mode pf --N 4 --K 1000
    python 5g_scheduling_lp.py --mode rg --N 3 --rg 80 50 0
    python 5g_scheduling_lp.py --mode all --N 3 --K 800 --plot
    python 5g_scheduling_lp.py --mode compare --N 2 --plot

Requirements:
    pip install numpy scipy matplotlib
"""

import argparse
import sys
import numpy as np
from scipy.optimize import linprog

# ── Try importing matplotlib (optional, only needed for --plot) ──
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: Rate Generation
# ═══════════════════════════════════════════════════════════════════

def generate_rates(N, K, seed=42, model="exponential"):
    """
    Generate per-TTI channel rates (Mbps) for N UEs over K TTIs.

    Parameters
    ----------
    N     : number of UEs
    K     : number of TTIs
    seed  : random seed for reproducibility
    model : 'exponential' (Rayleigh-like fading) or 'uniform'

    Returns
    -------
    rates    : (N, K) array of rates in Mbps
    avg_rates: (N,)   average rate per UE
    """
    np.random.seed(seed)
    # UEs are spread from 200m to 500m from gNB — closer UEs get higher rates
    avg_rates = np.linspace(120, 30, N)

    rates = np.zeros((N, K))
    for i in range(N):
        if model == "exponential":
            # Exponential distribution models Rayleigh fading well
            rates[i] = np.random.exponential(avg_rates[i], K)
        elif model == "uniform":
            rates[i] = np.random.uniform(avg_rates[i] * 0.5,
                                          avg_rates[i] * 1.5, K)
    return rates, avg_rates


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: Core LP Solver
# ═══════════════════════════════════════════════════════════════════

def solve_lp(rates, weights, RG=None, verbose=False):
    """
    Solve the weighted throughput LP.

    Formulation
    -----------
    maximize   (1/K) * Σ_i Σ_k  weights[i] * rates[i,k] * p[i,k]

    subject to  Σ_i p[i,k]  ≤  1          for all k   (one UE per TTI)
                θ[i]         ≥  RG[i]      for GBR UEs (optional)
                0 ≤ p[i,k]  ≤  1          for all i, k

    Where θ[i] = (1/K) * Σ_k rates[i,k] * p[i,k]

    Parameters
    ----------
    rates   : (N, K) per-UE per-TTI rates in Mbps
    weights : (N,)   per-UE utility weights
    RG      : (N,)   rate guarantees in Mbps, or None
    verbose : print solver details

    Returns
    -------
    dict with keys: p, theta, total, jain, status, feasible
    """
    N, K = rates.shape

    # ── Objective vector (negated for minimisation) ──────────────
    # x is flattened: x = [p[0,0],...,p[0,K-1], p[1,0],...,p[N-1,K-1]]
    c = np.zeros(N * K)
    for i in range(N):
        c[i*K:(i+1)*K] = -weights[i] * rates[i] / K

    # ── Inequality constraints: Σ_i p[i,k] ≤ 1 for each k ───────
    # Shape: (K, N*K)
    A_ub = np.zeros((K, N * K))
    for k in range(K):
        for i in range(N):
            A_ub[k, i*K + k] = 1.0
    b_ub = np.ones(K)

    # ── Rate guarantee constraints (optional) ────────────────────
    # θ[i] ≥ RG[i]  →  -(1/K)*Σ_k rates[i,k]*p[i,k] ≤ -RG[i]
    if RG is not None:
        A_rg = np.zeros((N, N * K))
        b_rg = np.zeros(N)
        for i in range(N):
            if RG[i] > 0:
                A_rg[i, i*K:(i+1)*K] = -rates[i] / K
                b_rg[i] = -RG[i]
        # Only add rows where RG > 0
        active = RG > 0
        if active.any():
            A_ub = np.vstack([A_ub, A_rg[active]])
            b_ub = np.concatenate([b_ub, b_rg[active]])

    # ── Variable bounds ──────────────────────────────────────────
    bounds = [(0.0, 1.0)] * (N * K)

    # ── Solve ────────────────────────────────────────────────────
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if verbose:
        print(f"  [LP] Status: {res.message} (code {res.status})")

    feasible = res.status == 0
    if not feasible:
        return {
            "p": None, "theta": None, "total": None,
            "jain": None, "status": res.message, "feasible": False
        }

    p = res.x.reshape(N, K)
    theta = np.array([np.mean(rates[i] * p[i]) for i in range(N)])
    total = theta.sum()
    jain  = (total ** 2) / (N * np.sum(theta ** 2)) if total > 0 else 0.0

    return {
        "p": p,
        "theta": theta,
        "total": total,
        "jain": jain,
        "status": res.message,
        "feasible": True
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: EWMA Throughput Tracker (online simulation)
# ═══════════════════════════════════════════════════════════════════

def run_online_simulation(rates, mode, RG=None, alpha=5e-4, T=5e-4):
    """
    Simulate slot-by-slot scheduling (greedy per-TTI rule).
    This mirrors what a real gNB scheduler does in hardware.

    Returns
    -------
    theta_history : (N, K) EWMA throughput over time
    scheduled     : (K,)   index of UE scheduled each TTI
    """
    N, K = rates.shape
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)], dtype=float)
    theta_history = np.zeros((N, K))
    scheduled = np.zeros(K, dtype=int)

    for k in range(K):
        r_k = rates[:, k]  # rates available to each UE this TTI

        # ── Per-TTI scheduling rule ──────────────────────────────
        if mode == "ms":
            # Max sum: schedule UE with highest rate
            scores = r_k

        elif mode == "mw":
            # Max weighted: multiply by fixed weights
            # (weights stored as global for this demo)
            scores = r_k  # weights applied outside if needed

        elif mode in ("pf", "rg"):
            # Proportional fair: r[i,k] / theta_bar[i]
            # with bias for UEs behind their guarantee
            scores = r_k / np.maximum(theta_bar, 1e-9)
            if RG is not None and mode == "rg":
                for i in range(N):
                    if RG[i] > 0 and theta_bar[i] < RG[i]:
                        scores[i] *= 3.0  # bias factor β

        i_star = int(np.argmax(scores))
        scheduled[k] = i_star

        # ── EWMA update ──────────────────────────────────────────
        for i in range(N):
            bits = rates[i, k] if i == i_star else 0.0
            tp = bits / T / 1e6  # bits → Mbps
            theta_bar[i] = (1 - alpha) * theta_bar[i] + alpha * tp

        theta_history[:, k] = theta_bar.copy()

    return theta_history, scheduled


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: Printing Results
# ═══════════════════════════════════════════════════════════════════

def print_result(label, result, RG=None, avg_rates=None):
    N = len(result["theta"]) if result["feasible"] else 0
    bar = "─" * 50

    print(f"\n{'═'*50}")
    print(f"  {label}")
    print(f"{'═'*50}")

    if not result["feasible"]:
        print(f"  ✗ LP INFEASIBLE: {result['status']}")
        if RG is not None:
            print(f"  Rate guarantees requested: {RG} Mbps")
            if avg_rates is not None:
                print(f"  Max achievable rates:      {np.round(avg_rates, 1)} Mbps")
        return

    print(f"  {'UE':<6} {'Avg Rate':>12} {'Throughput':>12} {'Share':>8}  {'RG':>8}  {'Met':>4}")
    print(f"  {bar}")
    for i in range(N):
        avg = f"{avg_rates[i]:.1f} Mbps" if avg_rates is not None else "  —"
        rg_str = f"{RG[i]:.0f} Mbps" if (RG is not None and RG[i] > 0) else "  —"
        met = ""
        if RG is not None and RG[i] > 0:
            met = "✓" if result["theta"][i] >= RG[i] - 0.5 else "✗"
        share = result["theta"][i] / result["total"] * 100
        print(f"  UE {i:<2}  {avg:>12}  {result['theta'][i]:>9.2f} Mbps"
              f"  {share:>6.1f}%  {rg_str:>8}  {met:>4}")

    print(f"  {bar}")
    print(f"  {'Total':>20}  {result['total']:>9.2f} Mbps")
    print(f"  {'Jain fairness index':>20}  {result['jain']:>9.4f}  (1.0 = perfectly fair)")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_results(results_dict, rates, avg_rates, online_histories=None):
    if not MATPLOTLIB_AVAILABLE:
        print("\n[INFO] matplotlib not installed — skipping plots.")
        print("       Install with: pip install matplotlib")
        return

    N = rates.shape[0]
    colors = ["#7F77DD", "#1D9E75", "#D85A30", "#378ADD", "#BA7517"][:N]
    labels = list(results_dict.keys())

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("5G LP Scheduling — Results Comparison", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Throughput per UE per scheduler ──────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(N)
    width = 0.8 / len(labels)
    for j, (lbl, res) in enumerate(results_dict.items()):
        if res["feasible"]:
            offset = (j - len(labels)/2 + 0.5) * width
            ax1.bar(x + offset, res["theta"], width,
                    label=lbl, color=colors[j % len(colors)], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"UE {i}" for i in range(N)])
    ax1.set_ylabel("Throughput (Mbps)")
    ax1.set_title("Per-UE Throughput")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── Plot 2: Total throughput vs Jain fairness ────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    totals = [r["total"] for r in results_dict.values() if r["feasible"]]
    jains  = [r["jain"]  for r in results_dict.values() if r["feasible"]]
    valid_labels = [l for l, r in results_dict.items() if r["feasible"]]
    sc = ax2.scatter(jains, totals, s=120, zorder=5,
                     c=colors[:len(totals)], edgecolors="k", linewidths=0.5)
    for lbl, x_val, y_val in zip(valid_labels, jains, totals):
        ax2.annotate(lbl, (x_val, y_val),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax2.set_xlabel("Jain Fairness Index →")
    ax2.set_ylabel("Total Throughput (Mbps) →")
    ax2.set_title("Efficiency vs Fairness Tradeoff")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1.05)

    # ── Plot 3: Rate distribution per UE ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(N):
        ax3.hist(rates[i], bins=40, alpha=0.6, color=colors[i],
                 label=f"UE {i} (avg {avg_rates[i]:.0f} Mbps)", density=True)
    ax3.set_xlabel("Instantaneous Rate (Mbps)")
    ax3.set_ylabel("Density")
    ax3.set_title("Channel Rate Distributions")
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # ── Plot 4: EWMA throughput convergence (online sim) ─────────
    if online_histories:
        ax4 = fig.add_subplot(gs[1, :2])
        K = rates.shape[1]
        t = np.arange(K)
        for j, (lbl, history) in enumerate(online_histories.items()):
            for i in range(N):
                ax4.plot(t, history[i], alpha=0.8,
                         color=colors[i],
                         linestyle=["-", "--", ":", "-."][j % 4],
                         label=f"{lbl} UE{i}")
        ax4.set_xlabel("TTI index (k)")
        ax4.set_ylabel("EWMA Throughput (Mbps)")
        ax4.set_title("EWMA Throughput Convergence (Online Simulation)")
        ax4.legend(fontsize=7, ncol=2)
        ax4.grid(alpha=0.3)

    # ── Plot 5: Scheduling fraction pie charts ───────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    if "pf" in results_dict and results_dict["pf"]["feasible"]:
        fracs = results_dict["pf"]["p"].sum(axis=1)
        fracs = fracs / fracs.sum()
        ax5.pie(fracs, labels=[f"UE {i}\n{fracs[i]*100:.1f}%" for i in range(N)],
                colors=colors[:N], autopct="", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax5.set_title("PF Scheduling Share")

    plt.savefig("/mnt/user-data/outputs/5g_scheduling_results.png",
                dpi=150, bbox_inches="tight")
    print("\n[INFO] Plot saved → 5g_scheduling_results.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: Individual Experiment Runners
# ═══════════════════════════════════════════════════════════════════

def run_max_sum(rates, avg_rates, verbose=False):
    N = rates.shape[0]
    weights = np.ones(N)
    result = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 1 — Max Sum Throughput  [U(θ) = θ]",
                 result, avg_rates=avg_rates)
    return result


def run_max_weighted(rates, avg_rates, weights=None, verbose=False):
    N = rates.shape[0]
    if weights is None:
        weights = np.arange(N, 0, -1, dtype=float)  # default: UE0 gets highest weight
    print(f"\n  [Config] Weights: {np.round(weights, 2)}")
    result = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 2 — Max Weighted Throughput  [U(θ) = w·θ]",
                 result, avg_rates=avg_rates)
    return result


def run_proportional_fair(rates, avg_rates, alpha=5e-4, verbose=False):
    N = rates.shape[0]
    # PF weight = 1 / EWMA(theta), initialised with average rates
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)], dtype=float)
    weights = 1.0 / np.maximum(theta_bar, 1e-9)
    print(f"\n  [Config] EWMA init (avg rates): {np.round(theta_bar, 2)} Mbps")
    print(f"  [Config] PF weights (1/θ̄):      {np.round(weights, 5)}")
    result = solve_lp(rates, weights, verbose=verbose)
    print_result("Problem Type 3 — Proportional Fair  [U(θ) = log(θ)]",
                 result, avg_rates=avg_rates)
    return result


def run_rate_guarantee(rates, avg_rates, RG, alpha=5e-4, verbose=False):
    N = rates.shape[0]
    if len(RG) != N:
        print(f"[ERROR] --rg must have exactly {N} values, got {len(RG)}")
        sys.exit(1)
    RG = np.array(RG, dtype=float)
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)], dtype=float)
    weights = 1.0 / np.maximum(theta_bar, 1e-9)
    print(f"\n  [Config] Rate Guarantees: {RG} Mbps")
    print(f"  [Config] PF weights:      {np.round(weights, 5)}")
    result = solve_lp(rates, weights, RG=RG, verbose=verbose)
    print_result("Problem Type 4 — PF with Rate Guarantee (GBR)",
                 result, RG=RG, avg_rates=avg_rates)
    return result


def run_all(rates, avg_rates, RG=None, alpha=5e-4, verbose=False, plot=False):
    """Run all four problem types and compare."""
    results = {}
    histories = {}

    results["MS"]  = run_max_sum(rates, avg_rates, verbose)
    results["MWT"] = run_max_weighted(rates, avg_rates, verbose=verbose)
    results["PF"]  = run_proportional_fair(rates, avg_rates, alpha, verbose)

    if RG is not None:
        results["PF-RG"] = run_rate_guarantee(rates, avg_rates, RG, alpha, verbose)

    # Online simulations for convergence plots
    histories["MS"], _  = run_online_simulation(rates, "ms", alpha=alpha)
    histories["PF"], _  = run_online_simulation(rates, "pf", alpha=alpha)
    if RG is not None:
        histories["PF-RG"], _ = run_online_simulation(
            rates, "rg", RG=np.array(RG), alpha=alpha)

    # Summary table
    print(f"\n{'═'*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Scheduler':<12} {'Total (Mbps)':>14} {'Jain Index':>12} {'Feasible':>10}")
    print(f"  {'─'*54}")
    for lbl, res in results.items():
        if res["feasible"]:
            print(f"  {lbl:<12} {res['total']:>14.2f} {res['jain']:>12.4f} {'Yes':>10}")
        else:
            print(f"  {lbl:<12} {'—':>14} {'—':>12} {'No (infeasible)':>10}")
    print(f"  {'─'*54}")

    if plot:
        plot_results(results, rates, avg_rates, histories)

    return results


def run_compare(rates, avg_rates, alpha=5e-4, plot=False):
    """
    Sweep rate guarantees to show the efficiency–fairness tradeoff.
    Useful for the professor experiment.
    """
    N = rates.shape[0]
    if N < 2:
        print("[ERROR] --mode compare requires N ≥ 2")
        sys.exit(1)

    max_rate = np.mean(rates[0])  # UE0 is closest → highest average rate
    rg_values = np.linspace(0, max_rate * 0.9, 15)

    print(f"\n{'═'*60}")
    print("  EXPERIMENT: Sweeping Rate Guarantee for UE0")
    print(f"  (UE0 avg rate = {max_rate:.1f} Mbps)")
    print(f"{'═'*60}")
    print(f"  {'RG[0] (Mbps)':>14} {'Total (Mbps)':>14} {'Jain':>10} {'Feasible':>10}")
    print(f"  {'─'*52}")

    totals, jains, rg_list, feasibles = [], [], [], []
    theta_bar = np.array([np.mean(rates[i]) for i in range(N)], dtype=float)
    weights_pf = 1.0 / np.maximum(theta_bar, 1e-9)

    for rg0 in rg_values:
        RG = np.zeros(N)
        RG[0] = rg0
        res = solve_lp(rates, weights_pf, RG=RG)
        feasibles.append(res["feasible"])
        rg_list.append(rg0)
        if res["feasible"]:
            totals.append(res["total"])
            jains.append(res["jain"])
            print(f"  {rg0:>14.1f} {res['total']:>14.2f} {res['jain']:>10.4f} {'✓':>10}")
        else:
            totals.append(None)
            jains.append(None)
            print(f"  {rg0:>14.1f} {'—':>14} {'—':>10} {'✗ infeasible':>10}")

    if plot and MATPLOTLIB_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Rate Guarantee Sweep — Efficiency vs Fairness", fontsize=12)

        valid = [(rg, t, j) for rg, t, j, f in
                 zip(rg_list, totals, jains, feasibles) if f]
        rgs, tots, jns = zip(*valid) if valid else ([], [], [])

        axes[0].plot(rgs, tots, "o-", color="#7F77DD", linewidth=2)
        axes[0].set_xlabel("Rate Guarantee for UE0 (Mbps)")
        axes[0].set_ylabel("Total Network Throughput (Mbps)")
        axes[0].set_title("Guarantee ↑ → Total Throughput ↓")
        axes[0].grid(alpha=0.3)

        axes[1].plot(rgs, jns, "o-", color="#1D9E75", linewidth=2)
        axes[1].set_xlabel("Rate Guarantee for UE0 (Mbps)")
        axes[1].set_ylabel("Jain Fairness Index")
        axes[1].set_title("Guarantee ↑ → Fairness changes non-linearly")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("/mnt/user-data/outputs/5g_rg_sweep.png",
                    dpi=150, bbox_inches="tight")
        print("\n[INFO] Plot saved → 5g_rg_sweep.png")
        plt.show()


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CLI Argument Parser
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="5G Scheduling LP Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scheduling modes:
  ms       Max Sum Throughput         — always pick the highest-rate UE
  mw       Max Weighted Throughput    — priority weights per UE
  pf       Proportional Fair          — balance efficiency and fairness
  rg       PF with Rate Guarantee     — GBR constraint for specific UEs
  all      Run all four modes         — side-by-side comparison
  compare  Sweep RG values            — see efficiency/fairness tradeoff curve

Examples:
  python 5g_scheduling_lp.py --mode ms
  python 5g_scheduling_lp.py --mode pf --N 4 --K 1000 --plot
  python 5g_scheduling_lp.py --mode rg --N 3 --K 500 --rg 80 50 0 --plot
  python 5g_scheduling_lp.py --mode all --N 3 --rg 60 0 0 --plot
  python 5g_scheduling_lp.py --mode compare --N 2 --K 500 --plot
        """
    )

    parser.add_argument("--mode", type=str, default="all",
                        choices=["ms", "mw", "pf", "rg", "all", "compare"],
                        help="Scheduling mode to run (default: all)")
    parser.add_argument("--N", type=int, default=2,
                        help="Number of UEs (default: 2)")
    parser.add_argument("--K", type=int, default=500,
                        help="Number of TTIs / scheduling slots (default: 500)")
    parser.add_argument("--alpha", type=float, default=5e-4,
                        help="EWMA smoothing factor (default: 5e-4, window=2000 TTIs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--rg", type=float, nargs="+", default=None,
                        help="Rate guarantees in Mbps, one per UE. E.g. --rg 80 50 0")
    parser.add_argument("--weights", type=float, nargs="+", default=None,
                        help="Weights for max-weighted mode, one per UE. E.g. --weights 1 3 2")
    parser.add_argument("--channel", type=str, default="exponential",
                        choices=["exponential", "uniform"],
                        help="Channel model for rate generation (default: exponential)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate and save matplotlib plots")
    parser.add_argument("--verbose", action="store_true",
                        help="Print LP solver details")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: Main Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    print(f"\n{'═'*60}")
    print("  5G SCHEDULING LP SOLVER")
    print(f"{'═'*60}")
    print(f"  UEs (N):         {args.N}")
    print(f"  TTIs (K):        {args.K}")
    print(f"  EWMA alpha:      {args.alpha}  (window ≈ {int(1/args.alpha)} TTIs)")
    print(f"  Channel model:   {args.channel}")
    print(f"  Random seed:     {args.seed}")
    print(f"  Mode:            {args.mode}")
    if args.rg:
        print(f"  Rate guarantees: {args.rg} Mbps")
    if args.weights:
        print(f"  UE weights:      {args.weights}")

    # Generate channel rates
    rates, avg_rates = generate_rates(args.N, args.K,
                                       seed=args.seed,
                                       model=args.channel)
    print(f"\n  Generated rates — avg per UE: {np.round(avg_rates, 1)} Mbps")

    # Dispatch to selected mode
    if args.mode == "ms":
        run_max_sum(rates, avg_rates, args.verbose)

    elif args.mode == "mw":
        run_max_weighted(rates, avg_rates,
                         weights=np.array(args.weights) if args.weights else None,
                         verbose=args.verbose)

    elif args.mode == "pf":
        run_proportional_fair(rates, avg_rates, args.alpha, args.verbose)

    elif args.mode == "rg":
        if args.rg is None:
            print("[ERROR] --mode rg requires --rg values. E.g. --rg 80 40 0")
            sys.exit(1)
        run_rate_guarantee(rates, avg_rates, args.rg, args.alpha, args.verbose)

    elif args.mode == "all":
        run_all(rates, avg_rates,
                RG=args.rg,
                alpha=args.alpha,
                verbose=args.verbose,
                plot=args.plot)

    elif args.mode == "compare":
        run_compare(rates, avg_rates, args.alpha, plot=args.plot)

    print()


if __name__ == "__main__":
    main()
