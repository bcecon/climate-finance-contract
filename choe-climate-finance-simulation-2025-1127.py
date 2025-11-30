#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Byeong-Hak Choe
"""

"""
"Climate Finance with Limited Commitment and Renegotiation: A Dynamic Contract Approach"

First-best and second-best (recursive contract) climate finance

This script:
  * Solves the first-best problem for C_FB(G)
  * Computes a_FB(G), m_FB(G), A/M ratio and funding shares
  * Plots and saves first-best figures
  * Calibrates a post-renegotiation funding cap from the first-best steady state
  * Simulates a reduced-form second-best contract with repeated renegotiations:
      - Each period, with probability (1-rho) a renegotiation occurs
      - Promised value w_t is non-decreasing between renegotiations
      - Initially, funding equals first-best whenever feasible
      - After renegotiation, funding is capped
  * Plots and saves first-best vs second-best dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. Parameters (baseline from paper + calibration for A/M)
# ============================================================

beta = 0.98          # discount factor
delta = 0.9917       # GHG persistence
G_bar = 613.0        # pre-industrial / baseline GHG stock
G0 = 788.0           # initial G in 1995 (Mason 2017)

A = 1.0e4            # baseline damage level
M = 1.0e2            # baseline emissions term for g(m)

# --- Technology parameters ---
tau   = 0.7          # curvature of mitigation: g(m) = M - tau*log(m)
omega = 15.0         # damage scale
psi   = 15.0         # adaptation intensity

# --- Calibration parameter for adaptation ---
kappa_a = 0.25       # scales a^{FB}(G); chosen to get ~40% adaptation share at G_ss

# Climate sensitivity shocks and probabilities
s_vals = np.array([8.75e-4, 3.9e-3])
pi_vals = np.array([0.5, 0.5])

# Second-best / renegotiation parameter
rho = 0.8            # per-period probability of *no* renegotiation

# ============================================================
# 2. Grids
# ============================================================

# State grid for G
G_min = G_bar
G_max = 12000.0
N_G   = 300
G_vals = np.linspace(G_min, G_max, N_G)

# Control grid in terms of x = log(m)
x_min = 0.0               # m = exp(0) = 1
x_max = 40.0
N_x   = 401               # fairly fine to smooth out kinks
x_grid = np.linspace(x_min, x_max, N_x)

# ============================================================
# 3. First-best adaptation: a_FB(G) and expected damage
# ============================================================

def compute_a_FB_and_D_expect(G_grid, kappa_a=1.0):
    """
    Compute a^{FB}(G) = kappa_a * psi * omega * E[e^{sG}] and
    E[D(a^{FB}(G); G, s)] for all G on the grid.

    D(a; G, s) = omega * exp(s G) * (A - psi * log a)
    """
    # E[e^{s G}]
    E_exp_sG = np.zeros_like(G_grid)
    for s, pi in zip(s_vals, pi_vals):
        E_exp_sG += pi * np.exp(s * G_grid)

    a_FB = kappa_a * psi * omega * E_exp_sG

    # Expected damage at the first-best a(G)
    D_expect = np.zeros_like(G_grid)
    for s, pi in zip(s_vals, pi_vals):
        D_expect += pi * (
            omega * np.exp(s * G_grid) * (A - psi * np.log(a_FB))
        )

    return a_FB, D_expect


a_FB, D_expect = compute_a_FB_and_D_expect(G_vals, kappa_a=kappa_a)

# ============================================================
# 4. Value iteration for C_FB(G) with control m = exp(x)
# ============================================================

def value_iteration_first_best(G_grid, a_FB, D_expect,
                               beta=beta, delta=delta,
                               G_bar=G_bar, M=M, tau=tau,
                               x_grid=x_grid,
                               max_iter=500, tol=1e-4):
    """
    Solve the first-best DP:

      C(G) = min_m [ a_FB(G) + m + E[D(a_FB(G); G, s)] + beta C(G') ]

      G'   = G_bar + delta (G - G_bar) + g(m),
      g(m) = M - tau * log(m),  with m = exp(x).

    Implemented using a grid over x = log(m).
    """

    G = G_grid[:, None]            # (N_G, 1)
    a_mat = a_FB[:, None]          # (N_G, 1)
    D_mat = D_expect[:, None]      # (N_G, 1)

    x_mat = x_grid[None, :]        # (1, N_x)
    m_mat = np.exp(x_mat)          # (1, N_x)

    I_mat = a_mat + m_mat          # total funding (adaptation + mitigation)

    # Initialize value function
    C_old = np.zeros_like(G_grid)

    for it in range(max_iter):
        # Law of motion: G' = G_bar + delta (G - G_bar) + (M - tau * x)
        G_next = G_bar + delta * (G - G_bar) + (M - tau * x_mat)  # (N_G, N_x)

        # Clip G' into the grid range for interpolation
        G_next_clipped = np.clip(G_next, G_grid[0], G_grid[-1])

        # Interpolate C_old(G')
        C_future = np.interp(
            G_next_clipped.ravel(), G_grid, C_old
        ).reshape(G_next.shape)

        # Bellman update
        total_cost = I_mat + D_mat + beta * C_future
        C_new = total_cost.min(axis=1)

        diff = np.max(np.abs(C_new - C_old))
        C_old = C_new

        if it % 50 == 0:
            print(f"[FB VI] Iteration {it}: sup-norm diff = {diff:.3e}")
        if diff < tol:
            print(f"[FB VI] Converged after {it} iterations; diff = {diff:.3e}")
            break

    # Recover policy for m(G)
    G_next = G_bar + delta * (G - G_bar) + (M - tau * x_mat)
    G_next_clipped = np.clip(G_next, G_grid[0], G_grid[-1])
    C_future = np.interp(
        G_next_clipped.ravel(), G_grid, C_old
    ).reshape(G_next.shape)
    total_cost = I_mat + D_mat + beta * C_future

    idx_min = np.argmin(total_cost, axis=1)
    x_star = x_grid[idx_min]
    m_star = np.exp(x_star)

    return C_old, m_star


C_FB, m_FB = value_iteration_first_best(G_vals, a_FB, D_expect)

# ============================================================
# 5. Approximate steady state G_ss under first-best policy
# ============================================================

def compute_G_next_policy(G_grid, m_policy,
                          delta=delta, G_bar=G_bar, M=M, tau=tau):
    """
    Given a policy m(G), compute G'(G) = G_bar + delta (G - G_bar) + g(m(G)),
    with g(m) = M - tau * log(m).
    """
    x_policy = np.log(m_policy)
    g_policy = M - tau * x_policy
    G_next = G_bar + delta * (G_grid - G_bar) + g_policy
    return G_next


G_next_FB = compute_G_next_policy(G_vals, m_FB)
G_diff = G_next_FB - G_vals
idx_ss = np.argmin(np.abs(G_diff))
G_ss_FB = G_vals[idx_ss]

print(f"\nApproximate first-best steady-state G_ss ≈ {G_ss_FB:.2f}, "
      f"G_next - G ≈ {G_diff[idx_ss]:.3e}")

# ============================================================
# 6. Simple smoothing for plotting (moving average)
# ============================================================

def moving_average(y, window=5):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='same')

a_FB_smooth = moving_average(a_FB, window=7)
m_FB_smooth = moving_average(m_FB, window=7)

# ============================================================
# 7. Adaptation–mitigation ratio and funding shares
# ============================================================

I_FB = a_FB + m_FB

am_ratio = np.where(m_FB > 0.0, a_FB / m_FB, np.nan)
share_adaptation = np.where(I_FB > 0.0, a_FB / I_FB, np.nan)
share_mitigation = np.where(I_FB > 0.0, m_FB / I_FB, np.nan)

print(f"At G_ss ≈ {G_ss_FB:.1f}:")
print(f"  a_ss = {a_FB[idx_ss]:.4f}")
print(f"  m_ss = {m_FB[idx_ss]:.4f}")
print(f"  a/m  = {am_ratio[idx_ss]:.4f}")
print(f"  share_a = {share_adaptation[idx_ss]:.4f}, "
      f"share_m = {share_mitigation[idx_ss]:.4f}")

# Restrict plots to a max G (so the vertical line at G_ss is visible)
G_plot_max = 9500.0
mask = (G_vals <= G_plot_max)
G_plot = G_vals[mask]

C_plot = C_FB[mask]
a_plot = a_FB_smooth[mask]
m_plot = m_FB_smooth[mask]

am_ratio_plot = am_ratio[mask]
share_adaptation_plot = share_adaptation[mask]
share_mitigation_plot = share_mitigation[mask]

# ============================================================
# 8. First-best figures
# ============================================================

# ---------- Figure 1: Climate cost and funding levels ----------
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: C_FB(G)
ax1 = axes1[0]
ax1.plot(G_plot, C_plot, label=r"$C_{FB}(G)$")

if G0 <= G_plot_max:
    ax1.axvline(G0, color="gray", linestyle="--", label=r"$G_0$")
if G_ss_FB <= G_plot_max:
    ax1.axvline(G_ss_FB, color="red", linestyle="--", label=r"$G^{FB}_{ss}$")

ax1.set_xlabel("GHG stock $G$")
ax1.set_ylabel("First-best climate cost $C_{FB}(G)$")
ax1.set_title("First-best Climate Cost")
ax1.set_xlim(G_vals[0], G_plot_max)
ax1.legend()
ax1.grid(alpha=0.3)

# Right panel: a_FB(G) and m_FB(G)
ax2 = axes1[1]
ax2.plot(G_plot, a_plot, label=r"Adaptation $a^{FB}(G)$")
ax2.plot(G_plot, m_plot, label=r"Mitigation $m^{FB}(G)$ (smoothed)",
         linestyle="--")

if G0 <= G_plot_max:
    ax2.axvline(G0, color="gray", linestyle=":", label=r"$G_0$")
if G_ss_FB <= G_plot_max:
    ax2.axvline(G_ss_FB, color="red", linestyle=":", label=r"$G^{FB}_{ss}$")

ax2.set_xlabel("GHG stock $G$")
ax2.set_ylabel("Funding level")
ax2.set_title("First-best Adaptation and Mitigation")
ax2.set_xlim(G_vals[0], G_plot_max)
ax2.legend()
ax2.grid(alpha=0.3)

fig1.tight_layout()

# ---------- Figure 2: A/M ratio and funding shares ----------
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Left: A/M ratio
ax3 = axes2[0]
ax3.plot(G_plot, am_ratio_plot)
if G_ss_FB <= G_plot_max:
    ax3.axvline(G_ss_FB, color="red", linestyle="--", label=r"$G^{FB}_{ss}$")
ax3.set_xlabel("GHG stock $G$")
ax3.set_ylabel("Adaptation–Mitigation ratio $a/m$")
ax3.set_title("First-best Adaptation–Mitigation Ratio")
ax3.set_xlim(G_vals[0], G_plot_max)
ax3.grid(alpha=0.3)
ax3.legend()

# Right: funding shares
ax4 = axes2[1]
ax4.plot(G_plot, share_adaptation_plot, label="Adaptation share")
ax4.plot(G_plot, share_mitigation_plot, label="Mitigation share")
if G_ss_FB <= G_plot_max:
    ax4.axvline(G_ss_FB, color="red", linestyle="--", label=r"$G^{FB}_{ss}$")
ax4.set_xlabel("GHG stock $G$")
ax4.set_ylabel("Funding share")
ax4.set_ylim(0.0, 1.0)
ax4.set_title("First-best Adaptation vs Mitigation Shares")
ax4.set_xlim(G_vals[0], G_plot_max)
ax4.grid(alpha=0.3)
ax4.legend()

fig2.tight_layout()

# ============================================================
# 9. Save first-best figures
# ============================================================

script_dir = "/Users/bchoe/My Drive/wp-climate-finance/"
out1_png = os.path.join(script_dir, "first_best_cost_and_policies.png")
out1_pdf = os.path.join(script_dir, "first_best_cost_and_policies.pdf")
out2_png = os.path.join(script_dir, "first_best_AMratio_shares.png")
out2_pdf = os.path.join(script_dir, "first_best_AMratio_shares.pdf")

fig1.savefig(out1_png, dpi=600, bbox_inches="tight")
fig1.savefig(out1_pdf, dpi=600, bbox_inches="tight")
fig2.savefig(out2_png, dpi=600, bbox_inches="tight")
fig2.savefig(out2_pdf, dpi=600, bbox_inches="tight")

print("\nSaved high-res first-best figures to:")
print("  ", out1_png)
print("  ", out1_pdf)
print("  ", out2_png)
print("  ", out2_pdf)

# ============================================================
# 10. Simple interpolants for first-best policies
# ============================================================

def a_FB_interp(G_val):
    return float(np.interp(G_val, G_vals, a_FB))

def m_FB_interp(G_val):
    return float(np.interp(G_val, G_vals, m_FB))

# For convenience: first-best total funding
I_FB_all = a_FB + m_FB

# ============================================================
# 11. Calibrate w_min and initial w_0 from FIRST-BEST AT G0
# ============================================================

rho = 0.8  # renegotiation probability parameter (used below)

# First-best funding at initial G0
G0_clip = float(np.clip(G0, G_vals[0], G_vals[-1]))
I_FB_0  = a_FB_interp(G0_clip) + m_FB_interp(G0_clip)
w_FB_0  = I_FB_0 / (1.0 - beta) if I_FB_0 > 0 else 0.0

# Outside option promise: a fraction of FB promise at G0
theta_w_min = 0.3           # you can tweak 0.2–0.5; must be < 1
w_min = theta_w_min * w_FB_0

# Initial promise: a bit above FB at G0 so we can implement FB initially
theta_w0 = 1.2              # 20% above w_FB_0
w0 = theta_w0 * w_FB_0

print("\nSecond-best calibration (based on G0):")
print(f"  I_FB(G0)    ≈ {I_FB_0:.4f}")
print(f"  w_FB_0      ≈ {w_FB_0:.4f}")
print(f"  w_min       ≈ {w_min:.4f} (= {theta_w_min*100:.0f}% of w_FB_0)")
print(f"  w_0         ≈ {w0:.4f} (= {theta_w0*100:.0f}% of w_FB_0)")
print(f"  cap after reneg (I_max = (1-β) w_min) ≈ {(1.0-beta)*w_min:.4f}")

# ============================================================
# 12. Helper: law of motion for G under given m (same form as FB)
# ============================================================

def g_m_from_m(m):
    # same g(m) as in first-best: g(m) = M - tau * log(m), with m>0
    m_safe = max(m, 1e-8)
    return M - tau * np.log(m_safe)

def next_G_from_m(G, m):
    return G_bar + delta * (G - G_bar) + g_m_from_m(m)

# ============================================================
# 13. Simulate first-best path (for comparison)
# ============================================================

T_sim = 150

G_FB_path = np.zeros(T_sim + 1)
a_FB_path = np.zeros(T_sim)
m_FB_path = np.zeros(T_sim)
I_FB_path = np.zeros(T_sim)

G_FB_path[0] = G0

for t in range(T_sim):
    G_t = float(np.clip(G_FB_path[t], G_vals[0], G_vals[-1]))
    a_t = a_FB_interp(G_t)
    m_t = m_FB_interp(G_t)
    I_t = a_t + m_t

    a_FB_path[t] = a_t
    m_FB_path[t] = m_t
    I_FB_path[t] = I_t

    G_next = next_G_from_m(G_t, m_t)
    G_FB_path[t+1] = float(np.clip(G_next, G_vals[0], G_vals[-1]))

# ============================================================
# 14. Simulate second-best with upward-drifting w_t and random drops
# ============================================================

rng = np.random.default_rng(12345)

G_SB_path = np.zeros(T_sim + 1)
w_SB_path = np.zeros(T_sim + 1)
a_SB_path = np.zeros(T_sim)
m_SB_path = np.zeros(T_sim)
I_SB_path = np.zeros(T_sim)

G_SB_path[0] = G0
w_SB_path[0] = w0      # from the calibration block based on G0

# 0.3 is quite high..
phi_w = 0.1            # speed of upward drift when no reneg occurs

reneg_times = []       # store periods t where renegotiation happens

for t in range(T_sim):
    G_t = float(np.clip(G_SB_path[t], G_vals[0], G_vals[-1]))
    w_t = w_SB_path[t]

    # First-best at current G_t
    a_fb_t = a_FB_interp(G_t)
    m_fb_t = m_FB_interp(G_t)
    I_fb_t = a_fb_t + m_fb_t
    w_needed_t = I_fb_t / (1.0 - beta) if I_fb_t > 0 else 0.0

    # 1. Funding given current promise
    if (w_t >= w_needed_t) and (I_fb_t > 0):
        # Can implement first-best exactly
        a_t = a_fb_t
        m_t = m_fb_t
        I_t = I_fb_t
    else:
        # Promise is binding: max total funding is (1-β) * w_t
        I_t = (1.0 - beta) * w_t
        if I_t <= 0:
            a_t = 0.0
            m_t = 0.0
        else:
            # Keep FB adaptation share at this G_t
            share_a_fb_t = a_fb_t / I_fb_t if I_fb_t > 0 else 0.0
            a_t = share_a_fb_t * I_t
            m_t = I_t - a_t

    a_SB_path[t] = a_t
    m_SB_path[t] = m_t
    I_SB_path[t] = I_t

    # 2. G dynamics under SB mitigation
    G_next = next_G_from_m(G_t, m_t)
    G_SB_path[t+1] = float(np.clip(G_next, G_vals[0], G_vals[-1]))

    # 3. Renegotiation shock at the end of period t
    if rng.random() > rho:
        # Renegotiation: promised value drops to a random fraction U∼U(0.2,0.67)
        u = rng.uniform(0.2, 0.67)
        # Optionally keep a floor at w_min:
        w_next = max(w_min, u * w_t)
        reneg_times.append(t+1)   # drop shows at t+1
    else:
        # No renegotiation: let w drift upwards (but never down)
        G_next_clip = float(np.clip(G_SB_path[t+1], G_vals[0], G_vals[-1]))
        a_fb_next = a_FB_interp(G_next_clip)
        m_fb_next = m_FB_interp(G_next_clip)
        I_fb_next = a_fb_next + m_fb_next
        w_needed_next = I_fb_next / (1.0 - beta) if I_fb_next > 0 else 0.0

        if w_needed_next > w_t:
            # drift part-way toward the FB promise required at G_{t+1}
            w_proposed = w_t + phi_w * (w_needed_next - w_t)
            w_next = w_proposed     # no cap, just upward drift
        else:
            # enforce non-decreasing w_t between renegotiations
            w_next = w_t

    w_SB_path[t+1] = w_next

# First-best A/M ratio along the simulated path
am_ratio_FB_path = np.where(m_FB_path > 0, a_FB_path / m_FB_path, np.nan)
# Second-best: a_t / m_t
am_ratio_SB = np.where(
    m_SB_path > 0,
    a_SB_path / m_SB_path,
    np.nan
)

# Difference in GHG stock: FB − SB
dG_path = G_SB_path - G_FB_path[:T_sim+1]

# ============================================================
# 15. Second-best vs first-best plots (6 panels)
# ============================================================
fig3, axes3 = plt.subplots(3, 2, figsize=(12, 12))

# (a) ΔG_t = G_t^{SB} − G_t^{FB}
axes3[0, 0].plot(range(T_sim + 1), dG_path, label=r"$G_t^{SB} - G_t^{FB}$")
axes3[0, 0].axhline(0.0, color='black', linestyle='-', linewidth=0.8)
for t_r in reneg_times:
    axes3[0, 0].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[0, 0].set_xlabel("t")
axes3[0, 0].set_ylabel(r"$G_t^{SB} - G_t^{FB}$")
axes3[0, 0].set_title("Difference in GHG stock: second-best minus first-best")
axes3[0, 0].legend()
axes3[0, 0].grid(alpha=0.3)

# (b) Promised value w_t (second-best)
axes3[0, 1].plot(
    range(T_sim + 1),
    w_SB_path,
    marker='o',
    markersize=3,      # smaller dots
    linestyle='-'
)
for t_r in reneg_times:
    axes3[0, 1].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[0, 1].set_xlabel("t")
axes3[0, 1].set_ylabel(r"$w_t$")
axes3[0, 1].set_title(r"Promised value $w_t$")
axes3[0, 1].grid(alpha=0.3)

# (c) Adaptation funding: first-best vs second-best
axes3[1, 0].plot(range(T_sim), a_FB_path[:T_sim], label=r"First-best $a_t^{FB}$")
axes3[1, 0].plot(range(T_sim), a_SB_path, '--', label=r"Second-best $a_t$")
for t_r in reneg_times:
    axes3[1, 0].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[1, 0].set_xlabel("t")
axes3[1, 0].set_ylabel("Adaptation funding")
axes3[1, 0].set_title("Adaptation funding: first-best vs second-best")
axes3[1, 0].legend()
axes3[1, 0].grid(alpha=0.3)

# (d) Mitigation funding: first-best vs second-best
axes3[1, 1].plot(range(T_sim), m_FB_path[:T_sim], label=r"First-best $m_t^{FB}$")
axes3[1, 1].plot(range(T_sim), m_SB_path, '--', label=r"Second-best $m_t$")
for t_r in reneg_times:
    axes3[1, 1].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[1, 1].set_xlabel("t")
axes3[1, 1].set_ylabel("Mitigation funding")
axes3[1, 1].set_title("Mitigation funding: first-best vs second-best")
axes3[1, 1].legend()
axes3[1, 1].grid(alpha=0.3)

# (e) Total funding: first-best vs second-best
axes3[2, 0].plot(range(T_sim), I_FB_path[:T_sim], label=r"First-best $I_t^{FB}$")
axes3[2, 0].plot(range(T_sim), I_SB_path, '--', label=r"Second-best $I_t$")
for t_r in reneg_times:
    axes3[2, 0].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[2, 0].set_xlabel("t")
axes3[2, 0].set_ylabel("Funding")
axes3[2, 0].set_title("Total funding: first-best vs second-best")
axes3[2, 0].legend()
axes3[2, 0].grid(alpha=0.3)

# (f) Adaptation–mitigation ratio a/m: FB vs SB
axes3[2, 1].plot(
    range(T_sim),
    am_ratio_FB_path,
    label=r"First-best $a_t^{FB} / m_t^{FB}$"
)
axes3[2, 1].plot(
    range(T_sim),
    am_ratio_SB,
    '--',
    label=r"Second-best $a_t / m_t$"
)
for t_r in reneg_times:
    axes3[2, 1].axvline(t_r, color='gray', linestyle='--', alpha=0.4)
axes3[2, 1].set_xlabel("t")
axes3[2, 1].set_ylabel(r"Adaptation–Mitigation ratio $a_t/m_t$")
axes3[2, 1].set_title("Adaptation–Mitigation ratio: first-best vs second-best")
axes3[2, 1].grid(alpha=0.3)
axes3[2, 1].legend()

fig3.tight_layout()

# ============================================================
# 16. Save high-resolution second-best vs first-best figure
# ============================================================

out3_png = os.path.join(script_dir, "second_best_vs_first_best.png")
out3_pdf = os.path.join(script_dir, "second_best_vs_first_best.pdf")

fig3.savefig(out3_png, dpi=600, bbox_inches="tight")
fig3.savefig(out3_pdf, bbox_inches="tight")

print("\nSaved high-res second-best figure to:")
print("  ", out3_png)
print("  ", out3_pdf)

plt.show()