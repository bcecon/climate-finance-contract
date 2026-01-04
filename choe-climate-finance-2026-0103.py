#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:32:42 2025

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

# %%
# ------------------------------------------------------------
# Python Libraries
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from pathlib import Path
import os

try:
    script_dir = str(Path(__file__).resolve().parent) + os.sep
except NameError:
    script_dir = os.getcwd() + os.sep
    

# %% 
# ------------------------------------------------------------
# Visualization: Adaptation and Mitigation Funding (2000-2023)
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

# Data source:
# Fan, S., Wang, C., Zhong, H. et al. 
# International Multilateral Public Climate Finance Dataset from 2000 to 2023. 
# Sci Data 12, 952 (2025). 
# https://doi.org/10.1038/s41597-025-05308-x
# https://doi.org/10.6084/m9.figshare.28171535.v1

file_path = (
    "/Users/bchoe/My Drive/wp-climate-finance/"
    "choe-mdpi-supp/"
    "FanEtAl2025_climate_finance_data_2000_2023.xlsx"
)

funding = pd.read_excel(file_path)


funding_dir = os.path.join(script_dir, "FanEtAl2025_figures")
os.makedirs(funding_dir, exist_ok=True)

out_level_png = os.path.join(funding_dir, "climate_finance_levels_2025_1222.png")
out_level_pdf = os.path.join(funding_dir, "climate_finance_levels_2025_1222.pdf")

out_share_png = os.path.join(funding_dir, "climate_finance_shares_2025_1222.png")
out_share_pdf = os.path.join(funding_dir, "climate_finance_shares_2025_1222.pdf")

# ---- aggregate ----
funding_sum = (
    funding
    .groupby(["Year", "climate_class"], as_index=False)["Financing"]
    .sum()
)

funding_sum["Financing"] = (funding_sum["Financing"] / 1_000_000_000).round(0)

wide = (funding_sum
        .pivot(index="Year", columns="climate_class", values="Financing")
        .fillna(0)
        .sort_index()
       )

stack_order  = ["Mitigation", "Adaptation"]   # bottom -> top
legend_order = ["Adaptation", "Mitigation"]   # legend order

pal = sns.color_palette("Set2", n_colors=2)
color_map = {
    "Adaptation": pal[0],  # swapped
    "Mitigation": pal[1],  # swapped
}
colors_stack = [color_map[c] for c in stack_order]

# make sure wide has both cols in any order you want
wide = wide.reindex(columns=["Adaptation", "Mitigation"]).fillna(0)

# ===== LEVELS FIGURE =====
fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.stackplot(
    wide.index,
    [wide[c].values for c in stack_order],
    labels=stack_order,
    colors=colors_stack,
    alpha=0.75
)

ax1.set_xlabel("Year")
ax1.set_ylabel("USD in billion")

# legend order: Adaptation first, then Mitigation
handles, labels = ax1.get_legend_handles_labels()
hmap = dict(zip(labels, handles))
ax1.legend([hmap[k] for k in legend_order], legend_order, title="Funding Type", loc="upper left")

plt.tight_layout()

fig1.savefig(out_level_png, dpi=600, bbox_inches="tight")
fig1.savefig(out_level_pdf, bbox_inches="tight")

print("\nSaved high-res climate finance LEVEL figures to:")
print("  ", out_level_png)
print("  ", out_level_pdf)

plt.show()


# ===== SHARES (100% STACK) FIGURE =====
row_totals = wide.sum(axis=1).replace(0, pd.NA)
wide_share = wide.div(row_totals, axis=0).fillna(0)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.stackplot(
    wide_share.index,
    [wide_share[c].values for c in stack_order],
    labels=stack_order,
    colors=colors_stack,
    alpha=0.75
)

ax2.set_xlabel("Year")
ax2.set_ylabel("Share of yearly total funding")
ax2.set_ylim(0, 1)
ax2.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")

# legend order: Adaptation first, then Mitigation
handles, labels = ax2.get_legend_handles_labels()
hmap = dict(zip(labels, handles))
# ax2.legend([hmap[k] for k in legend_order], legend_order, title="Funding Type", loc="upper left")

plt.tight_layout()

# NOTE: save fig2 here (you had fig1 by mistake)
fig2.savefig(out_share_png, dpi=600, bbox_inches="tight")
fig2.savefig(out_share_pdf, bbox_inches="tight")

print("\nSaved high-res climate finance SHARE figures to:")
print("  ", out_share_png)
print("  ", out_share_pdf)

plt.show()

# %% First-best Simulation
# ============================================================
# 1. Parameters (baseline from paper + calibration for A/M)
# ============================================================

beta = 0.98          # discount factor
delta = 0.9917       # GHG persistence
G_bar = 613.0        # pre-industrial / baseline GHG stock
G0 = 854.2           # initial G (year 2015)

A = 1e4            # baseline damage level
M = 20              # baseline emissions term for g(m)

# --- Technology parameters ---
tau   = 1.1957           # curvature of mitigation: g(m) = M - tau*log(1+m)
omega = 100         # damage scale
psi   = 400         # adaptation intensity

# Climate sensitivity shocks and probabilities
s_vals = np.array([8.75e-4, 3.9e-3])
pi_vals = np.array([0.5, 0.5])

# ============================================================
# 2. Grids
# ============================================================

G_min = G_bar
G_max = 3000
N_G   = 2000
G_vals = np.linspace(G_min, G_max, N_G)

# Control grid in terms of x = log(1+m)
x_min = 0
x_max = 40
N_x   = 401
x_grid = np.linspace(x_min, x_max, N_x)

# ============================================================
# 3. First-best adaptation: a_FB(G) and expected damage
# ============================================================

def compute_a_FB_and_D_expect(G_grid):
    """
    a^{FB}(G) = psi * omega * E[e^{sG}]
    D(a;G,s)  = omega * exp(sG) * (A - psi * log(1 + a))
    """
    E_exp_sG = np.zeros_like(G_grid)
    for s, pi in zip(s_vals, pi_vals):
        E_exp_sG += pi * np.exp(s * G_grid)

    a_FB =  psi * omega * E_exp_sG

    D_expect = np.zeros_like(G_grid)
    for s, pi in zip(s_vals, pi_vals):
        D_expect += pi * (omega * np.exp(s * G_grid) * (A - psi * np.log(1 + a_FB)))

    return a_FB, D_expect

a_FB, D_expect = compute_a_FB_and_D_expect(G_vals)

# ============================================================
# 4. Value iteration for C_FB(G)
# ============================================================

def value_iteration_first_best(G_grid, a_FB, D_expect,
                               beta=beta, delta=delta,
                               G_bar=G_bar, M=M, tau=tau,
                               x_grid=x_grid,
                               max_iter=1000, tol=1e-6):
    """
    C(G) = min_m [ a_FB(G) + m + E[D(a_FB(G);G,s)] + beta C(G') ]
    G'   = G_bar + delta (G - G_bar) + g(m)
    g(m) = M - tau * log(1+m), with x = log(1+m)

    Key upgrade:
      - After VI converges, recover policy using a 3-point quadratic fit
        around the discrete argmin to smooth out grid-jumps.
    """
    G = G_grid[:, None]
    a_mat = a_FB[:, None]
    D_mat = D_expect[:, None]

    x_mat = x_grid[None, :]
    m_mat = np.exp(x_mat) - 1
    I_mat = a_mat + m_mat

    C_old = np.zeros_like(G_grid)

    for it in range(max_iter):
        G_next = G_bar + delta * (G - G_bar) + (M - tau * x_mat)
        G_next_clipped = np.clip(G_next, G_grid[0], G_grid[-1])
        C_future = np.interp(G_next_clipped.ravel(), G_grid, C_old).reshape(G_next.shape)

        total_cost = I_mat + D_mat + beta * C_future
        C_new = total_cost.min(axis=1)

        diff = np.max(np.abs(C_new - C_old))
        C_old = C_new

        if it % 50 == 0:
            print(f"[FB VI] Iteration {it}: sup-norm diff = {diff:.3e}")
        if diff < tol:
            print(f"[FB VI] Converged after {it} iterations; diff = {diff:.3e}")
            break

    # ---------- Policy recovery (with quadratic refinement) ----------
    G_next = G_bar + delta * (G - G_bar) + (M - tau * x_mat)
    G_next_clipped = np.clip(G_next, G_grid[0], G_grid[-1])
    C_future = np.interp(G_next_clipped.ravel(), G_grid, C_old).reshape(G_next.shape)
    total_cost = I_mat + D_mat + beta * C_future   # (N_G, N_x)

    idx = np.argmin(total_cost, axis=1)            # discrete argmin index

    x_star = np.empty_like(G_grid, dtype=float)

    # 3-point quadratic fit per G_i (skip boundaries)
    for i in range(len(G_grid)):
        j = int(idx[i])

        if j == 0 or j == len(x_grid) - 1:
            x_star[i] = x_grid[j]
            continue

        x0, x1, x2 = x_grid[j-1], x_grid[j], x_grid[j+1]
        f0, f1, f2 = total_cost[i, j-1], total_cost[i, j], total_cost[i, j+1]

        # Fit f(x) = a x^2 + b x + c through (x0,f0), (x1,f1), (x2,f2)
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) < 1e-14:
            x_star[i] = x1
            continue

        a = (x2*(f1 - f0) + x1*(f0 - f2) + x0*(f2 - f1)) / denom
        b = (x2**2*(f0 - f1) + x1**2*(f2 - f0) + x0**2*(f1 - f2)) / denom

        # Vertex of parabola: x* = -b/(2a)
        if a <= 0:
            # if curvature isn't convex numerically, fall back to grid point
            x_star[i] = x1
        else:
            xv = -b / (2*a)
            # clamp within [x0,x2] to stay local
            x_star[i] = float(np.clip(xv, x0, x2))

    m_star = np.exp(x_star) - 1
    m_star = np.maximum(m_star, 0)

    return C_old, m_star

C_FB, m_FB = value_iteration_first_best(G_vals, a_FB, D_expect)

# ============================================================
# 5. Approximate steady state G_ss under first-best policy
# ============================================================

def compute_G_next_policy(G_grid, m_policy,
                          delta=delta, G_bar=G_bar, M=M, tau=tau):
    """
    g(m) = M - tau*log(1+m)
    """
    x_policy = np.log(1 + np.maximum(m_policy, 0))
    g_policy = M - tau * x_policy
    G_next = G_bar + delta * (G_grid - G_bar) + g_policy
    return G_next, g_policy

G_next_FB, g_FB = compute_G_next_policy(G_vals, m_FB)
G_diff = G_next_FB - G_vals
idx_ss = int(np.argmin(np.abs(G_diff)))
G_ss_FB = float(G_vals[idx_ss])

print(f"\nApproximate first-best steady-state G_ss ≈ {G_ss_FB:.2f}, "
      f"G_next - G ≈ {G_diff[idx_ss]:.3e}")


# ============================================================
# 6. Simulate first-best time path G_t
# ============================================================

def simulate_path_first_best(G0, T, G_grid, m_policy,
                             delta=delta, G_bar=G_bar, M=M, tau=tau):
    """
    G_{t+1} = G_bar + delta (G_t - G_bar) + (M - tau*log(1+m(G_t))).
    """
    G_path = np.empty(T + 1)
    m_path = np.empty(T)
    g_path = np.empty(T)

    G_path[0] = float(G0)

    for t in range(T):
        Gt = float(np.clip(G_path[t], G_grid[0], G_grid[-1]))
        mt = float(np.interp(Gt, G_grid, m_policy))
        mt = max(mt, 0)

        gt = M - tau * np.log(1 + mt)

        G_next = G_bar + delta * (Gt - G_bar) + gt
        G_path[t + 1] = float(np.clip(G_next, G_grid[0], G_grid[-1]))

        m_path[t] = mt
        g_path[t] = gt

    return G_path, m_path, g_path

T_sim = 500  # "annual" periods; adjust as you like
G_path_FB, m_path_FB, g_path_FB = simulate_path_first_best(G0, T_sim, G_vals, m_FB)


# ============================================================
# 7. Funding shares (keep) + plotting ranges
# ============================================================

I_FB = a_FB + m_FB
share_adaptation = np.where(I_FB > 0, a_FB / I_FB, np.nan)
share_mitigation = np.where(I_FB > 0, m_FB / I_FB, np.nan)

print(f"At G_ss ≈ {G_ss_FB:.1f}:")
print(f"  a_ss = {a_FB[idx_ss]:.4f}")
print(f"  m_ss = {m_FB[idx_ss]:.4f}")
print(f"  share_a = {share_adaptation[idx_ss]:.4f}, share_m = {share_mitigation[idx_ss]:.4f}")

G_plot_max = G_ss_FB * 1.1
mask = (G_vals <= G_plot_max)
G_plot = G_vals[mask]

C_plot = C_FB[mask]
a_plot = a_FB[mask]
m_plot = m_FB[mask]

share_adaptation_plot = share_adaptation[mask]
share_mitigation_plot = share_mitigation[mask]

# ============================================================
# 8. Figures
# ============================================================

# ---------- helper interpolants for a(G), m(G) ----------
def a_FB_interp(G_val):
    return float(np.interp(G_val, G_vals, a_FB))

def m_FB_interp(G_val):
    return float(np.interp(G_val, G_vals, m_FB))
        
def force_x_from_2015(ax, start_year, T_sim):
    ax.set_xlim(start_year, start_year + T_sim)   
    ticks = ax.get_xticks()
    ticks = [t for t in ticks if t >= start_year]
    if start_year not in ticks:
        ticks = [start_year] + ticks
    ax.set_xticks(sorted(set(ticks)))
    

# ---------- build time series of funding levels ----------
# funding is evaluated along the simulated G_path_FB
a_path_FB = np.empty(T_sim)
m_path_FB_levels = np.empty(T_sim)

for t in range(T_sim):
    Gt = float(np.clip(G_path_FB[t], G_vals[0], G_vals[-1]))
    a_path_FB[t] = max(a_FB_interp(Gt), 0)
    m_path_FB_levels[t] = max(m_FB_interp(Gt), 0)  # use implied policy level at that Gt

# year axis (annual)
start_year = 2015  # <-- change if needed
years = np.arange(start_year, start_year + T_sim)

# ----- enforce year axis limits (do NOT apply to climate-cost panel) -----
xmin, xmax = years[0], years[-1]   # 2015, 2214

# shares
I_path_FB = a_path_FB + m_path_FB_levels
I_safe_FB = np.where(I_path_FB > 0, I_path_FB, np.nan)
share_a = np.where(I_path_FB > 0, a_path_FB / I_safe_FB, 0)
share_m = np.where(I_path_FB > 0, m_path_FB_levels / I_safe_FB, 0)

# ---------- colors / legend ordering (your style) ----------
stack_order  = ["Mitigation", "Adaptation"]   # bottom -> top
legend_order = ["Adaptation", "Mitigation"]   # legend order

pal = sns.color_palette("Set2", n_colors=2)
color_map = {
    "Adaptation": pal[0],  # swapped
    "Mitigation": pal[1],  # swapped
}
colors_stack = [color_map[c] for c in stack_order]


comma0 = FuncFormatter(lambda x, pos: f"{x:,.0f}")

fb_dir = os.path.join(script_dir, "figures_fb")
os.makedirs(fb_dir, exist_ok=True)

def save_fig(fig, stem):
    png = os.path.join(fb_dir, f"{stem}.png")
    pdf = os.path.join(fb_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png)
    print("Saved:", pdf)

# -------------------------------
# Panel 1) Climate cost C_FB(G)
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(G_plot, C_plot, label=r"$C_{FB}(G)$")
ax.axvline(G0, color="gray", linestyle="--", label=r"$G_0$ (Year 2015 level)")
ax.axvline(G_ss_FB, color="red", linestyle="--", label=r"$G^{FB}_{ss}$")
ax.set_xlabel(r"GHG stock $G$")
ax.set_ylabel(r"First-best climate cost, $C_{FB}(G)$")
# ax.set_title("First-best Climate Cost")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
save_fig(fig, "fb_climate_cost")
plt.show()
plt.close(fig)


# -------------------------------
# Panel 2) GHG path G_t
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
t_grid = np.arange(T_sim + 1)
ax.plot(start_year + t_grid, G_path_FB, label=r"First-best path $G_t$")
ax.axhline(G_bar, color="gray", linestyle=":", label="Baseline $G$ (pre-industrial)")
ax.axhline(G_ss_FB, color="red", linestyle="--", label=r"$G^{FB}_{ss}$")
force_x_from_2015(ax, start_year, T_sim)   # your helper
ax.set_xlim(xmin, xmax)
ax.set_xlabel("Year")
ax.set_ylabel(r"GHG stock, $G_t$")
# ax.set_title("First-best GHG stock path")
ax.grid(alpha=0.3)
ax.legend(loc="center right")
save_fig(fig, "fb_GHG_path")
plt.show()
plt.close(fig)

# -------------------------------
# Panel 3) Funding levels (stacked area)
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
series_levels = {"Adaptation": a_path_FB, "Mitigation": m_path_FB_levels}
ax.stackplot(
    years,
    [series_levels[c] for c in stack_order],
    labels=stack_order,
    colors=colors_stack,
    alpha=0.75
)
ax.set_xlabel("Year")
ax.set_ylabel(r"Funding level, $I_t, a_t, m_t$")

ax.yaxis.set_major_formatter(comma0)
# ax.set_title("First-best funding levels")
force_x_from_2015(ax, start_year, T_sim)
ax.set_xlim(xmin, xmax)

handles, labels = ax.get_legend_handles_labels()
hmap = dict(zip(labels, handles))
ax.legend([hmap[k] for k in legend_order], legend_order, title="Funding Type", loc="upper left")
ax.grid(alpha=0.3)
save_fig(fig, "fb_funding_levels_stack")
plt.show()
plt.close(fig)

# -------------------------------
# Panel 4) Funding shares (100% stacked area)
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
series_shares = {"Adaptation": share_a, "Mitigation": share_m}
ax.stackplot(
    years,
    [series_shares[c] for c in stack_order],
    labels=stack_order,
    colors=colors_stack,
    alpha=0.75
)
ax.set_xlabel("Year")
ax.set_ylabel(r"Share of yearly total funding, $a_t/I_t, m_t/I_t$")

ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
# ax.set_title("First-best funding shares")
force_x_from_2015(ax, start_year, T_sim)
ax.set_xlim(xmin, xmax)
ax.grid(alpha=0.3)

# optional legend:
# handles, labels = ax.get_legend_handles_labels()
# hmap = dict(zip(labels, handles))
# ax.legend([hmap[k] for k in legend_order], legend_order, title="Funding Type", loc="upper left")

save_fig(fig, "fb_funding_shares_stack")
plt.show()
plt.close(fig)

# Steady-state GHG stock
G_ss_FB

# %% Second-best Simulation (reduced-form)
# ============================================================

# ----------------------------
# Settings
# ----------------------------
# Enforcement probabilities
# rho_values = [round(x, 2) for x in np.arange(0.00, 1.05, 0.05)]
rho_list = [1.0, 0.95, 0.90, 0.85, 0.80]

start_year = 2015
years_t   = np.arange(start_year, start_year + T_sim)         # length T_sim
years_tp1 = np.arange(start_year, start_year + T_sim + 1)     # length T_sim+1


def force_year_axis(ax, start_year, T_sim, series_len="T"):
    if series_len == "Tp1":
        xmin, xmax = start_year, start_year + T_sim
    else:
        xmin, xmax = start_year, start_year + T_sim - 1
    ax.set_xlim(xmin, xmax)
    ticks = [t for t in ax.get_xticks() if xmin <= t <= xmax]
    if start_year not in ticks:
        ticks = [start_year] + ticks
    ax.set_xticks(sorted(set(int(t) for t in ticks)))

# ----------------------------
# Interpolants (assumed existing grids: G_vals, a_FB, m_FB, C_FB)
# ----------------------------
def a_FB_interp(G_val):
    return float(np.interp(float(np.clip(G_val, G_vals[0], G_vals[-1])), G_vals, a_FB))

def m_FB_interp(G_val):
    return float(np.interp(float(np.clip(G_val, G_vals[0], G_vals[-1])), G_vals, m_FB))

def C_FB_interp(G_val):
    return float(np.interp(float(np.clip(G_val, G_vals[0], G_vals[-1])), G_vals, C_FB))

# ----------------------------
# FB time series along FB path (compute ONCE)
# ----------------------------
a_path_FB_ts = np.empty(T_sim)
m_path_FB_ts = np.empty(T_sim)
I_path_FB_ts = np.empty(T_sim)
w_needed_FB_ts = np.empty(T_sim)  # "FB required promise": I/(1-beta)

for t in range(T_sim):
    Gt_fb = float(np.clip(G_path_FB[t], G_vals[0], G_vals[-1]))
    a_t_fb = max(a_FB_interp(Gt_fb), 0.0)
    m_t_fb = max(m_FB_interp(Gt_fb), 0.0)
    I_t_fb = a_t_fb + m_t_fb
    a_path_FB_ts[t] = a_t_fb
    m_path_FB_ts[t] = m_t_fb
    I_path_FB_ts[t] = I_t_fb
    w_needed_FB_ts[t] = I_t_fb / (1.0 - beta) if I_t_fb > 0 else 0.0

# ----------------------------
# Model primitives used by SB
# ----------------------------
def g_m_from_m(m):
    m_safe = max(float(m), 1e-12)
    return M - tau * np.log(1.0 + m_safe)

def next_G_from_m(G, m):
    return G_bar + delta * (float(G) - G_bar) + g_m_from_m(m)

def D_expect_given_a(G_val, a_val, omega=omega, A=A, psi=psi, s_vals=s_vals, pi_vals=pi_vals):
    """E[D(a;G,s)] = sum_s pi_s * omega*exp(sG) * (A - psi*log(1+a))"""
    a_safe = max(float(a_val), 1e-12)
    out = 0.0
    Gf = float(G_val)
    for s, pi in zip(s_vals, pi_vals):
        out += pi * (omega * np.exp(s * Gf) * (A - psi * np.log(1.0 + a_safe)))
    return float(out)

def sb_allocate_given_cap(G_t, I_cap, C_FB_grid, G_grid,
                          beta=beta, delta=delta, G_bar=G_bar, M=M, tau=tau,
                          x_grid=x_grid, a_min=1e-8):
    """
    Given (G_t, I_cap), choose (a,m) to minimize:
        E[D(a;G_t)] + beta * C_FB(G')
    s.t. a+m = I_cap, a>=a_min, m>=0.
    Uses x = log(1+m) grid; includes local quadratic refinement for smoothness.
    """
    I_cap = float(I_cap)
    if I_cap <= a_min:
        return 0.0, 0.0

    m_max = max(I_cap - a_min, 0.0)
    x_max_feas = np.log(1.0 + m_max)

    x_feas = x_grid[x_grid <= x_max_feas]
    if x_feas.size == 0:
        return float(I_cap), 0.0

    m_cand = np.exp(x_feas) - 1.0
    a_cand = np.maximum(I_cap - m_cand, a_min)

    D_cand = np.array([D_expect_given_a(G_t, a) for a in a_cand], dtype=float)

    G_next = G_bar + delta * (float(G_t) - G_bar) + (M - tau * x_feas)
    G_next = np.clip(G_next, G_grid[0], G_grid[-1])
    C_next = np.interp(G_next, G_grid, C_FB_grid)

    obj = D_cand + beta * C_next
    j = int(np.argmin(obj))

    # quadratic refinement
    if 0 < j < len(x_feas) - 1:
        x0, x1, x2 = x_feas[j-1], x_feas[j], x_feas[j+1]
        f0, f1, f2 = obj[j-1], obj[j], obj[j+1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-14:
            a_q = (x2*(f1 - f0) + x1*(f0 - f2) + x0*(f2 - f1)) / denom
            b_q = (x2**2*(f0 - f1) + x1**2*(f2 - f0) + x0**2*(f1 - f2)) / denom
            if a_q > 0:
                x_star = float(np.clip(-b_q/(2*a_q), x0, x2))
            else:
                x_star = float(x1)
        else:
            x_star = float(x1)
    else:
        x_star = float(x_feas[j])

    m_star = max(float(np.exp(x_star) - 1.0), 0.0)
    a_star = float(I_cap - m_star)
    return a_star, m_star

# ----------------------------
# SB simulator (parameterized by rho)
# ----------------------------

lv_2015 = 36/440


def simulate_second_best_one_rho(
    rho,
    theta_w_min=lv_2015,
    theta_w0=lv_2015,
    cap_frac=0.2,
    seed=1,
    # promise dynamics params
    k = 1.75,
    g_floor = 0.02,
    phi_w_normal=0.05,
    phi_w_cool=0.01,
    cooldown_len=0,
    gw_max=0.02,
    u_low=0.50,
    u_high=1.00,
    eta_drop=0.50
):
    # FB promise at G0
    I_FB_0 = a_FB_interp(G0) + m_FB_interp(G0)
    w_FB_0 = I_FB_0 / (1.0 - beta) if I_FB_0 > 0 else 0.0
    
    w_min = theta_w_min * w_FB_0 * (rho**k)   # collapses floor when rho is low
    w0    = theta_w0 * w_FB_0

    rng = np.random.default_rng(seed)

    G_path_SB = np.zeros(T_sim + 1)
    w_path_SB = np.zeros(T_sim + 1)
    a_path_SB = np.zeros(T_sim)
    m_path_SB = np.zeros(T_sim)
    I_path_SB = np.zeros(T_sim)

    G_path_SB[0] = float(G0)
    w_path_SB[0] = float(w0)

    reneg_times = []
    cooldown = 0

    rho_is_one = (rho >= 1.0 - 1e-12)

    for t in range(T_sim):
        G_t = float(np.clip(G_path_SB[t], G_vals[0], G_vals[-1]))
        w_t = float(w_path_SB[t])

        # FB at current G
        a_fb_t = a_FB_interp(G_t)
        m_fb_t = m_FB_interp(G_t)
        I_fb_t = a_fb_t + m_fb_t
        w_needed_t = I_fb_t / (1.0 - beta) if I_fb_t > 0 else 0.0

        # floor only when rho<1, and also cap it relative to w_needed_t
        if (not rho_is_one) and (rho > 0.0):
            w_min_t_raw = w_min * ((1.0 + g_floor) ** t)
            w_min_t = min(w_min_t_raw, cap_frac * w_needed_t)
            w_t = max(w_t, w_min_t)

        # funding under promise
        if (w_t >= w_needed_t) and (I_fb_t > 0):
            a_t, m_t, I_t = a_fb_t, m_fb_t, I_fb_t
        else:
            I_t = (1.0 - beta) * w_t
            if I_t <= 1e-12:
                a_t, m_t = 0.0, 0.0
            else:
                a_t, m_t = sb_allocate_given_cap(G_t, I_t, C_FB, G_vals)

        a_path_SB[t] = a_t
        m_path_SB[t] = m_t
        I_path_SB[t] = I_t

        # G transition
        G_next = next_G_from_m(G_t, m_t)
        G_next = float(np.clip(G_next, G_vals[0], G_vals[-1]))
        G_path_SB[t + 1] = G_next

        # FB-needed at next state
        a_fb_next = a_FB_interp(G_next)
        m_fb_next = m_FB_interp(G_next)
        I_fb_next = a_fb_next + m_fb_next
        w_needed_next = I_fb_next / (1.0 - beta) if I_fb_next > 0 else 0.0

        # PROMISE UPDATE
        if rho_is_one:
            # If theta_w0==1, SB should track FB exactly (your intended convention)
            if abs(theta_w0 - 1.0) < 1e-12:
                w_path_SB[t + 1] = w_needed_next
            else:
                # no reneg, no floor; gradual rebuilding only
                if w_needed_next > w_t:
                    w_proposed = w_t + phi_w_normal * (w_needed_next - w_t)
                else:
                    w_proposed = w_t
                w_next = min(w_proposed, w_t * (1.0 + gw_max))
                w_path_SB[t + 1] = w_next
            continue

        # rho < 1
        w_min_tp1 = 0.0
        if rho > 0.0:
            w_min_tp1_raw = w_min * ((1.0 + g_floor) ** (t + 1))
            w_min_tp1 = min(w_min_tp1_raw, cap_frac * w_needed_next)

        if rng.random() > rho:
            u = rng.uniform(u_low, u_high)
            w_drop_target = max(w_min_tp1, u * w_t)
            w_next = (1.0 - eta_drop) * w_t + eta_drop * w_drop_target
            reneg_times.append(t + 1)
            cooldown = cooldown_len
        else:
            phi = phi_w_cool if cooldown > 0 else phi_w_normal
            if w_needed_next > w_t:
                w_proposed = w_t + phi * (w_needed_next - w_t)
            else:
                w_proposed = w_t
            w_next = min(w_proposed, w_t * (1.0 + gw_max))
            w_next = max(w_min_tp1, w_next)
            if cooldown > 0:
                cooldown -= 1

        w_path_SB[t + 1] = w_next

    # === continuation-cost difference dC_total ===
    flow_FB = np.empty(T_sim)
    flow_SB = np.empty(T_sim)

    for t in range(T_sim):
        # FB flow (along FB path)
        Gt_fb = float(np.clip(G_path_FB[t], G_vals[0], G_vals[-1]))
        a_fb  = float(a_path_FB_ts[t])
        m_fb  = float(m_path_FB_ts[t])
        flow_FB[t] = (a_fb + m_fb) + D_expect_given_a(Gt_fb, a_fb)

        # SB flow (along SB path)
        Gt_sb = float(np.clip(G_path_SB[t], G_vals[0], G_vals[-1]))
        a_sb  = float(a_path_SB[t])
        m_sb  = float(m_path_SB[t])
        flow_SB[t] = (a_sb + m_sb) + D_expect_given_a(Gt_sb, a_sb)

    Ctot_FB = np.zeros(T_sim + 1)
    Ctot_SB = np.zeros(T_sim + 1)

    # terminal values (stable apples-to-apples): FB value function on terminal G
    Ctot_FB[T_sim] = C_FB_interp(G_path_FB[T_sim])
    Ctot_SB[T_sim] = C_FB_interp(G_path_SB[T_sim])

    for t in range(T_sim - 1, -1, -1):
        Ctot_FB[t] = flow_FB[t] + beta * Ctot_FB[t + 1]
        Ctot_SB[t] = flow_SB[t] + beta * Ctot_SB[t + 1]

    dC_total = Ctot_SB[:T_sim] - Ctot_FB[:T_sim]
    dG = G_path_SB[:T_sim + 1] - G_path_FB[:T_sim + 1]

    return {
        "rho": rho,
        "reneg_times": reneg_times,
        "dC_total": dC_total,
        "dG": dG,
        "w_path_SB": w_path_SB,
        "I_path_SB": I_path_SB,
        "a_path_SB": a_path_SB,
        "m_path_SB": m_path_SB,
        "G_path_SB": G_path_SB,
    }

# ----------------------------
# Run sb -> store in DataFrames (LONG format)
# ----------------------------
dfs = {
    "dC_total": [],
    "dG": [],
    "w": [],
    "I": [],
    "a": [],
    "m": [],
}

results_by_rho = {}

for r in rho_list:
    out = simulate_second_best_one_rho(
        rho=r,
        seed=1
    )
    results_by_rho[r] = out

    dfs["dC_total"].append(pd.DataFrame({"year": years_t,   "rho": r, "value": out["dC_total"]}))
    dfs["dG"].append(      pd.DataFrame({"year": years_tp1, "rho": r, "value": out["dG"]}))
    dfs["w"].append(       pd.DataFrame({"year": years_tp1, "rho": r, "value": out["w_path_SB"]}))
    dfs["I"].append(       pd.DataFrame({"year": years_t,   "rho": r, "value": out["I_path_SB"]}))
    dfs["a"].append(       pd.DataFrame({"year": years_t,   "rho": r, "value": out["a_path_SB"]}))
    dfs["m"].append(       pd.DataFrame({"year": years_t,   "rho": r, "value": out["m_path_SB"]}))

df_dC = pd.concat(dfs["dC_total"], ignore_index=True)
df_dG = pd.concat(dfs["dG"],       ignore_index=True)
df_w  = pd.concat(dfs["w"],        ignore_index=True)
df_I  = pd.concat(dfs["I"],        ignore_index=True)
df_a  = pd.concat(dfs["a"],        ignore_index=True)
df_m  = pd.concat(dfs["m"],        ignore_index=True)

# ----------------------------
# Plot styling: colorblind-friendly + linestyles
#   - colors: matplotlib tab10 (good default, generally colorblind-friendly)
#   - linestyles: cycle distinct patterns across rho
# ----------------------------
tab10 = plt.get_cmap("tab10")
rho_to_color = {r: tab10(i) for i, r in enumerate(rho_list)}

linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
rho_to_ls = {r: linestyles[i] for i, r in enumerate(rho_list)}

def rho_label(r):
    # pretty label in legend
    if abs(r - 1.0) < 1e-12:
        return r"SB ($\rho=1$)"
    if abs(r - 0.0) < 1e-12:
        return r"SB ($\rho=0$)"
    return rf"SB ($\rho={r:.2f}$)"


sb_dir = os.path.join(script_dir, "figures_sb")
os.makedirs(sb_dir, exist_ok=True)

def save_fig(fig, stem):
    png = os.path.join(sb_dir, f"{stem}.png")
    pdf = os.path.join(sb_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print("Saved:", png)
    print("Saved:", pdf)


# ----------------------------
# 1) Total cost difference (continuation): dC_total
#    FB baseline = 0 line
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0.0, color="black", linewidth=1.0, linestyle="-", label="FB baseline (0)")
for r in rho_list:
    sub = df_dC[df_dC["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Difference in climate cost $C^{SB}_t - C^{FB}_t$")
ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb1_total_cost_diff")
plt.show()

# ----------------------------
# 2) GHG stock difference: dG
#    FB baseline = 0 line
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhline(0.0, color="black", linewidth=1.0, linestyle="-", label="FB baseline (0)")
for r in rho_list:
    sub = df_dG[df_dG["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Difference in GHG stock, $G^{SB}_t - G^{FB}_t$")
ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="Tp1")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb2_GHG_stock_diff")
plt.show()

# ----------------------------
# 3) Promised value: w_path_SB
#    Add FB "required promise" (black solid) for reference
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
#         label=r"FB required promise $w^{FB}_t = I^{FB}_t/(1-\beta)$")
ax.plot(years_t, w_needed_FB_ts, color="black", linewidth=2.5, linestyle="-", 
        label=r"FB ($w^{FB}_t = I^{FB}_t/(1-\beta)$)")
for r in rho_list:
    sub = df_w[df_w["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Promise value, $w_t$")
ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="Tp1")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb3_promised_value")
plt.show()

# ----------------------------
# 4) Total funding: FB vs SB(rho)
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years_t, I_path_FB_ts, color="black", linewidth=2.5, linestyle="-", label="FB")
for r in rho_list:
    sub = df_I[df_I["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Total funding, $I_t$")

ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb4_total_funding")
plt.show()

# ----------------------------
# 5) Adaptation funding: FB vs SB(rho)
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years_t, a_path_FB_ts, color="black", linewidth=2.5, linestyle="-", label="FB")
for r in rho_list:
    sub = df_a[df_a["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Adaptation funding, $a_t$")

ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb5_adaptation_funding")
plt.show()

# ----------------------------
# 6) Mitigation funding: FB vs SB(rho)
# ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years_t, m_path_FB_ts, color="black", linewidth=2.5, linestyle="-", label="FB")
for r in rho_list:
    sub = df_m[df_m["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))
ax.set_xlabel("Year")
ax.set_ylabel(r"Mitigation funding, $m_t$")

ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb6_mitigation_funding")
plt.show()


# ----------------------------
# 7) Adaptation share: FB vs SB(rho)
#    share_a_t = a_t / I_t
# ----------------------------

# FB share (safe divide)
share_a_FB_ts = np.divide(
    a_path_FB_ts, I_path_FB_ts,
    out=np.zeros_like(a_path_FB_ts, dtype=float),
    where=(I_path_FB_ts != 0)
)

# SB share long DF (merge then safe divide)
df_share_a = df_a.merge(df_I, on=["rho", "year"], suffixes=("_a", "_I"))
df_share_a["value"] = np.divide(
    df_share_a["value_a"].to_numpy(dtype=float),
    df_share_a["value_I"].to_numpy(dtype=float),
    out=np.full(len(df_share_a), np.nan, dtype=float),
    where=(df_share_a["value_I"].to_numpy(dtype=float) != 0)
)
df_share_a = df_share_a[["rho", "year", "value"]]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years_t, share_a_FB_ts, color="black", linewidth=2.5, linestyle="-", label="FB")
for r in rho_list:
    sub = df_share_a[df_share_a["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))

ax.set_xlabel("Year")
ax.set_ylabel(r"Adaptation share, $a_t / I_t$")
ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.set_ylim(0.0, 1.0)
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb7_adaptation_share")
plt.show()


# ----------------------------
# 8) Mitigation share: FB vs SB(rho)
#    share_m_t = m_t / I_t
# ----------------------------

# FB share (safe divide)
share_m_FB_ts = np.divide(
    m_path_FB_ts, I_path_FB_ts,
    out=np.zeros_like(m_path_FB_ts, dtype=float),
    where=(I_path_FB_ts != 0)
)

# SB share long DF (merge then safe divide)
df_share_m = df_m.merge(df_I, on=["rho", "year"], suffixes=("_m", "_I"))
df_share_m["value"] = np.divide(
    df_share_m["value_m"].to_numpy(dtype=float),
    df_share_m["value_I"].to_numpy(dtype=float),
    out=np.full(len(df_share_m), np.nan, dtype=float),
    where=(df_share_m["value_I"].to_numpy(dtype=float) != 0)
)
df_share_m = df_share_m[["rho", "year", "value"]]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(years_t, share_m_FB_ts, color="black", linewidth=2.5, linestyle="-", label="FB")
for r in rho_list:
    sub = df_share_m[df_share_m["rho"] == r]
    ax.plot(sub["year"], sub["value"], color=rho_to_color[r], linestyle=rho_to_ls[r], linewidth=2.0, label=rho_label(r))

ax.set_xlabel("Year")
ax.set_ylabel(r"Mitigation share, $m_t / I_t$")
ax.grid(alpha=0.3)
force_year_axis(ax, start_year, T_sim, series_len="T")
ax.set_ylim(0.0, 1.0)
ax.legend(loc="upper left", ncol=1)
fig.tight_layout()
save_fig(fig, "sb8_mitigation_share")
plt.show()


# Optional: save share DataFrames

outdir = os.path.join(script_dir, "data_output_sb")
os.makedirs(outdir, exist_ok=True)
print("Saved share DataFrames to:", outdir)

df_dC.to_csv(os.path.join(outdir, "df_dC_total_long.csv"), index=False)
df_dG.to_csv(os.path.join(outdir, "df_dG_long.csv"), index=False)
df_w.to_csv( os.path.join(outdir, "df_w_long.csv"), index=False)
df_I.to_csv( os.path.join(outdir, "df_I_long.csv"), index=False)
df_a.to_csv( os.path.join(outdir, "df_a_long.csv"), index=False)
df_m.to_csv( os.path.join(outdir, "df_m_long.csv"), index=False)
df_share_a.to_csv(os.path.join(outdir, "df_share_a_long.csv"), index=False)
df_share_m.to_csv(os.path.join(outdir, "df_share_m_long.csv"), index=False)
print("Saved long DataFrames to:", outdir)

# %% 
# ============================================================
# This section is intentionally left as a blank.
# ============================================================

