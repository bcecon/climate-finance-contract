## Choe, B.-H., [Climate Finance under Limited Commitment and Renegotiations: A Dynamic Contract Approach](https://www.mdpi.com/1911-8074/19/1/76), _Journal of Risk and Financial Management_ (2026)

<br>

### First-best Benchmark

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="figures_fb/fb_climate_cost.png" width="100%">
      <br>
      <b>(a)</b> Climate cost C_FB(G)
    </td>
    <td align="center" width="50%">
      <img src="figures_fb/fb_GHG_path.png" width="100%">
      <br>
      <b>(b)</b> Dynamics of GHG stock
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <img src="figures_fb/fb_funding_levels_stack.png" width="100%">
      <br>
      <b>(c)</b> Adaptation and mitigation funding levels
    </td>
    <td align="center" width="50%">
      <img src="figures_fb/fb_funding_shares_stack.png" width="100%">
      <br>
      <b>(d)</b> Adaptation and mitigation funding shares
    </td>
  </tr>
</table>

<br>
<p align="center">
  <b>Figure 1.</b> First-best benchmarks. Author's own elaboration.
</p>

<br>

### Second-best Simulation

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="figures_sb/sb1_total_cost_diff.png" width="100%">
      <br>
      <b>(a)</b> Difference in climate cost: C(w_t, G_t) − C_FB(G_t)
    </td>
    <td align="center" width="50%">
      <img src="figures_sb/sb2_GHG_stock_diff.png" width="100%">
      <br>
      <b>(b)</b> Difference in GHG stock: G_SB,t − G_FB,t
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <img src="figures_sb/sb3_promised_value.png" width="100%">
      <br>
      <b>(c)</b> Promised contribution value: w_t
    </td>
    <td align="center" width="50%">
      <img src="figures_sb/sb4_total_funding.png" width="100%">
      <br>
      <b>(d)</b> Total funding: I_t
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <img src="figures_sb/sb5_adaptation_funding.png" width="100%">
      <br>
      <b>(e)</b> Adaptation funding: a_t
    </td>
    <td align="center" width="50%">
      <img src="figures_sb/sb6_mitigation_funding.png" width="100%">
      <br>
      <b>(f)</b> Mitigation funding: m_t
    </td>
  </tr>

  <tr>
    <td align="center" width="50%">
      <img src="figures_sb/sb7_adaptation_share.png" width="100%">
      <br>
      <b>(g)</b> Adaptation share: a_t / I_t
    </td>
    <td align="center" width="50%">
      <img src="figures_sb/sb8_mitigation_share.png" width="100%">
      <br>
      <b>(h)</b> Mitigation share: m_t / I_t
    </td>
  </tr>
</table>

<br>
<p align="center">
  <b>Figure 2.</b> Comparison of dynamics under the first-best and the second-best (SB) contract for probabilities of contract enforcement
  rho in {0.80, 0.85, 0.90, 0.95, 1.00}, with initial total funding set to 8% of the first-best level in 2015.
  Author's own elaboration.
</p>


The author does not numerically solve the full Bellman problem in this repository because a global solution is computationally intensive and highly sensitive to grid construction and interpolation choices. Nevertheless, a full-solution implementation can reproduce the patterns shown in Figure 2.

