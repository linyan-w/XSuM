# Dataset summary

- mcnu rows: 3962 (signal neutrinos); evt rows: 18650 (selected slices); hdr rows: 7500530 (triggers)
- files (file_idx, __ntuple): 565463; total POT: 4.077e+20
- reco columns modelled: 53

| nuint_categ of selected slices | rows |
|---|---|
| 5 RES | 9456 |
| -1 out of FV | 1926 |
| 6 DIS | 1702 |
| 1 signal | 1463 |
| 3 CCQE | 1227 |
| 0 NC | 1131 |
| 4 2p2h | 1121 |
| -99 cosmic / unmatched | 624 |

## Truth -> reco table

```
{
 "n_truth": 3962,
 "n_selected": 1463,
 "efficiency": 0.36925795053003535,
 "range_p_mu_over_true_q16_50_84": [
  0.4430803406238556,
  0.8925008177757263,
  1.0174343061447144
 ],
 "range_p_cpi_over_true_q16_50_84": [
  0.33317756295204165,
  0.568609356880188,
  0.8535809421539307
 ]
}
```

| split (by file_idx) | neutrinos | selected |
|---|---|---|
| train | 3169 | 1166 |
| val | 394 | 142 |
| test | 399 | 155 |

## Tier 1 columns on selected signal (mean, std, NaN fraction)

| column | mean | std | NaN |
|---|---|---|---|
| is_fv | 1 | 0 | 0.000 |
| bcfm_score | 0.6283 | 1.033 | 0.000 |
| nu_score | 0.6849 | 0.0449 | 0.000 |
| is_not_clear_cosmic | 1 | 0 | 0.000 |
| n_prong | 2 | 0 | 0.000 |
| n_trk | 2 | 0 | 0.000 |
| n_proton | 0 | 0 | 0.000 |
| n_shower | 0 | 0 | 0.000 |
| long_dirx | 0.006673 | 0.1965 | 0.000 |
| long_diry | 0.001503 | 0.1766 | 0.000 |
| long_dirz | 0.9633 | 0.04691 | 0.000 |
| short_dirx | 0.007912 | 0.3169 | 0.000 |
| short_diry | 0.001807 | 0.2834 | 0.000 |
| short_dirz | 0.8874 | 0.1778 | 0.000 |
| vtx_x | -16.52 | 106.5 | 0.000 |
| vtx_y | -16.7 | 98.62 | 0.000 |
| vtx_z | 220.9 | 126 | 0.000 |
| long_trk_len | 199.3 | 105.3 | 0.000 |
| short_trk_len | 67.82 | 41.77 | 0.000 |
| long_trk_pid | 0.03218 | 0.01911 | 0.000 |
| short_trk_pid | 0.04133 | 0.03199 | 0.000 |
| long_trk_chi2_mu | 7.894 | 4.891 | 0.000 |
| long_trk_chi2_pro | 182.2 | 47.19 | 0.000 |
| long_trk_chi2_mu_new | 7.796 | 4.661 | 0.000 |
| long_trk_chi2_pro_new | 180.4 | 47.08 | 0.000 |
| short_trk_chi2_mu | 8.859 | 5.501 | 0.000 |
| short_trk_chi2_pro | 173.9 | 51.12 | 0.000 |
| short_trk_chi2_mu_new | 8.735 | 5.263 | 0.000 |
| short_trk_chi2_pro_new | 172.8 | 50.98 | 0.000 |
| reco_t | 0.02015 | 0.02334 | 0.000 |
| opening_angle | 0.8263 | 0.1199 | 0.000 |
| is_muon_contained | 0.3308 | 0.4705 | 0.000 |
| is_cpi_contained | 0.7553 | 0.4299 | 0.000 |
| muon_start_x | -16.52 | 106.5 | 0.000 |
| muon_start_y | -16.7 | 98.62 | 0.000 |
| muon_start_z | 221.5 | 126 | 0.000 |
| muon_end_x | -15.61 | 112.4 | 0.000 |
| muon_end_y | -16.24 | 103.3 | 0.000 |
| muon_end_z | 413.3 | 111.2 | 0.000 |
| cpi_start_x | -16.51 | 106.5 | 0.000 |
| cpi_start_y | -16.68 | 98.65 | 0.000 |
| cpi_start_z | 222.2 | 126 | 0.000 |
| cpi_end_x | -16.08 | 110.3 | 0.000 |
| cpi_end_y | -16.81 | 100.4 | 0.000 |
| cpi_end_z | 281.6 | 129.8 | 0.000 |
| range_p_mu | 0.5571 | 0.2345 | 0.000 |
| range_p_cpi | 0.256 | 0.09733 | 0.000 |
| total_pe | 2.191e+04 | 1.224e+04 | 0.000 |
| sum_integ_4cm | 8000 | 2501 | 0.014 |
| sum_integ_3cm | 5831 | 2293 | 0.025 |
| sum_integ_2cm | 3731 | 1854 | 0.079 |
| sum_integ_1cm | 2040 | 1247 | 0.334 |
| sphere_3_over_4m3 | 2.813 | 1.586 | 0.025 |
