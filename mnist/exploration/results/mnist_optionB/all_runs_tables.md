| run | t_mem | test@mem | best | end test | end train | w ratio | r0.5 | r0.8 | r0.9 | half r | verdict | failure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optB_b1_s3_ek1e-05_seed0 | 3420 | 0.58 | 0.91 | 0.91 | 1.00 | 0.17 | pre | 8.1 | 27.5 | 5.0 | weak |  |
| optB_b1_s3_ek1e-05_seed1 | 3420 | 0.58 | 0.91 | 0.91 | 1.00 | 0.17 | pre | 8.2 | 27.5 | 5.3 | weak |  |
| optB_b1_s3_ek1e-05_seed2 | 2673 | 0.58 | 0.92 | 0.92 | 1.00 | 0.18 | pre | 9.2 | 29.9 | 6.0 | weak |  |
| optB_b1_s3_ek3e-05_seed0 | 3420 | 0.63 | 0.90 | 0.90 | 0.99 | 0.08 | pre | 3.0 | 12.3 | 2.7 | none | gain but no delay |
| optB_b1_s3_ek3e-05_seed1 | 3420 | 0.62 | 0.90 | 0.90 | 0.99 | 0.08 | pre | 3.4 | 33.3 | 2.7 | none | gain but no delay |
| optB_b1_s3_ek3e-05_seed2 | 2673 | 0.61 | 0.91 | 0.91 | 0.99 | 0.09 | pre | 3.7 | 13.3 | 3.0 | none | gain but no delay |
| optB_b1_s9_ek1e-05_seed0 | None | - | 0.68 | 0.68 | 0.80 | 0.13 | @126000 | never | never | - | none | never fitted (stalled) |
| optB_b1_s9_ek1e-05_seed1 | None | - | 0.68 | 0.68 | 0.81 | 0.13 | @122000 | never | never | - | none | never fitted (stalled) |
| optB_b1_s9_ek1e-05_seed2 | 100000 | 0.37 | 0.69 | 0.69 | 0.81 | 0.13 | 1.2 | never | never | 1.3 | none | gain but no delay, lost fit (train 0.81 at end) |
| optB_b1_s9_ek3e-05_seed0 | None | - | 0.55 | 0.43 | 0.50 | 0.00 | @58000 | never | never | - | none | collapsed |
| optB_b1_s9_ek3e-05_seed1 | None | - | 0.53 | 0.48 | 0.55 | 0.00 | @64000 | never | never | - | none | collapsed |
| optB_b1_s9_ek3e-05_seed2 | None | - | 0.55 | 0.49 | 0.55 | 0.00 | @58287 | never | never | - | none | collapsed |
| optB_b2_curv_s9_ek1e-05_seed0 | 23712 | 0.14 | 0.92 | 0.92 | 1.00 | 0.04 | 4.0 | 6.2 | 10.2 | 4.1 | strong |  |
| optB_b2_curv_s9_ek1e-05_seed1 | 23712 | 0.17 | 0.91 | 0.91 | 1.00 | 0.04 | 3.6 | 6.2 | 9.9 | 3.9 | strong |  |
| optB_b2_lam1.9_s9_ek1e-05_seed0 | 70000 | 0.25 | 0.74 | 0.23 | 0.27 | 0.01 | 1.6 | never | never | 1.6 | none | gain but no delay, lost fit (train 0.27 at end) |
| optB_b2_lam1.9_s9_ek1e-05_seed1 | 68468 | 0.27 | 0.75 | 0.16 | 0.18 | 0.01 | 1.6 | never | never | 1.6 | none | gain but no delay, lost fit (train 0.18 at end) |
| optB_b2_lam1_s9_ek1e-05_seed0 | None | - | 0.68 | 0.10 | 0.11 | 0.01 | @126000 | never | never | - | none | collapsed |
| optB_b2_lam1_s9_ek1e-05_seed1 | None | - | 0.68 | 0.10 | 0.11 | 0.01 | @124000 | never | never | - | none | collapsed |
| optB_b2_norm_s9_ek1e-05_seed0 | 52000 | 0.20 | 0.91 | 0.91 | 1.00 | 0.04 | 1.8 | 2.7 | 4.7 | 1.9 | none | gain but no delay |
| optB_b2_norm_s9_ek1e-05_seed1 | 52000 | 0.23 | 0.92 | 0.92 | 1.00 | 0.04 | 1.7 | 2.6 | 4.3 | 1.9 | none | gain but no delay |
| optB_b3_curv_s3_ek1e-05_seed0 | 3473 | 0.58 | 0.92 | 0.92 | 1.00 | 0.15 | pre | 8.6 | 34.0 | 4.6 | weak |  |
| optB_b3_curv_s3_ek1e-05_seed1 | 3473 | 0.58 | 0.91 | 0.91 | 1.00 | 0.15 | pre | 9.2 | 34.6 | 4.6 | weak |  |
| optB_b3_curv_s3_ek2e-05_seed0 | 3058 | 0.59 | 0.92 | 0.92 | 1.00 | 0.13 | pre | 4.6 | 18.3 | 3.6 | weak |  |
| optB_b3_curv_s3_ek2e-05_seed1 | 3473 | 0.61 | 0.91 | 0.91 | 1.00 | 0.13 | pre | 4.0 | 17.3 | 3.1 | weak |  |
| optB_b3_curv_s3_ek3e-06_seed0 | 3058 | 0.55 | 0.89 | 0.89 | 1.00 | 0.43 | pre | 34.0 | never | 13.1 | weak |  |
| optB_b3_curv_s3_ek3e-06_seed1 | 3473 | 0.57 | 0.89 | 0.89 | 1.00 | 0.43 | pre | 31.2 | never | 12.8 | weak |  |
| optB_b3_curv_s5_ek1e-05_seed0 | 12000 | 0.30 | 0.91 | 0.91 | 1.00 | 0.10 | 2.8 | 7.2 | 15.0 | 4.0 | weak |  |
| optB_b3_curv_s5_ek1e-05_seed1 | 12000 | 0.34 | 0.91 | 0.91 | 1.00 | 0.10 | 2.3 | 7.2 | 14.3 | 3.8 | weak |  |
| optB_b3_curv_s5_ek2e-05_seed0 | 10932 | 0.35 | 0.92 | 0.92 | 1.00 | 0.08 | 1.6 | 3.8 | 8.0 | 2.4 | none | gain but no delay |
| optB_b3_curv_s5_ek2e-05_seed1 | 10932 | 0.39 | 0.92 | 0.92 | 1.00 | 0.08 | 1.5 | 3.8 | 7.9 | 2.4 | none | gain but no delay |
| optB_b3_curv_s5_ek3e-06_seed0 | 12000 | 0.25 | 0.81 | 0.81 | 1.00 | 0.41 | 9.3 | 24.3 | never | 10.5 | strong |  |
| optB_b3_curv_s5_ek3e-06_seed1 | 12000 | 0.29 | 0.80 | 0.80 | 1.00 | 0.41 | 7.8 | 24.7 | never | 9.7 | strong |  |
| optB_b3_curv_s5_ek3e-06_seed2 | 10932 | 0.26 | 0.81 | 0.81 | 1.00 | 0.41 | 9.9 | 26.7 | never | 11.2 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed0 | 18000 | 0.18 | 0.91 | 0.91 | 1.00 | 0.06 | 3.8 | 6.7 | 12.0 | 4.1 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed1 | 15932 | 0.20 | 0.92 | 0.92 | 1.00 | 0.06 | 3.9 | 7.5 | 13.4 | 4.4 | strong |  |
| optB_b3_curv_s7_ek1e-05_seed2 | 13954 | 0.18 | 0.92 | 0.92 | 1.00 | 0.06 | 4.9 | 8.7 | 14.3 | 5.3 | strong |  |
| optB_b3_curv_s7_ek2e-05_seed0 | 15932 | 0.23 | 0.92 | 0.92 | 1.00 | 0.05 | 2.1 | 3.8 | 6.5 | 2.4 | none | gain but no delay |
| optB_b3_curv_s7_ek2e-05_seed1 | 13954 | 0.24 | 0.92 | 0.92 | 1.00 | 0.05 | 2.2 | 4.3 | 7.5 | 2.6 | none | gain but no delay |
| optB_b3_curv_s7_ek3e-06_seed0 | 18000 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 12.7 | 22.7 | never | 12.8 | strong |  |
| optB_b3_curv_s7_ek3e-06_seed1 | 18000 | 0.18 | 0.85 | 0.85 | 1.00 | 0.23 | 11.4 | 22.7 | never | 11.8 | strong |  |
| optB_b3_curv_s7_ek3e-06_seed2 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.1 | 25.9 | never | 14.3 | strong |  |
| optB_b3_curv_s9_ek1e-05_seed2 | 20000 | 0.14 | 0.92 | 0.92 | 1.00 | 0.04 | 4.6 | 7.4 | 11.2 | 4.8 | strong |  |
| optB_b3_curv_s9_ek2e-05_seed0 | 20000 | 0.17 | 0.92 | 0.92 | 1.00 | 0.04 | 2.4 | 3.6 | 5.6 | 2.5 | none | gain but no delay |
| optB_b3_curv_s9_ek2e-05_seed1 | 20000 | 0.20 | 0.92 | 0.92 | 1.00 | 0.04 | 2.2 | 3.6 | 5.6 | 2.4 | none | gain but no delay |
| optB_b3_curv_s9_ek3e-06_seed0 | 27073 | 0.12 | 0.81 | 0.81 | 1.00 | 0.23 | 11.5 | 18.2 | never | 10.9 | strong |  |
| optB_b3_curv_s9_ek3e-06_seed1 | 27073 | 0.14 | 0.81 | 0.81 | 1.00 | 0.23 | 10.8 | 18.1 | never | 10.3 | strong |  |
| optB_b3_curv_s9_ek3e-06_seed2 | 22000 | 0.13 | 0.81 | 0.81 | 1.00 | 0.23 | 14.0 | 22.5 | never | 13.4 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed0 | 18190 | 0.15 | 0.86 | 0.86 | 1.00 | 0.23 | 12.4 | 22.3 | never | 12.5 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed1 | 18000 | 0.17 | 0.85 | 0.85 | 1.00 | 0.23 | 11.6 | 22.8 | never | 11.9 | strong |  |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed2 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.2 | 26.0 | never | 14.3 | strong |  |
| optB_b4_curv_s7_ek3e-06_n2000_seed0 | 74000 | 0.74 | 0.90 | 0.90 | 1.00 | 0.23 | pre | 1.6 | never | 1.9 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n2000_seed1 | 98000 | 0.77 | 0.89 | 0.89 | 1.00 | 0.23 | pre | 1.2 | never | 1.6 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n2000_seed2 | 78000 | 0.76 | 0.90 | 0.90 | 1.00 | 0.23 | pre | 1.4 | never | 2.9 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_n500_seed0 | 12000 | 0.12 | 0.83 | 0.83 | 1.00 | 0.23 | 25.2 | 37.8 | never | 24.3 | strong |  |
| optB_b4_curv_s7_ek3e-06_n500_seed1 | 12000 | 0.14 | 0.83 | 0.83 | 1.00 | 0.23 | 24.3 | 38.5 | never | 23.7 | strong |  |
| optB_b4_curv_s7_ek3e-06_n500_seed2 | 12000 | 0.13 | 0.84 | 0.84 | 1.00 | 0.23 | 24.7 | 37.7 | never | 24.2 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed3 | 15932 | 0.17 | 0.85 | 0.85 | 1.00 | 0.23 | 14.3 | 26.4 | never | 14.6 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed4 | 13954 | 0.16 | 0.85 | 0.85 | 1.00 | 0.23 | 16.6 | 31.0 | never | 16.8 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed5 | 18000 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 12.5 | 22.2 | never | 12.8 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed6 | 18000 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 12.9 | 22.3 | never | 13.1 | strong |  |
| optB_b4_curv_s7_ek3e-06_seed7 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.3 | 25.4 | never | 14.6 | strong |  |
| optB_b4_curv_s7_ek3e-06_w100_seed0 | 256000 | 0.48 | 0.60 | 0.60 | 1.00 | 0.29 | 1.2 | never | never | 1.5 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_w100_seed1 | 142000 | 0.39 | 0.62 | 0.62 | 1.00 | 0.28 | 2.4 | never | never | 2.4 | none | fitted, little gain after memorisation |
| optB_b4_curv_s7_ek3e-06_w100_seed2 | 134000 | 0.27 | 0.57 | 0.57 | 1.00 | 0.30 | 3.2 | never | never | 2.6 | none | gain but no delay |
| optB_b5_bias_curv_s7_ek3e-06_seed0 | 18000 | 0.15 | 0.86 | 0.86 | 1.00 | 0.23 | 13.0 | 23.0 | never | 13.0 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed1 | 20000 | 0.18 | 0.86 | 0.86 | 1.00 | 0.23 | 10.4 | 20.2 | never | 10.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed2 | 15932 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 14.9 | 26.5 | never | 14.9 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed3 | 18000 | 0.15 | 0.85 | 0.85 | 1.00 | 0.23 | 12.9 | 23.7 | never | 12.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed4 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.6 | 26.0 | never | 14.8 | strong |  |
| optB_b5_bias_curv_s7_ek3e-06_seed5 | 15932 | 0.16 | 0.86 | 0.86 | 1.00 | 0.23 | 14.4 | 25.7 | never | 14.7 | strong |  |
| run | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| optB_b1_s3_ek1e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.4 | 5.0 | 8.1 | 13.3 | 27.5 |
| optB_b1_s3_ek1e-05_seed1 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.4 | 5.3 | 8.2 | 13.3 | 27.5 |
| optB_b1_s3_ek1e-05_seed2 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.7 | 6.0 | 9.2 | 15.7 | 29.9 |
| optB_b1_s3_ek3e-05_seed0 | pre | pre | pre | pre | pre | pre | pre | 1.2 | 1.6 | 2.3 | 3.0 | 4.7 | 12.3 |
| optB_b1_s3_ek3e-05_seed1 | pre | pre | pre | pre | pre | pre | pre | 1.3 | 1.8 | 2.3 | 3.4 | 5.3 | 33.3 |
| optB_b1_s3_ek3e-05_seed2 | pre | pre | pre | pre | pre | pre | pre | 1.4 | 1.9 | 2.7 | 3.7 | 6.0 | 13.3 |
| optB_b1_s9_ek1e-05_seed0 | @88000 | @96000 | @104000 | @116000 | @126000 | @136000 | @148000 | @164000 | never | never | never | never | never |
| optB_b1_s9_ek1e-05_seed1 | @82000 | @92000 | @102000 | @112000 | @122000 | @136000 | @152000 | @174000 | never | never | never | never | never |
| optB_b1_s9_ek1e-05_seed2 | pre | pre | 1.1 | 1.1 | 1.2 | 1.3 | 1.5 | 1.6 | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed0 | @35595 | @40000 | @45549 | @51526 | @58000 | @70000 | never | never | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed1 | @35595 | @40000 | @45549 | @52000 | @64000 | never | never | never | never | never | never | never | never |
| optB_b1_s9_ek3e-05_seed2 | @35595 | @40000 | @45549 | @51526 | @58287 | @74000 | never | never | never | never | never | never | never |
| optB_b2_curv_s9_ek1e-05_seed0 | 2.9 | 3.1 | 3.4 | 3.7 | 4.0 | 4.2 | 4.5 | 4.8 | 5.1 | 5.6 | 6.2 | 7.3 | 10.2 |
| optB_b2_curv_s9_ek1e-05_seed1 | 2.4 | 2.8 | 3.0 | 3.4 | 3.6 | 4.0 | 4.3 | 4.6 | 5.0 | 5.5 | 6.2 | 7.2 | 9.9 |
| optB_b2_lam1.9_s9_ek1e-05_seed0 | 1.1 | 1.3 | 1.4 | 1.5 | 1.6 | 1.7 | 1.9 | 2.0 | 2.2 | never | never | never | never |
| optB_b2_lam1.9_s9_ek1e-05_seed1 | 1.1 | 1.2 | 1.3 | 1.5 | 1.6 | 1.7 | 1.9 | 2.1 | 2.3 | never | never | never | never |
| optB_b2_lam1_s9_ek1e-05_seed0 | @88000 | @96000 | @104000 | @116000 | @126000 | @136000 | @146000 | @164000 | never | never | never | never | never |
| optB_b2_lam1_s9_ek1e-05_seed1 | @82000 | @92000 | @101903 | @112000 | @124000 | @136000 | @151664 | @173160 | never | never | never | never | never |
| optB_b2_norm_s9_ek1e-05_seed0 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.0 | 2.2 | 2.3 | 2.5 | 2.7 | 3.2 | 4.7 |
| optB_b2_norm_s9_ek1e-05_seed1 | 1.2 | 1.4 | 1.5 | 1.6 | 1.7 | 1.8 | 1.9 | 2.1 | 2.2 | 2.4 | 2.6 | 3.0 | 4.3 |
| optB_b3_curv_s3_ek1e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.1 | 5.2 | 8.6 | 15.0 | 34.0 |
| optB_b3_curv_s3_ek1e-05_seed1 | pre | pre | pre | pre | pre | pre | 1.3 | 2.1 | 3.1 | 4.6 | 9.2 | 15.0 | 34.6 |
| optB_b3_curv_s3_ek2e-05_seed0 | pre | pre | pre | pre | pre | pre | 1.1 | 1.7 | 2.4 | 3.3 | 4.6 | 7.8 | 18.3 |
| optB_b3_curv_s3_ek2e-05_seed1 | pre | pre | pre | pre | pre | pre | pre | 1.5 | 2.3 | 3.1 | 4.0 | 7.5 | 17.3 |
| optB_b3_curv_s3_ek3e-06_seed0 | pre | pre | pre | pre | pre | pre | 1.9 | 3.6 | 8.5 | 20.3 | 34.0 | 60.2 | never |
| optB_b3_curv_s3_ek3e-06_seed1 | pre | pre | pre | pre | pre | pre | 1.7 | 3.1 | 6.8 | 18.4 | 31.2 | 53.6 | never |
| optB_b3_curv_s5_ek1e-05_seed0 | pre | 1.5 | 1.8 | 2.3 | 2.8 | 3.3 | 4.0 | 4.5 | 5.2 | 6.0 | 7.2 | 9.0 | 15.0 |
| optB_b3_curv_s5_ek1e-05_seed1 | pre | 1.0 | 1.3 | 1.8 | 2.3 | 2.8 | 3.5 | 4.2 | 5.0 | 6.0 | 7.2 | 9.2 | 14.3 |
| optB_b3_curv_s5_ek2e-05_seed0 | pre | 1.1 | 1.3 | 1.5 | 1.6 | 1.8 | 2.1 | 2.4 | 2.7 | 3.3 | 3.8 | 4.9 | 8.0 |
| optB_b3_curv_s5_ek2e-05_seed1 | pre | pre | 1.1 | 1.3 | 1.5 | 1.6 | 1.9 | 2.4 | 2.7 | 3.1 | 3.8 | 4.9 | 7.9 |
| optB_b3_curv_s5_ek3e-06_seed0 | 2.0 | 3.8 | 5.8 | 7.7 | 9.3 | 11.2 | 13.2 | 15.0 | 17.5 | 20.5 | 24.3 | never | never |
| optB_b3_curv_s5_ek3e-06_seed1 | 1.2 | 2.0 | 3.8 | 5.8 | 7.8 | 9.8 | 12.0 | 14.5 | 17.2 | 20.5 | 24.7 | never | never |
| optB_b3_curv_s5_ek3e-06_seed2 | 1.9 | 3.8 | 6.0 | 8.0 | 9.9 | 11.9 | 14.1 | 16.5 | 19.0 | 22.3 | 26.7 | never | never |
| optB_b3_curv_s7_ek1e-05_seed0 | 2.3 | 2.8 | 3.1 | 3.4 | 3.8 | 4.1 | 4.6 | 4.9 | 5.4 | 5.9 | 6.7 | 8.1 | 12.0 |
| optB_b3_curv_s7_ek1e-05_seed1 | 2.1 | 2.6 | 3.0 | 3.4 | 3.9 | 4.3 | 4.8 | 5.3 | 5.9 | 6.7 | 7.5 | 9.2 | 13.4 |
| optB_b3_curv_s7_ek1e-05_seed2 | 2.9 | 3.4 | 3.9 | 4.4 | 4.9 | 5.3 | 5.7 | 6.3 | 6.9 | 7.7 | 8.7 | 10.5 | 14.3 |
| optB_b3_curv_s7_ek2e-05_seed0 | 1.4 | 1.6 | 1.8 | 1.9 | 2.1 | 2.4 | 2.5 | 2.8 | 3.0 | 3.3 | 3.8 | 4.5 | 6.5 |
| optB_b3_curv_s7_ek2e-05_seed1 | 1.3 | 1.6 | 1.9 | 2.0 | 2.2 | 2.4 | 2.7 | 3.0 | 3.3 | 3.7 | 4.3 | 5.2 | 7.5 |
| optB_b3_curv_s7_ek3e-06_seed0 | 7.6 | 9.0 | 10.2 | 11.4 | 12.7 | 13.9 | 15.2 | 16.4 | 18.1 | 20.1 | 22.7 | 27.3 | never |
| optB_b3_curv_s7_ek3e-06_seed1 | 6.0 | 7.3 | 8.7 | 10.1 | 11.4 | 12.8 | 14.2 | 15.8 | 17.6 | 19.9 | 22.7 | 27.6 | never |
| optB_b3_curv_s7_ek3e-06_seed2 | 8.3 | 9.9 | 11.4 | 12.8 | 14.1 | 15.3 | 16.8 | 18.5 | 20.3 | 22.6 | 25.9 | 30.3 | never |
| optB_b3_curv_s9_ek1e-05_seed2 | 3.3 | 3.7 | 4.0 | 4.3 | 4.6 | 4.9 | 5.3 | 5.6 | 6.1 | 6.6 | 7.4 | 8.4 | 11.2 |
| optB_b3_curv_s9_ek2e-05_seed0 | 1.7 | 1.9 | 2.0 | 2.2 | 2.4 | 2.5 | 2.6 | 2.8 | 3.0 | 3.3 | 3.6 | 4.2 | 5.6 |
| optB_b3_curv_s9_ek2e-05_seed1 | 1.5 | 1.7 | 1.9 | 2.0 | 2.2 | 2.4 | 2.5 | 2.7 | 2.9 | 3.2 | 3.6 | 4.1 | 5.6 |
| optB_b3_curv_s9_ek3e-06_seed0 | 8.2 | 9.0 | 9.8 | 10.7 | 11.5 | 12.3 | 13.2 | 14.1 | 15.1 | 16.4 | 18.2 | never | never |
| optB_b3_curv_s9_ek3e-06_seed1 | 7.0 | 8.1 | 9.0 | 9.8 | 10.8 | 11.6 | 12.6 | 13.7 | 14.9 | 16.3 | 18.1 | never | never |
| optB_b3_curv_s9_ek3e-06_seed2 | 9.8 | 11.0 | 12.0 | 13.0 | 14.0 | 14.9 | 16.0 | 17.2 | 18.6 | 20.3 | 22.5 | never | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed0 | 7.5 | 8.9 | 10.1 | 11.3 | 12.4 | 13.6 | 15.0 | 16.4 | 17.9 | 19.8 | 22.3 | 26.7 | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed1 | 6.0 | 7.4 | 8.9 | 10.2 | 11.6 | 12.9 | 14.3 | 15.9 | 17.7 | 20.0 | 22.8 | 27.7 | never |
| optB_b4_curv_s7_ek3e-06_fullbatch_seed2 | 8.3 | 9.9 | 11.5 | 12.8 | 14.2 | 15.4 | 16.9 | 18.5 | 20.5 | 22.8 | 26.0 | 30.6 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed0 | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.1 | 1.6 | 2.8 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed1 | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.2 | 1.9 | never |
| optB_b4_curv_s7_ek3e-06_n2000_seed2 | pre | pre | pre | pre | pre | pre | pre | pre | pre | pre | 1.4 | 3.8 | never |
| optB_b4_curv_s7_ek3e-06_n500_seed0 | 18.2 | 20.2 | 21.8 | 23.5 | 25.2 | 26.7 | 28.2 | 29.8 | 31.8 | 34.3 | 37.8 | never | never |
| optB_b4_curv_s7_ek3e-06_n500_seed1 | 17.2 | 19.2 | 21.0 | 22.7 | 24.3 | 26.0 | 27.5 | 29.5 | 31.8 | 34.7 | 38.5 | never | never |
| optB_b4_curv_s7_ek3e-06_n500_seed2 | 17.8 | 19.8 | 21.7 | 23.2 | 24.7 | 26.3 | 28.0 | 29.7 | 31.7 | 34.3 | 37.7 | never | never |
| optB_b4_curv_s7_ek3e-06_seed3 | 7.9 | 9.7 | 11.2 | 12.7 | 14.3 | 15.6 | 17.2 | 19.0 | 20.7 | 23.1 | 26.4 | 31.3 | never |
| optB_b4_curv_s7_ek3e-06_seed4 | 9.5 | 11.5 | 13.2 | 14.9 | 16.6 | 18.3 | 20.2 | 22.2 | 24.5 | 27.4 | 31.0 | never | never |
| optB_b4_curv_s7_ek3e-06_seed5 | 7.4 | 8.9 | 10.2 | 11.3 | 12.5 | 13.8 | 15.0 | 16.4 | 18.0 | 19.9 | 22.2 | 26.6 | never |
| optB_b4_curv_s7_ek3e-06_seed6 | 7.8 | 9.2 | 10.4 | 11.7 | 12.9 | 14.0 | 15.3 | 16.7 | 18.2 | 19.9 | 22.3 | 26.2 | never |
| optB_b4_curv_s7_ek3e-06_seed7 | 8.9 | 10.4 | 11.8 | 13.1 | 14.3 | 15.7 | 17.1 | 18.6 | 20.3 | 22.7 | 25.4 | 29.9 | never |
| optB_b4_curv_s7_ek3e-06_w100_seed0 | pre | pre | pre | pre | 1.2 | 1.6 | 2.0 | never | never | never | never | never | never |
| optB_b4_curv_s7_ek3e-06_w100_seed1 | pre | pre | 1.2 | 1.8 | 2.4 | 2.9 | 3.3 | never | never | never | never | never | never |
| optB_b4_curv_s7_ek3e-06_w100_seed2 | 1.4 | 2.0 | 2.4 | 2.8 | 3.2 | 3.6 | never | never | never | never | never | never | never |
| optB_b5_bias_curv_s7_ek3e-06_seed0 | 8.0 | 9.4 | 10.7 | 11.8 | 13.0 | 14.2 | 15.4 | 16.8 | 18.4 | 20.4 | 23.0 | 26.9 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed1 | 5.2 | 6.6 | 7.9 | 9.1 | 10.4 | 11.5 | 12.9 | 14.3 | 15.9 | 17.7 | 20.2 | 24.2 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed2 | 9.2 | 10.7 | 12.2 | 13.6 | 14.9 | 16.3 | 17.7 | 19.3 | 21.1 | 23.5 | 26.5 | 31.4 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed3 | 7.4 | 9.0 | 10.4 | 11.6 | 12.9 | 14.2 | 15.6 | 17.1 | 18.8 | 20.9 | 23.7 | never | never |
| optB_b5_bias_curv_s7_ek3e-06_seed4 | 8.5 | 10.2 | 11.8 | 13.2 | 14.6 | 16.1 | 17.6 | 19.2 | 21.0 | 23.2 | 26.0 | 30.5 | never |
| optB_b5_bias_curv_s7_ek3e-06_seed5 | 8.8 | 10.3 | 11.7 | 13.1 | 14.4 | 15.8 | 17.2 | 18.7 | 20.3 | 22.7 | 25.7 | 30.3 | never |
