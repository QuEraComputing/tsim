## Speed (attempted shots/s; best configuration per thread setting)

| circuit | postselect | Tsim width | Clifft k_max | Tsim one lane, 1 thread | Tsim shared, 1 thread | Clifft, 1 thread | Tsim best / Clifft, 1 thread | Tsim one lane, all | Tsim shared, all | Clifft, all | Tsim best / Clifft, all |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | yes | 0 | 0 | 10,721,584 | — | 6,392,616 | 1.68x | 30,633,188 | — | 39,203,783 | 0.78x |
| coherent_d3_r1 | yes | 0 | 4 | 99,213,008 | — | 11,287,415 | 8.79x | 101,870,339 | — | 78,736,122 | 1.29x |
| coherent_d5_r1 | yes | 0 | 12 | 80,837,807 | — | 43,917 | 1840.70x | 89,059,808 | — | 415,906 | 214.13x |
| distillation | yes | 3 | 5 | 4,236,272 | 5,759,703 | 2,803,953 | 2.05x | 14,482,907 | 22,523,029 | 25,644,186 | 0.88x |
| cultivation_d3 | yes | 4 | 4 | 1,936,528 | 7,388,443 | 4,871,445 | 1.52x | 8,048,276 | 16,124,869 | 35,608,066 | 0.45x |
| coherent_d3_r3 | yes | 4 | 5 | 869,435 | 5,593,905 | 2,003,935 | 2.79x | 3,853,441 | 12,913,334 | 16,678,729 | 0.77x |
| cultivation_d5 | yes | 6 | 10 | 566,038 | 3,577,174 | 148,179 | 24.14x | 1,423,820 | 10,915,247 | 1,366,339 | 7.99x |
| coherent_d5_r5 | yes | 11 | 13 | 1,394 | 36,870 | 3,791 | 9.73x | 6,791 | 105,626 | 33,793 | 3.13x |
| qv_q10 | no | 10 | 10 | 1,709 | 13,083,756 | 29,706 | 440.45x | 8,997 | 25,225,056 | 186,433 | 135.30x |
| qv_q20 | no | 20 | 20 | 0 | 38 | 5 | 7.88x | 1 | 148 | 20 | 7.26x |
| symft_msc_d7 | no | 8 | 19 | 26,650 | 56,632 | 83 | 683.91x | 145,212 | 191,220 | 350 | 546.03x |
| symft_msc_d7_postselected | yes | 8 | 19 | 99,933 | 1,608,211 | 1,630 | 986.83x | 271,203 | 6,376,373 | 6,671 | 955.81x |

## Check (every run pooled; z = Tsim vs Clifft)

| circuit | Tsim acceptance | Clifft acceptance | z | Tsim obs-0 / accepted | Clifft obs-0 / accepted | z | check |
|---|---:|---:|---:|---:|---:|---:|---|
| surface_d7_r7 | 0.0879 | 0.0879 | +0.6 | 0 / 36358714 | 0 / 47481910 | +0.0 | ok |
| coherent_d3_r1 | 0.9743 | 0.9743 | +1.0 | 24 / 1959548421 | 23 / 1194571908 | -1.6 | ok |
| coherent_d5_r1 | 0.9287 | 0.9288 | -0.9 | 0 / 1577984148 | 0 / 4382432 | +0.0 | ok |
| distillation | 0.0797 | 0.0798 | -0.5 | 15829304 / 37525775 | 12862198 / 30505343 | +1.6 | ok |
| cultivation_d3 | 0.6867 | 0.6868 | -0.9 | 203 / 230434284 | 349 / 395174601 | -0.0 | ok |
| coherent_d3_r3 | 0.3732 | 0.3732 | -0.3 | 2 / 86867124 | 3 / 117489404 | -0.1 | ok |
| cultivation_d5 | 0.1440 | 0.1440 | -0.3 | 0 / 38256774 | 0 / 2689227 | +0.0 | ok |
| coherent_d5_r5 | 0.0077 | 0.0081 | -2.5 | 0 / 21030 | 0 / 3172 | +0.0 | ok |
| qv_q10 | 1.0000 | 1.0000 | +0.0 | 195772072 / 383385600 | 1906962 / 3735552 | +0.6 | ok |
| qv_q20 | 1.0000 | 1.0000 | +0.0 | 977 / 1944 | 133 / 264 | -0.0 | ok |
| symft_msc_d7 | 1.0000 | 1.0000 | +0.0 | 2097301 / 4227072 | 2360 / 4608 | -2.2 | ok |
| symft_msc_d7_postselected | 0.0056 | 0.0058 | -1.0 | 476685 / 650356 | 347 / 488 | +1.1 | ok |

## Every measurement

| circuit | simulator | configuration | threads | batch | shots/call | compile s | warmup s | shots/s | cores busy | calls |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.46 | 0.55 | 30,633,188 | 5.1 | 1169 |
| surface_d7_r7 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.46 | 0.39 | 10,721,584 | 1.0 | 409 |
| surface_d7_r7 | clifft | — | 1 | 1 | 262,144 | 0.01 | nan | 799,399 | 1.0 | 31 |
| surface_d7_r7 | clifft | — | 1 | 2,048 | 262,144 | 0.01 | nan | 6,392,616 | 1.0 | 244 |
| surface_d7_r7 | clifft | — | all | 1 | 262,144 | 0.01 | nan | 7,590,298 | 12.3 | 290 |
| surface_d7_r7 | clifft | — | all | 2,048 | 262,144 | 0.01 | nan | 39,203,783 | 8.3 | 1496 |
| coherent_d3_r1 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.01 | 0.00 | 101,870,339 | 1.3 | 3887 |
| coherent_d3_r1 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.02 | 0.00 | 99,213,008 | 1.0 | 3785 |
| coherent_d3_r1 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 3,135,644 | 1.0 | 120 |
| coherent_d3_r1 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 11,287,415 | 1.0 | 431 |
| coherent_d3_r1 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 29,395,063 | 12.5 | 1122 |
| coherent_d3_r1 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 78,736,122 | 9.1 | 3004 |
| coherent_d5_r1 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.02 | 0.00 | 89,059,808 | 1.6 | 3398 |
| coherent_d5_r1 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.02 | 0.00 | 80,837,807 | 1.0 | 3084 |
| coherent_d5_r1 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 43,917 | 1.0 | 2 |
| coherent_d5_r1 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 415,906 | 12.2 | 16 |
| distillation | tsim | one lane per shot | all | 65,536 | 262,144 | 0.10 | 0.51 | 14,482,907 | 4.7 | 553 |
| distillation | tsim | shared | all | 65,536 | 262,144 | 0.12 | 0.11 | 22,523,029 | 4.9 | 860 |
| distillation | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.10 | 0.87 | 4,236,272 | 1.0 | 162 |
| distillation | tsim | shared | 1 | 65,536 | 262,144 | 0.07 | 0.13 | 5,759,703 | 1.0 | 220 |
| distillation | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 888,784 | 1.0 | 34 |
| distillation | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 2,803,953 | 1.0 | 107 |
| distillation | clifft | — | all | 1 | 262,144 | 0.00 | nan | 8,863,683 | 12.3 | 339 |
| distillation | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 25,644,186 | 10.6 | 979 |
| cultivation_d3 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.09 | 0.43 | 8,048,276 | 7.6 | 308 |
| cultivation_d3 | tsim | shared | all | 65,536 | 262,144 | 0.07 | 0.28 | 16,124,869 | 2.6 | 616 |
| cultivation_d3 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.09 | 0.51 | 1,936,528 | 1.0 | 74 |
| cultivation_d3 | tsim | shared | 1 | 65,536 | 262,144 | 0.06 | 0.04 | 7,388,443 | 1.0 | 282 |
| cultivation_d3 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 1,718,622 | 1.0 | 66 |
| cultivation_d3 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 4,871,445 | 1.0 | 186 |
| cultivation_d3 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 15,301,589 | 11.8 | 584 |
| cultivation_d3 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 35,608,066 | 9.5 | 1359 |
| coherent_d3_r3 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.06 | 0.47 | 3,853,441 | 8.9 | 147 |
| coherent_d3_r3 | tsim | shared | all | 65,536 | 262,144 | 0.03 | 0.02 | 12,913,334 | 2.8 | 493 |
| coherent_d3_r3 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.06 | 0.72 | 869,435 | 1.0 | 34 |
| coherent_d3_r3 | tsim | shared | 1 | 65,536 | 262,144 | 0.03 | 0.05 | 5,593,905 | 1.0 | 214 |
| coherent_d3_r3 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 1,236,016 | 1.0 | 48 |
| coherent_d3_r3 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 2,003,935 | 1.0 | 77 |
| coherent_d3_r3 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 11,466,313 | 12.2 | 439 |
| coherent_d3_r3 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 16,678,729 | 11.0 | 637 |
| cultivation_d5 | tsim | one lane per shot | all | 16,384 | 65,536 | 0.64 | 0.46 | 1,423,820 | 3.9 | 218 |
| cultivation_d5 | tsim | shared | all | 16,384 | 65,536 | 0.58 | 0.30 | 6,681,076 | 2.6 | 1020 |
| cultivation_d5 | tsim | shared | all | 65,536 | 262,144 | 0.58 | 0.30 | 10,915,247 | 4.1 | 417 |
| cultivation_d5 | tsim | one lane per shot | 1 | 16,384 | 65,536 | 0.59 | 0.50 | 566,038 | 1.0 | 87 |
| cultivation_d5 | tsim | shared | 1 | 16,384 | 65,536 | 0.57 | 0.02 | 3,363,317 | 1.0 | 514 |
| cultivation_d5 | tsim | shared | 1 | 65,536 | 262,144 | 0.56 | 0.07 | 3,577,174 | 1.0 | 137 |
| cultivation_d5 | clifft | — | 1 | 1 | 65,536 | 0.01 | nan | 148,179 | 1.0 | 23 |
| cultivation_d5 | clifft | — | 1 | 2,048 | 65,536 | 0.01 | nan | 77,515 | 1.0 | 12 |
| cultivation_d5 | clifft | — | all | 1 | 65,536 | 0.01 | nan | 1,366,339 | 12.3 | 209 |
| cultivation_d5 | clifft | — | all | 2,048 | 65,536 | 0.01 | nan | 264,732 | 10.8 | 41 |
| coherent_d5_r5 | tsim | one lane per shot | all | 4,096 | 16,384 | 0.51 | 2.92 | 6,791 | 10.9 | 5 |
| coherent_d5_r5 | tsim | shared | all | 4,096 | 16,384 | 0.47 | 0.28 | 57,339 | 3.2 | 36 |
| coherent_d5_r5 | tsim | shared | all | 65,536 | 262,144 | 0.46 | 2.45 | 105,626 | 3.4 | 5 |
| coherent_d5_r5 | tsim | one lane per shot | 1 | 4,096 | 16,384 | 0.49 | 12.10 | 1,394 | 1.0 | 1 |
| coherent_d5_r5 | tsim | shared | 1 | 4,096 | 16,384 | 0.45 | 0.81 | 20,930 | 1.0 | 13 |
| coherent_d5_r5 | tsim | shared | 1 | 65,536 | 262,144 | 0.48 | 7.11 | 36,870 | 1.0 | 2 |
| coherent_d5_r5 | clifft | — | 1 | 1 | 16,384 | 0.00 | nan | 3,791 | 1.0 | 3 |
| coherent_d5_r5 | clifft | — | all | 1 | 16,384 | 0.00 | nan | 33,793 | 12.4 | 21 |
| qv_q10 | tsim | one lane per shot | all | 16,384 | 65,536 | 0.24 | 7.54 | 8,997 | 11.3 | 2 |
| qv_q10 | tsim | shared | all | 16,384 | 65,536 | 0.22 | 0.00 | 25,225,056 | 2.4 | 3850 |
| qv_q10 | tsim | one lane per shot | 1 | 16,384 | 65,536 | 0.23 | 38.68 | 1,709 | 1.0 | 1 |
| qv_q10 | tsim | shared | 1 | 16,384 | 65,536 | 0.22 | 0.01 | 13,083,756 | 1.0 | 1997 |
| qv_q10 | clifft | — | 1 | 1 | 65,536 | 0.00 | nan | 20,778 | 1.0 | 4 |
| qv_q10 | clifft | — | 1 | 2,048 | 65,536 | 0.00 | nan | 29,706 | 1.0 | 5 |
| qv_q10 | clifft | — | all | 1 | 65,536 | 0.00 | nan | 186,433 | 12.5 | 29 |
| qv_q10 | clifft | — | all | 2,048 | 65,536 | 0.00 | nan | 121,383 | 10.8 | 19 |
| qv_q20 | tsim | one lane per shot | all | 8 | 8 | 2.76 | 7.65 | 1 | 4.7 | 2 |
| qv_q20 | tsim | shared | all | 128 | 128 | 2.98 | 1.79 | 148 | 6.2 | 12 |
| qv_q20 | tsim | one lane per shot | 1 | 8 | 8 | 2.75 | 24.08 | 0 | 1.0 | 1 |
| qv_q20 | tsim | shared | 1 | 128 | 128 | 2.77 | 3.31 | 38 | 1.0 | 3 |
| qv_q20 | clifft | — | 1 | 1 | 8 | 0.01 | nan | 5 | 1.0 | 7 |
| qv_q20 | clifft | — | all | 1 | 8 | 0.01 | nan | 20 | 6.0 | 26 |
| symft_msc_d7 | tsim | one lane per shot | all | 4,096 | 16,384 | 4.10 | 0.54 | 145,212 | 10.5 | 89 |
| symft_msc_d7 | tsim | shared | all | 4,096 | 16,384 | 4.18 | 0.09 | 191,220 | 3.8 | 117 |
| symft_msc_d7 | tsim | one lane per shot | 1 | 4,096 | 16,384 | 4.13 | 1.04 | 26,650 | 1.0 | 17 |
| symft_msc_d7 | tsim | shared | 1 | 4,096 | 16,384 | 4.09 | 0.30 | 56,632 | 1.0 | 35 |
| symft_msc_d7 | clifft | — | 1 | 1 | 512 | 0.07 | nan | 83 | 1.0 | 2 |
| symft_msc_d7 | clifft | — | all | 1 | 512 | 0.07 | nan | 350 | 11.7 | 7 |
| symft_msc_d7_postselected | tsim | one lane per shot | all | 4,096 | 16,384 | 4.02 | 0.06 | 271,203 | 4.3 | 166 |
| symft_msc_d7_postselected | tsim | shared | all | 4,096 | 16,384 | 4.12 | 0.01 | 2,088,725 | 2.4 | 1275 |
| symft_msc_d7_postselected | tsim | shared | all | 65,536 | 262,144 | 4.19 | 0.04 | 6,376,373 | 5.6 | 244 |
| symft_msc_d7_postselected | tsim | one lane per shot | 1 | 4,096 | 16,384 | 4.08 | 0.16 | 99,933 | 1.0 | 62 |
| symft_msc_d7_postselected | tsim | shared | 1 | 4,096 | 16,384 | 4.07 | 0.01 | 1,232,431 | 1.0 | 753 |
| symft_msc_d7_postselected | tsim | shared | 1 | 65,536 | 262,144 | 4.05 | 0.16 | 1,608,211 | 1.0 | 62 |
| symft_msc_d7_postselected | clifft | — | 1 | 1 | 2,048 | 0.08 | nan | 1,630 | 1.0 | 8 |
| symft_msc_d7_postselected | clifft | — | all | 1 | 2,048 | 0.08 | nan | 6,671 | 11.1 | 33 |
