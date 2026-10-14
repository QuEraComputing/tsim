## Speed (attempted shots/s; best configuration per thread setting)

| circuit | postselect | Tsim width | Clifft k_max | Tsim one lane, 1 thread | Tsim shared, 1 thread | Clifft, 1 thread | Tsim best / Clifft, 1 thread | Tsim one lane, all | Tsim shared, all | Clifft, all | Tsim best / Clifft, all |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | yes | 0 | 0 | 10,188,309 | — | 6,248,098 | 1.63x | 24,552,848 | — | 40,359,137 | 0.61x |
| coherent_d3_r1 | yes | 0 | 4 | 106,772,791 | — | 11,408,974 | 9.36x | 97,106,700 | — | 80,756,892 | 1.20x |
| coherent_d5_r1 | yes | 0 | 12 | 87,582,229 | — | 45,540 | 1923.19x | 82,992,395 | — | 425,451 | 195.07x |
| distillation | yes | 3 | 5 | 4,302,714 | 5,408,551 | 2,892,125 | 1.87x | 13,707,608 | 16,525,216 | 25,661,124 | 0.64x |
| cultivation_d3 | yes | 5 | 4 | 1,130,073 | 7,047,762 | 4,997,655 | 1.41x | 3,001,344 | 9,518,240 | 39,254,871 | 0.24x |
| coherent_d3_r3 | yes | 5 | 5 | 461,835 | 5,567,949 | 2,027,850 | 2.75x | 1,258,737 | 6,926,598 | 16,994,357 | 0.41x |
| cultivation_d5 | yes | 6 | 10 | 560,965 | 3,153,269 | 149,638 | 21.07x | 1,405,723 | 7,981,384 | 1,368,985 | 5.83x |
| coherent_d5_r5 | yes | 12 | 13 | 638 | 146,104 | 3,831 | 38.14x | 2,077 | 425,257 | 34,386 | 12.37x |
| qv_q10 | no | 10 | 10 | 1,556 | 700,218 | 30,483 | 22.97x | 4,923 | 850,719 | 186,354 | 4.57x |
| qv_q20 | no | 20 | 20 | 0 | 2 | 5 | 0.38x | 1 | 13 | 21 | 0.64x |
| symft_msc_d7 | no | 12 | 19 | 924 | 2,127 | 86 | 24.63x | 2,520 | 13,188 | 381 | 34.60x |
| symft_msc_d7_postselected | yes | 12 | 19 | 3,680 | 576,339 | 1,705 | 338.04x | 10,033 | 1,500,250 | 7,317 | 205.05x |

## Check (every run pooled; z = Tsim vs Clifft)

| circuit | Tsim acceptance | Clifft acceptance | z | Tsim obs-0 / accepted | Clifft obs-0 / accepted | z | check |
|---|---:|---:|---:|---:|---:|---:|---|
| surface_d7_r7 | 0.0879 | 0.0879 | +0.5 | 0 / 30552463 | 0 / 48357838 | +0.0 | ok |
| coherent_d3_r1 | 0.9743 | 0.9743 | +0.9 | 25 / 1987132043 | 26 / 1208874849 | -1.9 | ok |
| coherent_d5_r1 | 0.9287 | 0.9288 | -1.2 | 0 / 1584074371 | 0 / 4626053 | +0.0 | ok |
| distillation | 0.0797 | 0.0798 | -0.5 | 13453263 / 31902523 | 13012007 / 30860938 | +0.5 | ok |
| cultivation_d3 | 0.6868 | 0.6868 | -0.2 | 134 / 142584411 | 368 / 430279459 | +0.9 | ok |
| coherent_d3_r3 | 0.3732 | 0.3732 | -0.5 | 0 / 53312376 | 3 / 118468381 | -1.2 | ok |
| cultivation_d5 | 0.1440 | 0.1440 | +0.1 | 1 / 30505136 | 0 / 2726689 | +0.3 | ok |
| coherent_d5_r5 | 0.0077 | 0.0081 | -2.3 | 0 / 55946 | 0 / 3305 | +0.0 | ok |
| qv_q10 | 1.0000 | 1.0000 | +0.0 | 7997999 / 15663104 | 1940574 / 3801088 | +0.3 | ok |
| qv_q20 | 1.0000 | 1.0000 | +0.0 | 202 / 408 | 133 / 264 | -0.2 | ok |
| symft_msc_d7 | 1.0000 | 1.0000 | +0.0 | 114081 / 229376 | 2605 / 5120 | -1.6 | ok |
| symft_msc_d7_postselected | 0.0056 | 0.0059 | -1.3 | 106844 / 145683 | 380 / 543 | +1.8 | ok |

## Every measurement

| circuit | simulator | configuration | threads | batch | shots/call | compile s | warmup s | shots/s | cores busy | calls |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.74 | 0.19 | 24,552,848 | 3.1 | 937 |
| surface_d7_r7 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.73 | 0.18 | 10,188,309 | 1.0 | 389 |
| surface_d7_r7 | clifft | — | 1 | 1 | 262,144 | 0.01 | nan | 796,571 | 1.0 | 31 |
| surface_d7_r7 | clifft | — | 1 | 2,048 | 262,144 | 0.01 | nan | 6,248,098 | 1.0 | 239 |
| surface_d7_r7 | clifft | — | all | 1 | 262,144 | 0.01 | nan | 7,559,833 | 12.2 | 289 |
| surface_d7_r7 | clifft | — | all | 2,048 | 262,144 | 0.01 | nan | 40,359,137 | 8.5 | 1540 |
| coherent_d3_r1 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.02 | 0.00 | 97,106,700 | 1.2 | 3706 |
| coherent_d3_r1 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.01 | 0.00 | 106,772,791 | 1.0 | 4074 |
| coherent_d3_r1 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 3,166,055 | 1.0 | 121 |
| coherent_d3_r1 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 11,408,974 | 1.0 | 436 |
| coherent_d3_r1 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 28,687,338 | 12.2 | 1095 |
| coherent_d3_r1 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 80,756,892 | 9.3 | 3081 |
| coherent_d5_r1 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.02 | 0.00 | 82,992,395 | 1.3 | 3166 |
| coherent_d5_r1 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.02 | 0.00 | 87,582,229 | 1.0 | 3341 |
| coherent_d5_r1 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 45,540 | 1.0 | 2 |
| coherent_d5_r1 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 425,451 | 12.5 | 17 |
| distillation | tsim | one lane per shot | all | 65,536 | 262,144 | 0.10 | 0.81 | 13,707,608 | 4.5 | 523 |
| distillation | tsim | shared | all | 65,536 | 262,144 | 0.07 | 0.10 | 16,525,216 | 4.0 | 631 |
| distillation | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.09 | 0.52 | 4,302,714 | 1.0 | 165 |
| distillation | tsim | shared | 1 | 65,536 | 262,144 | 0.07 | 0.13 | 5,408,551 | 1.0 | 207 |
| distillation | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 919,222 | 1.0 | 36 |
| distillation | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 2,892,125 | 1.0 | 111 |
| distillation | clifft | — | all | 1 | 262,144 | 0.00 | nan | 9,151,164 | 12.5 | 350 |
| distillation | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 25,661,124 | 10.6 | 979 |
| cultivation_d3 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.08 | 0.46 | 3,001,344 | 4.2 | 115 |
| cultivation_d3 | tsim | shared | all | 65,536 | 262,144 | 0.06 | 0.03 | 9,518,240 | 2.2 | 364 |
| cultivation_d3 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.11 | 0.57 | 1,130,073 | 1.0 | 44 |
| cultivation_d3 | tsim | shared | 1 | 65,536 | 262,144 | 0.06 | 0.04 | 7,047,762 | 1.0 | 269 |
| cultivation_d3 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 1,770,051 | 1.0 | 68 |
| cultivation_d3 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 4,997,655 | 1.0 | 191 |
| cultivation_d3 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 16,584,278 | 12.6 | 633 |
| cultivation_d3 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 39,254,871 | 10.3 | 1498 |
| coherent_d3_r3 | tsim | one lane per shot | all | 65,536 | 262,144 | 0.05 | 0.61 | 1,258,737 | 4.4 | 49 |
| coherent_d3_r3 | tsim | shared | all | 65,536 | 262,144 | 0.03 | 0.04 | 6,926,598 | 2.0 | 265 |
| coherent_d3_r3 | tsim | one lane per shot | 1 | 65,536 | 262,144 | 0.05 | 0.92 | 461,835 | 1.0 | 18 |
| coherent_d3_r3 | tsim | shared | 1 | 65,536 | 262,144 | 0.03 | 0.05 | 5,567,949 | 1.0 | 213 |
| coherent_d3_r3 | clifft | — | 1 | 1 | 262,144 | 0.00 | nan | 1,252,960 | 1.0 | 48 |
| coherent_d3_r3 | clifft | — | 1 | 2,048 | 262,144 | 0.00 | nan | 2,027,850 | 1.0 | 78 |
| coherent_d3_r3 | clifft | — | all | 1 | 262,144 | 0.00 | nan | 11,403,439 | 12.4 | 436 |
| coherent_d3_r3 | clifft | — | all | 2,048 | 262,144 | 0.00 | nan | 16,994,357 | 11.2 | 649 |
| cultivation_d5 | tsim | one lane per shot | all | 16,384 | 65,536 | 0.68 | 0.47 | 1,405,723 | 4.1 | 215 |
| cultivation_d5 | tsim | shared | all | 16,384 | 65,536 | 0.64 | 0.02 | 5,080,142 | 2.6 | 776 |
| cultivation_d5 | tsim | shared | all | 65,536 | 262,144 | 0.64 | 0.12 | 7,981,384 | 3.3 | 305 |
| cultivation_d5 | tsim | one lane per shot | 1 | 16,384 | 65,536 | 0.70 | 0.48 | 560,965 | 1.0 | 86 |
| cultivation_d5 | tsim | shared | 1 | 16,384 | 65,536 | 0.66 | 0.03 | 2,958,615 | 1.0 | 452 |
| cultivation_d5 | tsim | shared | 1 | 65,536 | 262,144 | 0.67 | 0.16 | 3,153,269 | 1.0 | 121 |
| cultivation_d5 | clifft | — | 1 | 1 | 65,536 | 0.01 | nan | 149,638 | 1.0 | 23 |
| cultivation_d5 | clifft | — | 1 | 2,048 | 65,536 | 0.01 | nan | 78,852 | 1.0 | 13 |
| cultivation_d5 | clifft | — | all | 1 | 65,536 | 0.01 | nan | 1,368,985 | 12.4 | 209 |
| cultivation_d5 | clifft | — | all | 2,048 | 65,536 | 0.01 | nan | 281,946 | 10.9 | 44 |
| coherent_d5_r5 | tsim | one lane per shot | all | 4,096 | 16,384 | 0.51 | 8.40 | 2,077 | 5.1 | 2 |
| coherent_d5_r5 | tsim | shared | all | 4,096 | 16,384 | 0.45 | 0.25 | 70,150 | 3.6 | 43 |
| coherent_d5_r5 | tsim | shared | all | 65,536 | 262,144 | 0.45 | 0.69 | 425,257 | 4.5 | 17 |
| coherent_d5_r5 | tsim | one lane per shot | 1 | 4,096 | 16,384 | 0.50 | 26.18 | 638 | 1.0 | 1 |
| coherent_d5_r5 | tsim | shared | 1 | 4,096 | 16,384 | 0.46 | 0.40 | 43,458 | 1.0 | 27 |
| coherent_d5_r5 | tsim | shared | 1 | 65,536 | 262,144 | 0.49 | 1.86 | 146,104 | 1.0 | 6 |
| coherent_d5_r5 | clifft | — | 1 | 1 | 16,384 | 0.00 | nan | 3,831 | 1.0 | 3 |
| coherent_d5_r5 | clifft | — | all | 1 | 16,384 | 0.00 | nan | 34,386 | 12.7 | 22 |
| qv_q10 | tsim | one lane per shot | all | 16,384 | 65,536 | 0.18 | 13.62 | 4,923 | 4.8 | 1 |
| qv_q10 | tsim | shared | all | 16,384 | 65,536 | 0.13 | 0.09 | 850,719 | 2.9 | 130 |
| qv_q10 | tsim | one lane per shot | 1 | 16,384 | 65,536 | 0.15 | 42.43 | 1,556 | 1.0 | 1 |
| qv_q10 | tsim | shared | 1 | 16,384 | 65,536 | 0.13 | 0.10 | 700,218 | 1.0 | 107 |
| qv_q10 | clifft | — | 1 | 1 | 65,536 | 0.00 | nan | 21,114 | 1.0 | 4 |
| qv_q10 | clifft | — | 1 | 2,048 | 65,536 | 0.00 | nan | 30,483 | 1.0 | 5 |
| qv_q10 | clifft | — | all | 1 | 65,536 | 0.00 | nan | 186,354 | 12.4 | 29 |
| qv_q10 | clifft | — | all | 2,048 | 65,536 | 0.00 | nan | 127,673 | 11.0 | 20 |
| qv_q20 | tsim | one lane per shot | all | 8 | 8 | 1.14 | 7.38 | 1 | 4.7 | 2 |
| qv_q20 | tsim | shared | all | 128 | 128 | 1.16 | 9.57 | 13 | 8.7 | 2 |
| qv_q20 | tsim | one lane per shot | 1 | 8 | 8 | 1.13 | 23.83 | 0 | 1.0 | 1 |
| qv_q20 | tsim | shared | 1 | 128 | 128 | 1.12 | 67.11 | 2 | 1.0 | 1 |
| qv_q20 | clifft | — | 1 | 1 | 8 | 0.01 | nan | 5 | 1.0 | 7 |
| qv_q20 | clifft | — | all | 1 | 8 | 0.01 | nan | 21 | 6.2 | 26 |
| symft_msc_d7 | tsim | one lane per shot | all | 4,096 | 16,384 | 5.53 | 6.90 | 2,520 | 4.5 | 2 |
| symft_msc_d7 | tsim | shared | all | 4,096 | 16,384 | 5.50 | 1.25 | 13,188 | 7.8 | 9 |
| symft_msc_d7 | tsim | one lane per shot | 1 | 4,096 | 16,384 | 5.42 | 18.29 | 924 | 1.0 | 1 |
| symft_msc_d7 | tsim | shared | 1 | 4,096 | 16,384 | 5.42 | 7.77 | 2,127 | 1.0 | 2 |
| symft_msc_d7 | clifft | — | 1 | 1 | 512 | 0.07 | nan | 86 | 1.0 | 2 |
| symft_msc_d7 | clifft | — | all | 1 | 512 | 0.07 | nan | 381 | 11.8 | 8 |
| symft_msc_d7_postselected | tsim | one lane per shot | all | 4,096 | 16,384 | 5.48 | 1.65 | 10,033 | 4.5 | 7 |
| symft_msc_d7_postselected | tsim | shared | all | 4,096 | 16,384 | 5.50 | 0.04 | 318,869 | 3.6 | 195 |
| symft_msc_d7_postselected | tsim | shared | all | 65,536 | 262,144 | 5.48 | 0.26 | 1,500,250 | 3.6 | 58 |
| symft_msc_d7_postselected | tsim | one lane per shot | 1 | 4,096 | 16,384 | 5.42 | 4.50 | 3,680 | 1.0 | 3 |
| symft_msc_d7_postselected | tsim | shared | 1 | 4,096 | 16,384 | 5.39 | 0.11 | 157,427 | 1.0 | 97 |
| symft_msc_d7_postselected | tsim | shared | 1 | 65,536 | 262,144 | 5.42 | 0.57 | 576,339 | 1.0 | 23 |
| symft_msc_d7_postselected | clifft | — | 1 | 1 | 2,048 | 0.08 | nan | 1,705 | 1.0 | 9 |
| symft_msc_d7_postselected | clifft | — | all | 1 | 2,048 | 0.08 | nan | 7,317 | 11.3 | 36 |
