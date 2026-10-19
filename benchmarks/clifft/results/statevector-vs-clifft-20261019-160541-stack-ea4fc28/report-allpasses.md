
### 1 thread

| circuit | Tsim width | Clifft k_max default → all | Tsim one lane | Tsim shared | Clifft default | Clifft all passes | Tsim best / Clifft all |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | 0 | 0 → 0 | 10.7M | — | 6.39M | 6.18M | 1.74x |
| coherent_d3_r1 | 0 | 4 → 0 | 99.2M | — | 11.3M | 682M | 0.146x |
| coherent_d5_r1 | 0 | 12 → 0 | 80.8M | — | 43.9k | 275M | 0.293x |
| distillation | 3 | 5 → 3 | 4.24M | 5.76M | 2.8M | 3.45M | 1.67x |
| cultivation_d3 | 4 | 4 → 4 | 1.94M | 7.39M | 4.87M | 4.98M | 1.48x |
| coherent_d3_r3 | 4 | 5 → 4 | 869k | 5.59M | 2M | 5.11M | 1.1x |
| cultivation_d5 | 6 | 10 → 10 | 566k | 3.58M | 148k | 172k | 20.8x |
| coherent_d5_r5 | 11 | 13 → 13 | 1.39k | 36.9k | 3.79k | 9.55k | 3.86x |
| qv_q10 | 10 | 10 → 10 | 1.71k | 13.1M | 29.7k | 25.7k | 508x |
| qv_q20 | 20 | 20 → 20 | 0.336 | 38.4 | 4.87 | 4.44 | 8.65x |
| symft_msc_d7 | 8 | 19 → 19 | 26.7k | 56.6k | 82.8 | 200 | 284x |
| symft_msc_d7_postselected | 8 | 19 → 19 | 99.9k | 1.61M | 1.63k | 7.18k | 224x |

### all threads

| circuit | Tsim width | Clifft k_max default → all | Tsim one lane | Tsim shared | Clifft default | Clifft all passes | Tsim best / Clifft all |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface_d7_r7 | 0 | 0 → 0 | 30.6M | — | 39.2M | 37.8M | 0.809x |
| coherent_d3_r1 | 0 | 4 → 0 | 102M | — | 78.7M | 682M | 0.149x |
| coherent_d5_r1 | 0 | 12 → 0 | 89.1M | — | 416k | 447M | 0.199x |
| distillation | 3 | 5 → 3 | 28.4M | 31.4M | 25.6M | 37.1M | 0.847x |
| cultivation_d3 | 4 | 4 → 4 | 9.43M | 26.5M | 35.6M | 46.3M | 0.573x |
| coherent_d3_r3 | 4 | 5 → 4 | 4.39M | 16.7M | 16.7M | 46.7M | 0.357x |
| cultivation_d5 | 6 | 10 → 10 | 3.82M | 13.4M | 1.37M | 1.65M | 8.09x |
| coherent_d5_r5 | 11 | 13 → 13 | 8.73k | 124k | 33.8k | 88.5k | 1.4x |
| qv_q10 | 10 | 10 → 10 | 9.5k | 26M | 186k | 174k | 149x |
| qv_q20 | 20 | 20 → 20 | 1.11 | 148 | 20.4 | 17 | 8.7x |
| symft_msc_d7 | 8 | 19 → 19 | 145k | 335k | 350 | 1.38k | 243x |
| symft_msc_d7_postselected | 8 | 19 → 19 | 1.8M | 6.38M | 6.67k | 38.7k | 165x |

compile s (default → all): {'surface_d7_r7': '0.01 → 0.01', 'coherent_d3_r1': '0.00 → 0.00', 'coherent_d5_r1': '0.00 → 0.00', 'distillation': '0.00 → 0.00', 'cultivation_d3': '0.00 → 0.00', 'coherent_d3_r3': '0.00 → 0.00', 'cultivation_d5': '0.01 → 0.02', 'coherent_d5_r5': '0.00 → 0.01', 'qv_q10': '0.00 → 0.00', 'qv_q20': '0.01 → 0.02', 'symft_msc_d7': '0.07 → 0.09', 'symft_msc_d7_postselected': '0.08 → 0.10'}
