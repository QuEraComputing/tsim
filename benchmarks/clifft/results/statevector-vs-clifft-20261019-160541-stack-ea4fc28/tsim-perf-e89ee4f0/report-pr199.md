<details open>
<summary><b>All threads · trajectory sharing on (default)</b></summary>

| circuit | Tsim width | Clifft k_max | base | this PR | Clifft | this PR / base | this PR / Clifft |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface d7r7† | 0 | 0 | 1.59M | 30.7M | 37.8M | **19×** | 0.81× |
| coherent d3r1† | 0 | 0 (4) | 93.6M | 103M | 682M | **1.1×** | 0.15× |
| coherent d5r1† | 0 | 0 (12) | 66.3M | 90.5M | 447M | **1.4×** | 0.2× |
| distillation | 3 | 3 (5) | 3.21M | 32.9M | 37.1M | **10×** | 0.89× |
| cultivation d3 | 4 | 4 | 2.24M | 30.5M | 46.3M | **14×** | 0.66× |
| coherent d3r3 | 4 | 4 (5) | 1.73M | 21.3M | 46.7M | **12×** | 0.46× |
| cultivation d5 | 6 | 10 | 1.25M | 16M | 1.65M | **13×** | **9.7×** |
| coherent d5r5 | 11 | 13 | 18.4k | 155k | 88.5k | **8.5×** | **1.8×** |
| quantum volume, 10 qubits | 10 | 10 | 837k | 93M | 174k | **111×** | **535×** |
| quantum volume, 20 qubits | 20 | 20 | 8.1 | 154 | 17 | **19×** | **9.1×** |
| SymFT d=7, no postselection | 8 | 19 | 11.2k | 475k | 1.38k | **42×** | **344×** |
| SymFT d=7, postselected | 8 | 19 | 423k | 6.99M | 38.7k | **17×** | **181×** |

</details>

<details>
<summary><b>All threads · trajectory sharing off</b></summary>

| circuit | Tsim width | Clifft k_max | base | this PR | Clifft | this PR / base | this PR / Clifft |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface d7r7 | 0 | 0 | 1.59M | 30.7M | 37.8M | **19×** | 0.81× |
| coherent d3r1 | 0 | 0 (4) | 93.6M | 103M | 682M | **1.1×** | 0.15× |
| coherent d5r1 | 0 | 0 (12) | 66.3M | 90.5M | 447M | **1.4×** | 0.2× |
| distillation | 3 | 3 (5) | 3.13M | 26.3M | 37.1M | **8.4×** | 0.71× |
| cultivation d3 | 4 | 4 | 2.06M | 10.1M | 46.3M | **4.9×** | 0.22× |
| coherent d3r3 | 4 | 4 (5) | 1.27M | 7.11M | 46.7M | **5.6×** | 0.15× |
| cultivation d5 | 6 | 10 | 733k | 4.48M | 1.65M | **6.1×** | **2.7×** |
| coherent d5r5 | 11 | 13 | 4.47k | 151k | 88.5k | **34×** | **1.7×** |
| quantum volume, 10 qubits | 10 | 10 | 5.35k | 13.7k | 174k | **2.6×** | 0.079× |
| quantum volume, 20 qubits | 20 | 20 | 0.99 | 1.8 | 17 | **1.9×** | 0.11× |
| SymFT d=7, no postselection | 8 | 19 | 53.8k | 450k | 1.38k | **8.4×** | **326×** |
| SymFT d=7, postselected | 8 | 19 | 130k | 2.97M | 38.7k | **23×** | **77×** |

</details>

<details>
<summary><b>1 thread · trajectory sharing on (default)</b></summary>

| circuit | Tsim width | Clifft k_max | base | this PR | Clifft | this PR / base | this PR / Clifft |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface d7r7† | 0 | 0 | 1.54M | 11.2M | 6.18M | **7.3×** | **1.8×** |
| coherent d3r1† | 0 | 0 (4) | 90.4M | 108M | 682M | **1.2×** | 0.16× |
| coherent d5r1† | 0 | 0 (12) | 63.9M | 79.6M | 275M | **1.2×** | 0.29× |
| distillation | 3 | 3 (5) | 3.16M | 5.98M | 3.45M | **1.9×** | **1.7×** |
| cultivation d3 | 4 | 4 | 2.01M | 8.13M | 4.98M | **4.1×** | **1.6×** |
| coherent d3r3 | 4 | 4 (5) | 1.65M | 5.91M | 5.11M | **3.6×** | **1.2×** |
| cultivation d5 | 6 | 10 | 1.09M | 3.74M | 172k | **3.4×** | **22×** |
| coherent d5r5 | 11 | 13 | 6.29k | 37.4k | 9.55k | **5.9×** | **3.9×** |
| quantum volume, 10 qubits | 10 | 10 | 853k | 16.2M | 25.7k | **19×** | **629×** |
| quantum volume, 20 qubits | 20 | 20 | 2.5 | 39 | 4.4 | **15×** | **8.8×** |
| SymFT d=7, no postselection | 8 | 19 | 8.04k | 58.1k | 200 | **7.2×** | **291×** |
| SymFT d=7, postselected | 8 | 19 | 398k | 1.7M | 7.18k | **4.3×** | **237×** |

</details>

<details>
<summary><b>1 thread · trajectory sharing off</b></summary>

| circuit | Tsim width | Clifft k_max | base | this PR | Clifft | this PR / base | this PR / Clifft |
|---|---:|---:|---:|---:|---:|---:|---:|
| surface d7r7 | 0 | 0 | 1.54M | 11.2M | 6.18M | **7.3×** | **1.8×** |
| coherent d3r1 | 0 | 0 (4) | 90.4M | 108M | 682M | **1.2×** | 0.16× |
| coherent d5r1 | 0 | 0 (12) | 63.9M | 79.6M | 275M | **1.2×** | 0.29× |
| distillation | 3 | 3 (5) | 2.8M | 5.03M | 3.45M | **1.8×** | **1.5×** |
| cultivation d3 | 4 | 4 | 1.3M | 1.63M | 4.98M | **1.3×** | 0.33× |
| coherent d3r3 | 4 | 4 (5) | 669k | 1.13M | 5.11M | **1.7×** | 0.22× |
| cultivation d5 | 6 | 10 | 432k | 788k | 172k | **1.8×** | **4.6×** |
| coherent d5r5 | 11 | 13 | 1.36k | 21.7k | 9.55k | **16×** | **2.3×** |
| quantum volume, 10 qubits | 10 | 10 | 1.66k | 1.45k | 25.7k | 0.87× | 0.056× |
| quantum volume, 20 qubits | 20 | 20 | 0.33 | 0.26 | 4.4 | 0.78× | 0.059× |
| SymFT d=7, no postselection | 8 | 19 | 23.8k | 57.2k | 200 | **2.4×** | **287×** |
| SymFT d=7, postselected | 8 | 19 | 73.1k | 1.03M | 7.18k | **14×** | **144×** |

</details>

