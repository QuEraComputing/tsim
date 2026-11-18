# Cultivation circuits with T gates

Noiseless copies of published magic state cultivation circuits, for the
integration tests in `test/integration/test_determined_literature.py`.

| file | source |
|---|---|
| `chan_d5a19[f13][_uncorrected]_T.stim` | Chan, Strikis, Sun, Cai, arXiv:2609.17706, [timchan0/cliffordea](https://github.com/timchan0/cliffordea) `sim/stim_files/` at `27385c6` (MIT). S proxies, with S/S_DAG swapped for T/T_DAG as in `circuit_text.swap_t_and_s_gates`. The `_uncorrected` ones are the inputs of the paper's d=5 correction (`sim/d5-correction.md`). |
| `clifft_d5_sproxy.stim`, `clifft_d5_T*.stim` | Clifft, arXiv:2604.27058, [unitaryfoundation/clifft-paper](https://github.com/unitaryfoundation/clifft-paper) `magic_state_cultivation/convert_s_to_t.py` (Apache-2.0). Gidney's d=5 inject+cultivate S proxy; `_T_rule1` only swaps S for T, `_T_rule12` also applies the errata flips, `_T` is the full conversion with feedforward and detector healing. |
| `wan_clifft_d3_T.stim`, `wan_soft_d5_T.stim` | Wan and Zapirain, arXiv:2609.18922, [kh428/exact_ler_for_msc](https://github.com/kh428/exact_ler_for_msc) `data/inputs/` at `332bc04` (Apache-2.0): Clifft's d=3 and SOFT's d=5 native-T circuits. |
