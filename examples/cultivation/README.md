# Magic state cultivation: T versus S and Z

`t_vs_s.ipynb` reproduces the "Comparing Cultivation of T|+⟩, S|+⟩, and Z|+⟩ States" figure of Gidney, Shutty and
Jones, [arXiv:2409.17595](https://arxiv.org/abs/2409.17595): distance-3 cultivation without an escape stage, with every
detector postselected (no decoder). The paper sampled its 18 points with a state-vector simulator in 7.2 CPU-years;
Tsim samples 8.1 billion shots of them with `compile_detector_sampler` and `sample`.

- `cultivation.py`: the circuits' paths, the shot budget and the figure.
- `circuits/`: the circuits, from `make_circuits.py` (which uses the authors' generator).
- `paper_data.csv`: the paper's samples of this figure, from its `stats.csv`
  ([Zenodo 10.5281/zenodo.13777072](https://doi.org/10.5281/zenodo.13777072)).

```bash
uv run --with jupyterlab jupyter lab examples/cultivation/t_vs_s.ipynb
```

On an Apple-silicon Mac, Tsim samples on the GPU.
