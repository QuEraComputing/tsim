## ParamZX

This repository hosts the necessary code to demonstrate the parameterised ZX-calculus reduction (and GPU parallel evaluations) method outlined in the paper 'Fast classical simulation of quantum circuits via parametric rewriting in the ZX-calculus'.

Contained within is:
  - an updated version of the PyZX library (https://github.com/Quantomatic/pyzx) to support parameterised diagrams and reduction
  - a Jupyter notebook which demonstrates how Clifford+T circuits may be reduced into parameterised scalars (and structured to be GPU-ready)
  - CUDA code that reads these GPU-ready parameterised scalars and performs speedy evaluations upon then (and benchmarks the speed measurements)

## Attribution

Anyone is welcome to use this work, and for those who do it would be appreciated if you cite the related paper https://arxiv.org/abs/2403.06777:
<pre>
  @misc{sutcliffe2024fastclassicalsimulationquantum,
      title={Fast classical simulation of quantum circuits via parametric rewriting in the ZX-calculus}, 
      author={Matthew Sutcliffe and Aleks Kissinger},
      year={2024},
      eprint={2403.06777},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2403.06777}, 
}
</pre>
