# Decoder recipes for Tsim samples

Each function takes a `stim.DetectorErrorModel` and returns `decode(dets) -> predictions`,
where `dets` is the boolean detector array from
`sampler.sample(..., separate_observables=True)` and `predictions` is a boolean array with one
column per observable. They plug into `run_point` in `SKILL.md`.

All four were checked on a distance-3 rotated surface code memory sampled with Tsim; the
logical error rates agreed with each other within the expected decoder differences.

## PyMatching

`pip install pymatching`. Needs a graph-like model: build it with `decompose_errors=True`.

```python
import numpy as np
import pymatching


def make_pymatching(dem):
    matching = pymatching.Matching.from_detector_error_model(dem)
    return lambda dets: matching.decode_batch(dets).astype(bool)


# dem = circuit.detector_error_model(decompose_errors=True)
```

## Tesseract

`pip install tesseract-decoder`. Works with any model (`decompose_errors=False`). Decodes one
shot at a time; time it on a few thousand shots before committing to a budget.

```python
import numpy as np
from tesseract_decoder import tesseract


def make_tesseract(dem):
    decoder = tesseract.TesseractConfig(dem=dem).compile_decoder()
    return lambda dets: np.array([decoder.decode(row) for row in dets], dtype=bool)
```

## BP-LSD (ldpc)

`pip install ldpc`. Works with hyperedge models.

```python
import numpy as np
from ldpc import BpLsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices


def make_bplsd(dem, max_iter=30, lsd_order=0):
    mats = detector_error_model_to_check_matrices(dem, allow_undecomposed_hyperedges=True)
    lsd = BpLsdDecoder(
        mats.check_matrix,
        error_channel=list(mats.priors),
        max_iter=max_iter,
        bp_method="minimum_sum",
        lsd_order=lsd_order,
    )

    def decode(dets):
        return np.array(
            [(mats.observables_matrix @ lsd.decode(row)) % 2 for row in dets.astype(np.uint8)],
            dtype=bool,
        )

    return decode
```

## Relay-BP

`pip install "relay-bp[stim]"`. Uses its sinter adapter's compiled decoder directly.

```python
import numpy as np
from relay_bp.stim.sinter import SinterDecoder_RelayBP


def make_relay_bp(dem):
    compiled = SinterDecoder_RelayBP().compile_decoder_for_dem(dem=dem)

    def decode(dets):
        packed = np.packbits(dets, axis=1, bitorder="little")
        pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        return np.unpackbits(
            pred, axis=1, bitorder="little", count=dem.num_observables
        ).astype(bool)

    return decode
```

## Using a recipe

```python
import stim
import tsim

circuit = tsim.Circuit(
    str(
        stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=3,
            rounds=3,
            after_clifford_depolarization=0.003,
            before_measure_flip_probability=0.003,
        )
    )
)
decode = make_pymatching(circuit.detector_error_model(decompose_errors=True))
dets, obs = circuit.compile_detector_sampler(seed=1).sample(
    shots=20_000, separate_observables=True
)
logical_error_rate = np.any(decode(dets) != obs, axis=1).mean()
```
