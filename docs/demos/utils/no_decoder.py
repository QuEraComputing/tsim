import numpy as np
import sinter
import stim
from sinter import CompiledDecoder


class CompiledNoDecoder(CompiledDecoder):

    def __init__(self, num_observables: int) -> None:
        self.num_observables = num_observables

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(
            (len(bit_packed_detection_event_data), (self.num_observables + 7) // 8),
            dtype=np.uint8,
        )


class NoDecoder(sinter.Decoder):
    def __init__(self) -> None:
        pass

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> sinter.CompiledDecoder:
        return CompiledNoDecoder(dem.num_observables)
