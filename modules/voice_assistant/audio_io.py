from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd


@dataclass
class AudioFrame:
    pcm16: bytes  # little-endian int16 mono


class MicStream:
    """
    Captures mic audio as int16 mono frames and pushes to a Queue.
    sounddevice on macOS can be installed via pip; PortAudio is handled automatically. 
    """
    def __init__(self, sample_rate: int, frame_samples: int):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._q: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=200)
        self._stream: Optional[sd.InputStream] = None

    @property
    def queue(self) -> "queue.Queue[AudioFrame]":
        return self._q

    def start(self) -> None:
        if self._stream is not None:
            return

        def callback(indata, frames, time, status):  # noqa: ANN001
            if status:
                # we keep it silent; audio glitches happen sometimes
                pass
            try:
                self._q.put_nowait(AudioFrame(pcm16=indata.tobytes()))
            except queue.Full:
                # Drop frames to keep latency bounded
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
