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
    def __init__(self, sample_rate: int, frame_samples: int, input_device: str = ""):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.input_device = input_device.strip()
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

        device = self._resolve_input_device()
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.frame_samples,
                callback=callback,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(self._format_device_error(e)) from e

        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _resolve_input_device(self) -> Optional[int]:
        if not self.input_device:
            return None

        devices = sd.query_devices()
        spec = self.input_device

        if spec.lstrip("-").isdigit():
            idx = int(spec)
            if idx < 0 or idx >= len(devices):
                raise RuntimeError(f"VA_INPUT_DEVICE={spec}: invalid device index")

            info = devices[idx]
            if int(info.get("max_input_channels", 0)) <= 0:
                raise RuntimeError(f"VA_INPUT_DEVICE={spec}: selected device has no input channels")
            return idx

        needle = spec.lower()
        matches = []
        for idx, info in enumerate(devices):
            if int(info.get("max_input_channels", 0)) <= 0:
                continue
            name = str(info.get("name", ""))
            if needle in name.lower():
                matches.append(idx)

        if not matches:
            raise RuntimeError(f"VA_INPUT_DEVICE={spec}: no input device matched by name")
        return matches[0]

    @staticmethod
    def _format_device_error(err: Exception) -> str:
        lines = [f"Не удалось открыть микрофон: {err}"]
        lines.append("Подсказка: укажи VA_INPUT_DEVICE (индекс или часть имени).")

        try:
            devices = sd.query_devices()
            input_rows = []
            for idx, info in enumerate(devices):
                if int(info.get("max_input_channels", 0)) <= 0:
                    continue
                input_rows.append(f"  - {idx}: {info.get('name', 'unknown')}")

            if input_rows:
                lines.append("Доступные input-устройства:")
                lines.extend(input_rows[:15])
            else:
                lines.append("Input-устройства не найдены.")
        except Exception:  # noqa: BLE001
            lines.append("Не удалось получить список устройств через sounddevice.")

        return "\n".join(lines)
