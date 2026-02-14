from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional, Tuple

import webrtcvad


EventType = Literal["speech_start", "speech_end", "utterance"]


@dataclass
class VADConfig:
    sample_rate: int
    frame_ms: int
    aggressiveness: int
    window_ms: int
    speech_start_ratio: float
    speech_end_ratio: float
    max_utterance_s: float
    min_utterance_s: float


class VADSegmenter:
    """
    Streaming VAD state machine:
    - Emits speech_start when speech is detected
    - Emits utterance (bytes) and speech_end when speech ends
    """
    def __init__(self, cfg: VADConfig):
        self.cfg = cfg
        self.vad = webrtcvad.Vad(cfg.aggressiveness)

        self.frame_bytes = int(cfg.sample_rate * (cfg.frame_ms / 1000.0)) * 2  # int16 => 2 bytes
        self.window_frames = max(1, int(cfg.window_ms / cfg.frame_ms))

        self.ring: Deque[Tuple[bytes, bool]] = deque(maxlen=self.window_frames)
        self.triggered: bool = False

        self._utt: bytearray = bytearray()
        self._utt_frames: int = 0

    def process_frame(self, frame_pcm16: bytes) -> List[Tuple[EventType, Optional[bytes]]]:
        if len(frame_pcm16) != self.frame_bytes:
            # ignore malformed frames
            return []

        is_voiced = self.vad.is_speech(frame_pcm16, self.cfg.sample_rate)
        events: List[Tuple[EventType, Optional[bytes]]] = []

        if not self.triggered:
            self.ring.append((frame_pcm16, is_voiced))
            voiced = sum(1 for _, v in self.ring if v)
            if len(self.ring) == self.window_frames and (voiced / len(self.ring)) >= self.cfg.speech_start_ratio:
                self.triggered = True
                events.append(("speech_start", None))

                # include buffered audio leading into speech
                for f, _ in self.ring:
                    self._utt.extend(f)
                    self._utt_frames += 1
                self.ring.clear()
        else:
            self._utt.extend(frame_pcm16)
            self._utt_frames += 1
            self.ring.append((frame_pcm16, is_voiced))

            # stop conditions
            total_s = (self._utt_frames * self.cfg.frame_ms) / 1000.0
            if total_s >= self.cfg.max_utterance_s:
                # force cut
                utt = bytes(self._utt)
                self._reset()
                events.append(("utterance", utt))
                events.append(("speech_end", None))
                return events

            unvoiced = sum(1 for _, v in self.ring if not v)
            if len(self.ring) == self.window_frames and (unvoiced / len(self.ring)) >= self.cfg.speech_end_ratio:
                utt_s = (self._utt_frames * self.cfg.frame_ms) / 1000.0
                utt = bytes(self._utt)

                self._reset()

                if utt_s >= self.cfg.min_utterance_s:
                    events.append(("utterance", utt))
                events.append(("speech_end", None))

        return events

    def _reset(self) -> None:
        self.triggered = False
        self.ring.clear()
        self._utt = bytearray()
        self._utt_frames = 0

    def reset(self) -> None:
        self._reset()
