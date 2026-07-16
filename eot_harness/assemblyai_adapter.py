from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlencode

from .io import DEFAULT_INFERENCE_INTERVAL
from .languages import supports_language_code
from .streaming_stt import (
    DEFAULT_CHUNK_MS,
    SAMPLE_RATE,
    build_event_prediction_rows,
    chunk_size_bytes,
    import_websockets,
    prepare_pcm16_audio,
    resolve_api_key,
)

DEFAULT_ASSEMBLYAI_MODEL = "universal-3-5-pro"
DEFAULT_CONCURRENCY = 4
DEFAULT_MIN_TURN_SILENCE_MS = 100
DEFAULT_MAX_TURN_SILENCE_MS = 3000
DEFAULT_END_OF_TURN_CONFIDENCE_THRESHOLD = 0.1
# U3 Pro models use punctuation-based turn detection; `end_of_turn_confidence_threshold`
# is not part of their API and only applies to the older universal-streaming models.
ASSEMBLYAI_U3_PRO_FAMILY_MODELS = {"universal-3-5-pro", "u3-rt-pro"}
# Benchmark languages within universal-3-5-pro's 18 supported languages
# (en, es, de, fr, pt, it, tr, nl, sv, no, da, fi, hi, vi, ar, he, ja, zh).
ASSEMBLYAI_UNIVERSAL_3_5_PRO_SUPPORTED_LANGUAGES = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "nl",
    "pt",
    "tr",
    "zh",
}
ASSEMBLYAI_U3_RT_PRO_SUPPORTED_LANGUAGES = {"en", "es", "de", "fr", "pt", "it"}
ASSEMBLYAI_UNIVERSAL_STREAMING_MULTILINGUAL_SUPPORTED_LANGUAGES = {"en", "es", "de", "fr", "pt", "it"}
ASSEMBLYAI_UNIVERSAL_STREAMING_EN_SUPPORTED_LANGUAGES = {"en"}
ASSEMBLYAI_WHISPER_STREAMING_BENCHMARK_LANGUAGES = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pt",
    "tr",
    "zh",
}


class AssemblyAIStreamingAdapter:
    """Replay full turns through AssemblyAI Streaming STT and score native turn detection."""

    display_name = "AssemblyAI"

    def __init__(
        self,
        *,
        model: str = DEFAULT_ASSEMBLYAI_MODEL,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        concurrency: int = DEFAULT_CONCURRENCY,
        min_turn_silence: int | None = DEFAULT_MIN_TURN_SILENCE_MS,
        max_turn_silence: int | None = DEFAULT_MAX_TURN_SILENCE_MS,
        end_of_turn_confidence_threshold: float | None = DEFAULT_END_OF_TURN_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if min_turn_silence is not None and min_turn_silence < 0:
            raise ValueError("min_turn_silence must be non-negative")
        if max_turn_silence is not None and max_turn_silence < 0:
            raise ValueError("max_turn_silence must be non-negative")
        if (
            min_turn_silence is not None
            and max_turn_silence is not None
            and max_turn_silence < min_turn_silence
        ):
            raise ValueError("max_turn_silence must be >= min_turn_silence")
        if end_of_turn_confidence_threshold is not None and not 0 <= end_of_turn_confidence_threshold <= 1:
            raise ValueError("end_of_turn_confidence_threshold must be in [0, 1]")

        self.model = model
        self.chunk_ms = int(chunk_ms)
        self.concurrency = int(concurrency)
        self.min_turn_silence = None if min_turn_silence is None else int(min_turn_silence)
        self.max_turn_silence = None if max_turn_silence is None else int(max_turn_silence)
        self.end_of_turn_confidence_threshold = (
            None if end_of_turn_confidence_threshold is None else float(end_of_turn_confidence_threshold)
        )

    @property
    def adapter_id(self) -> str:
        return f"assemblyai/{self.model}"

    def supports_language(self, lang_code: str) -> bool:
        if self.model == "universal-3-5-pro":
            return supports_language_code(lang_code, ASSEMBLYAI_UNIVERSAL_3_5_PRO_SUPPORTED_LANGUAGES)
        if self.model == "u3-rt-pro":
            return supports_language_code(lang_code, ASSEMBLYAI_U3_RT_PRO_SUPPORTED_LANGUAGES)
        if self.model == "universal-streaming-multilingual":
            return supports_language_code(lang_code, ASSEMBLYAI_UNIVERSAL_STREAMING_MULTILINGUAL_SUPPORTED_LANGUAGES)
        if self.model == "universal-streaming-english":
            return supports_language_code(lang_code, ASSEMBLYAI_UNIVERSAL_STREAMING_EN_SUPPORTED_LANGUAGES)
        if self.model == "whisper-rt":
            return supports_language_code(lang_code, ASSEMBLYAI_WHISPER_STREAMING_BENCHMARK_LANGUAGES)
        return False

    async def predict_turn(
        self,
        row: dict[str, Any],
        *,
        inference_interval: float = DEFAULT_INFERENCE_INTERVAL,
    ) -> dict[str, Any]:
        return await self._replay_turn(
            row,
            resolve_api_key("ASSEMBLYAI_API_KEY", "ASSEMBLY_API_KEY", "ASSEMBLY_AI_KEY"),
            inference_interval=inference_interval,
        )

    async def _replay_turn(
        self,
        row: dict[str, Any],
        api_key: str,
        *,
        inference_interval: float,
    ) -> dict[str, Any]:
        websockets = import_websockets()
        audio_bytes, total_audio_sec = prepare_pcm16_audio(row, sample_rate=SAMPLE_RATE)
        params = self._connection_params()
        url = f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}"

        events: list[dict[str, Any]] = []
        chunk_size = chunk_size_bytes(sample_rate=SAMPLE_RATE, chunk_ms=self.chunk_ms)
        async with websockets.connect(
            url,
            additional_headers={"Authorization": api_key},
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            started = time.perf_counter()
            recv_task = asyncio.create_task(_recv_assemblyai_events(ws, row_id=row["id"], events=events, started=started))
            for start in range(0, len(audio_bytes), chunk_size):
                await ws.send(audio_bytes[start : start + chunk_size])
                await asyncio.sleep(self.chunk_ms / 1000.0)
            await ws.send(json.dumps({"type": "Terminate"}))
            try:
                await asyncio.wait_for(recv_task, timeout=15.0)
            except asyncio.TimeoutError as exc:
                recv_task.cancel()
                raise RuntimeError(f"Timed out waiting for AssemblyAI termination on {row['id']!r}.") from exc

        if not events:
            raise RuntimeError(f"No AssemblyAI turn events received for row {row['id']!r}.")

        return {
            "id": row["id"],
            "audio_sec": total_audio_sec,
            "events": events,
            "prediction_rows": build_event_prediction_rows(
                row,
                events,
                inference_interval=inference_interval,
            ),
        }

    def _connection_params(self) -> dict[str, str]:
        params = {
            "speech_model": self.model,
            "encoding": "pcm_s16le",
            "sample_rate": str(SAMPLE_RATE),
        }
        if self.min_turn_silence is not None:
            params["min_turn_silence"] = str(int(self.min_turn_silence))
        if self.max_turn_silence is not None:
            params["max_turn_silence"] = str(int(self.max_turn_silence))
        if (
            self.end_of_turn_confidence_threshold is not None
            and self.model not in ASSEMBLYAI_U3_PRO_FAMILY_MODELS
        ):
            params["end_of_turn_confidence_threshold"] = str(float(self.end_of_turn_confidence_threshold))
        return params


async def _recv_assemblyai_events(
    ws,
    *,
    row_id: Any,
    events: list[dict[str, Any]],
    started: float,
) -> None:
    async for message in ws:
        data = json.loads(message)
        msg_type = data.get("type")
        if msg_type == "Termination":
            return
        if msg_type == "Error":
            raise RuntimeError(f"AssemblyAI error for row {row_id!r}: {data}")
        if msg_type != "Turn":
            continue
        event = _assemblyai_event_from_turn(data, received_audio_sec=time.perf_counter() - started)
        if event is not None:
            events.append(event)


def _assemblyai_event_from_turn(data: dict[str, Any], *, received_audio_sec: float) -> dict[str, Any] | None:
    confidence = data.get("end_of_turn_confidence")
    end_of_turn = bool(data.get("end_of_turn"))
    if confidence is None and not end_of_turn:
        return None

    score = 1.0 if confidence is None else float(confidence)
    return {
        "event": "Turn",
        "timestamp": float(received_audio_sec),
        "p_eot": score,
        "end_of_turn": end_of_turn,
        "end_of_turn_confidence": confidence,
        "turn_order": data.get("turn_order"),
        "transcript": data.get("transcript", ""),
        "utterance": data.get("utterance", ""),
    }
