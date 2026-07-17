from __future__ import annotations

import pytest

from eot_harness.assemblyai_adapter import (
    AssemblyAIBalancedModeAdapter,
    AssemblyAIMaxAccuracyModeAdapter,
    AssemblyAIStreamingAdapter,
    _assemblyai_event_from_turn,
)
from eot_harness.openai_realtime_adapter import OpenAIRealtime2Adapter, _openai_speech_stopped_event
from eot_harness.soniox_adapter import SonioxStreamingAdapter, _soniox_endpoint_event
from eot_harness.streaming_stt import build_event_prediction_rows, resolve_api_key
from eot_harness.xai_stt_adapter import XAIStreamingSTTAdapter, _xai_endpoint_event


def _row():
    return {
        "id": "turn-1",
        "silence_spans": [
            {"start": 1.0, "end": 1.4},
            {"start": 2.0, "end": 2.3},
        ],
    }


def test_build_event_prediction_rows_forwards_latest_score():
    rows = build_event_prediction_rows(
        _row(),
        [
            {"event": "ignored", "timestamp": None, "p_eot": None},
            {"event": "before_span", "timestamp": 0.9, "p_eot": 1.0},
            {"event": "low", "timestamp": 1.2, "p_eot": 0.12},
            {"event": "endpoint", "timestamp": 2.2, "p_eot": 1.0},
        ],
        inference_interval=0.2,
    )

    assert [row["timestamp"] for row in rows] == [1.0, 1.2, 1.4, 2.0, 2.2, 2.3]
    assert [row["label"] for row in rows] == ["hold", "hold", "hold", "eot", "eot", "eot"]
    assert [row["p_eot"] for row in rows] == [0.0, 0.12, 0.12, 0.0, 1.0, 1.0]


def test_build_event_prediction_rows_all_zero_when_no_events():
    rows = build_event_prediction_rows(
        _row(),
        [],
        inference_interval=0.2,
    )

    assert [row["timestamp"] for row in rows] == [1.0, 1.2, 1.4, 2.0, 2.2, 2.3]
    assert [row["p_eot"] for row in rows] == [0.0] * 6


def test_assemblyai_turn_event_uses_realtime_receive_position():
    event = _assemblyai_event_from_turn(
        {
            "type": "Turn",
            "turn_order": 2,
            "end_of_turn": True,
            "end_of_turn_confidence": 0.72,
            "transcript": "done",
            "utterance": "done",
        },
        received_audio_sec=1.35,
    )

    assert event == {
        "event": "Turn",
        "timestamp": 1.35,
        "p_eot": 0.72,
        "end_of_turn": True,
        "end_of_turn_confidence": 0.72,
        "turn_order": 2,
        "transcript": "done",
        "utterance": "done",
    }


def test_assemblyai_defaults_and_key(monkeypatch):
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    monkeypatch.setenv("ASSEMBLY_API_KEY", "aai-test-key")
    adapter = AssemblyAIStreamingAdapter()

    assert adapter.adapter_id == "assemblyai/universal-3-5-pro"
    assert adapter.display_name == "AssemblyAI Universal-3.5 Pro"
    assert not hasattr(adapter, "score_point")
    assert resolve_api_key("ASSEMBLYAI_API_KEY", "ASSEMBLY_API_KEY", "ASSEMBLY_AI_KEY") == "aai-test-key"
    assert adapter._connection_params() == {
        "speech_model": "universal-3-5-pro",
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "min_turn_silence": "100",
        "max_turn_silence": "3000",
    }


def test_assemblyai_universal_3_5_pro_language_support():
    adapter = AssemblyAIStreamingAdapter(model="universal-3-5-pro")

    for lang in ("ar", "de", "en", "es", "fr", "hi", "it", "ja", "nl", "pt", "tr", "zh"):
        assert adapter.supports_language(lang), lang
    assert not adapter.supports_language("id")
    assert not adapter.supports_language("ko")


def test_assemblyai_u3_pro_models_omit_end_of_turn_confidence_threshold():
    for model in ("universal-3-5-pro", "u3-rt-pro"):
        adapter = AssemblyAIStreamingAdapter(model=model, end_of_turn_confidence_threshold=0.1)
        assert "end_of_turn_confidence_threshold" not in adapter._connection_params(), model


def test_assemblyai_u3_rt_pro_language_support():
    adapter = AssemblyAIStreamingAdapter(model="u3-rt-pro")

    for lang in ("en", "es", "de", "fr", "pt", "it"):
        assert adapter.supports_language(lang), lang
    assert not adapter.supports_language("ja")
    assert not adapter.supports_language("tr")


def test_assemblyai_mode_validation():
    with pytest.raises(ValueError, match="mode"):
        AssemblyAIStreamingAdapter(mode="turbo")


def test_assemblyai_mode_param_only_sent_for_u3_pro_models():
    adapter = AssemblyAIStreamingAdapter(mode="balanced")
    assert adapter._connection_params()["mode"] == "balanced"

    legacy = AssemblyAIStreamingAdapter(model="universal-streaming-multilingual", mode="balanced")
    assert "mode" not in legacy._connection_params()


def test_assemblyai_mode_is_part_of_adapter_identity():
    adapter = AssemblyAIStreamingAdapter(mode="balanced")
    assert adapter.adapter_id == "assemblyai/universal-3-5-pro-mode-balanced"
    assert adapter.display_name == "AssemblyAI Universal-3.5 Pro (balanced)"


def test_assemblyai_balanced_mode_adapter_uses_server_turn_silence_defaults():
    adapter = AssemblyAIBalancedModeAdapter()

    assert adapter.adapter_id == "assemblyai/universal-3-5-pro-mode-balanced"
    assert adapter.supports_language("ja")
    assert adapter._connection_params() == {
        "speech_model": "universal-3-5-pro",
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "mode": "balanced",
    }


def test_assemblyai_max_accuracy_mode_adapter_uses_server_turn_silence_defaults():
    adapter = AssemblyAIMaxAccuracyModeAdapter()

    assert adapter.adapter_id == "assemblyai/universal-3-5-pro-mode-max_accuracy"
    assert adapter.display_name == "AssemblyAI Universal-3.5 Pro (max_accuracy)"
    assert adapter._connection_params() == {
        "speech_model": "universal-3-5-pro",
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "mode": "max_accuracy",
    }


def test_assemblyai_display_name_tracks_model():
    assert AssemblyAIStreamingAdapter(model="universal-3-5-pro").display_name == "AssemblyAI Universal-3.5 Pro"
    # Matches the committed leaderboard artifacts for the previous default model.
    assert AssemblyAIStreamingAdapter(model="universal-streaming-multilingual").display_name == "AssemblyAI"
    assert AssemblyAIStreamingAdapter(model="u3-rt-pro").display_name == "AssemblyAI u3-rt-pro"


def test_assemblyai_universal_streaming_multilingual_keeps_confidence_threshold():
    adapter = AssemblyAIStreamingAdapter(model="universal-streaming-multilingual")

    assert adapter.adapter_id == "assemblyai/universal-streaming-multilingual"
    assert adapter.supports_language("en")
    assert adapter.supports_language("pt")
    assert not adapter.supports_language("ja")
    assert adapter._connection_params() == {
        "speech_model": "universal-streaming-multilingual",
        "encoding": "pcm_s16le",
        "sample_rate": "16000",
        "min_turn_silence": "100",
        "max_turn_silence": "3000",
        "end_of_turn_confidence_threshold": "0.1",
    }


def test_assemblyai_cli_style_model_override_rebinds_params_and_languages():
    adapter = AssemblyAIStreamingAdapter()
    adapter.model = "universal-streaming-multilingual"

    assert adapter.adapter_id == "assemblyai/universal-streaming-multilingual"
    assert adapter._connection_params()["end_of_turn_confidence_threshold"] == "0.1"
    assert not adapter.supports_language("ja")


def test_assemblyai_english_model_supports_only_english():
    adapter = AssemblyAIStreamingAdapter(model="universal-streaming-english")

    assert adapter.supports_language("en")
    assert not adapter.supports_language("es")
    assert not adapter.supports_language("fr")


def test_assemblyai_unknown_model_supports_no_benchmark_languages():
    adapter = AssemblyAIStreamingAdapter(model="unknown-model")

    assert not adapter.supports_language("en")
    assert not adapter.supports_language("es")


def test_assemblyai_turn_silence_validation():
    with pytest.raises(ValueError, match="min_turn_silence"):
        AssemblyAIStreamingAdapter(min_turn_silence=-1)
    with pytest.raises(ValueError, match="max_turn_silence"):
        AssemblyAIStreamingAdapter(max_turn_silence=-1)
    with pytest.raises(ValueError, match="max_turn_silence must be >= min_turn_silence"):
        AssemblyAIStreamingAdapter(min_turn_silence=500, max_turn_silence=100)


def test_soniox_endpoint_event_detects_final_end_token():
    event = _soniox_endpoint_event(
        {
            "tokens": [
                {"text": "hello", "is_final": True, "end_ms": 600},
                {"text": "<end>", "is_final": True},
            ],
            "final_audio_proc_ms": 1250,
            "total_audio_proc_ms": 1300,
        },
    )

    assert event == {
        "event": "Endpoint",
        "timestamp": 1.25,
        "p_eot": 1.0,
        "token": "<end>",
        "final_audio_proc_ms": 1250,
        "total_audio_proc_ms": 1300,
    }


def test_soniox_defaults_and_key(monkeypatch):
    monkeypatch.setenv("SONIOX_API_KEY", "soniox-test-key")
    adapter = SonioxStreamingAdapter()

    assert adapter.adapter_id == "soniox/stt-rt-preview"
    assert adapter.display_name == "Soniox"
    assert not hasattr(adapter, "score_point")
    assert adapter.supports_language("ja")
    assert resolve_api_key("SONIOX_API_KEY") == "soniox-test-key"
    start = adapter._start_message("redacted", language="de")
    assert start["enable_endpoint_detection"] is True
    assert start["language_hints"] == ["de"]


def test_xai_endpoint_event_uses_start_plus_duration():
    event = _xai_endpoint_event(
        {
            "type": "transcript.partial",
            "text": "done",
            "is_final": True,
            "speech_final": True,
            "start": 1.1,
            "duration": 0.4,
        },
    )

    assert event == {
        "event": "UtteranceFinal",
        "timestamp": 1.5,
        "p_eot": 1.0,
        "transcript": "done",
        "start": 1.1,
        "duration": 0.4,
    }


def test_xai_defaults_and_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    adapter = XAIStreamingSTTAdapter()

    assert adapter.adapter_id == "xai/stt-streaming"
    assert adapter.display_name == "xAI STT"
    assert not hasattr(adapter, "score_point")
    assert adapter.supports_language("ko")
    assert not adapter.supports_language("zh")
    assert resolve_api_key("XAI_API_KEY") == "xai-test-key"
    params = adapter._connection_params(language="es")
    assert params["endpointing"] == "10"
    assert params["language"] == "es"


def test_openai_realtime_speech_stopped_event_uses_audio_end_ms():
    event = _openai_speech_stopped_event(
        {
            "type": "input_audio_buffer.speech_stopped",
            "audio_start_ms": 420,
            "audio_end_ms": 1337,
            "item_id": "item_123",
        },
    )

    assert event == {
        "event": "SpeechStopped",
        "timestamp": 1.337,
        "p_eot": 1.0,
        "audio_end_ms": 1337,
        "audio_start_ms": 420,
        "item_id": "item_123",
    }


def test_openai_realtime_defaults_and_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    adapter = OpenAIRealtime2Adapter()

    assert adapter.adapter_id == "openai/gpt-realtime-2/semantic_vad_auto"
    assert adapter.display_name == "OpenAI GPT Realtime 2"
    assert not hasattr(adapter, "score_point")
    assert adapter.supports_language("zh")
    assert resolve_api_key("OPENAI_API_KEY") == "openai-test-key"
    session = adapter._session_update()["session"]
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24000,
    }
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "auto",
        "create_response": False,
        "interrupt_response": False,
    }


def test_missing_api_key_error(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        resolve_api_key("XAI_API_KEY")
