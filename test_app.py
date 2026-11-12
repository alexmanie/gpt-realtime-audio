import io
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
from scipy.io import wavfile

import app


class Signal:
    def connect(self, callback):
        self.callback = callback


class SpeechToTextTests(unittest.TestCase):
    def test_speak_last_reply_uses_latest_assistant_message(self):
        history = [
            {"role": "assistant", "content": "Older reply"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Latest reply"},
            {"role": "user", "content": "Pending question"},
        ]
        audio = (24000, np.array([0, 100, -100], dtype=np.int16))
        with patch.object(app, "synthesize_speech", return_value=audio) as tts:
            result = app.speak_last_reply(history, app.DEFAULT_VOICE)
        tts.assert_called_once_with("Latest reply", app.DEFAULT_VOICE, False)
        self.assertIs(result, audio)
        self.assertEqual(len(history), 4)

    def test_speak_without_reply_reports_error(self):
        for history in (None, [], [{"role": "user", "content": "Hello"}]):
            with self.subTest(history=history):
                with patch.object(app, "synthesize_speech") as tts:
                    with self.assertRaises(gr.Error):
                        app.speak_last_reply(history, app.DEFAULT_VOICE)
                    tts.assert_not_called()

    def test_ui_routes_tts_from_session_history(self):
        ui = app.build_ui()
        speak = next(fn for fn in ui.fns.values() if fn.fn is app.speak_last_reply)
        send = next(fn for fn in ui.fns.values() if fn.fn is app.respond)
        self.assertIs(speak.inputs[0], send.inputs[2])
        self.assertIsInstance(speak.outputs[0], gr.Audio)
        self.assertTrue(speak.outputs[0].autoplay)
        self.assertEqual(speak.inputs[1].value, app.DEFAULT_VOICE)
        self.assertIs(speak.inputs[2], send.inputs[3])
        self.assertIsInstance(send.inputs[3], gr.Checkbox)
        self.assertFalse(send.inputs[3].value)

    def test_selected_mode_is_forwarded_to_stt_and_tts(self):
        for use_container in (False, True):
            with self.subTest(use_container=use_container):
                with (
                    patch.object(app, "transcribe_audio", return_value="Hello") as stt,
                    patch.object(app, "generate_reply", return_value="Hi"),
                    patch.object(app, "synthesize_speech", return_value="audio") as tts,
                ):
                    updates = list(app.respond("recording.wav", "en-US", [], use_container))
                    self.assertEqual(
                        app.speak_last_reply(updates[-1][1], app.DEFAULT_VOICE, use_container),
                        "audio",
                    )
                stt.assert_called_once_with("recording.wav", "en-US", use_container)
                tts.assert_called_once_with("Hi", app.DEFAULT_VOICE, use_container)

    def test_transcription_selects_mode_and_language_endpoint(self):
        cases = [
            (False, "en-US", "_speech_config"),
            (True, "en-US", "_speech_config_container_en_stt"),
            (True, "sl-SI", "_speech_config_container_sl_stt"),
        ]
        for use_container, language, expected in cases:
            with self.subTest(use_container=use_container, language=language):
                recognizer = Mock(
                    recognized=Signal(), canceled=Signal(), session_stopped=Signal()
                )

                def start():
                    recognizer.recognized.callback(SimpleNamespace(result=SimpleNamespace(
                        reason=app.speechsdk.ResultReason.RecognizedSpeech, text="Hello"
                    )))
                    recognizer.session_stopped.callback(None)

                recognizer.start_continuous_recognition_async.return_value.get.side_effect = start
                with (
                    patch.object(Path, "is_file", return_value=True),
                    patch.object(app, "_prepare_stt_audio", return_value=(b"audio", 1, "audio")),
                    patch.object(app, "_speech_config", return_value=Mock()) as online,
                    patch.object(app, "_speech_config_container_en_stt", return_value=Mock()) as en,
                    patch.object(app, "_speech_config_container_sl_stt", return_value=Mock()) as sl,
                    patch.object(app.speechsdk.audio, "PushAudioInputStream"),
                    patch.object(app.speechsdk, "AudioConfig"),
                    patch.object(app.speechsdk, "SpeechRecognizer", return_value=recognizer),
                ):
                    self.assertEqual(app.transcribe_audio("recording.wav", language, use_container), "Hello")
                for name, config in (
                    ("_speech_config", online),
                    ("_speech_config_container_en_stt", en),
                    ("_speech_config_container_sl_stt", sl),
                ):
                    self.assertEqual(config.call_count, int(name == expected))

    def test_synthesis_selects_mode_and_voice_endpoint(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24000)
            audio.writeframes(b"\x00\x00")
        for use_container in (False, True):
            for voice in app.AVAILABLE_VOICES:
                with self.subTest(use_container=use_container, voice=voice):
                    result = SimpleNamespace(
                        reason=app.speechsdk.ResultReason.SynthesizingAudioCompleted,
                        audio_data=wav.getvalue(),
                    )
                    synthesizer = Mock()
                    synthesizer.speak_text_async.return_value.get.return_value = result
                    with (
                        patch.object(app, "_speech_config", return_value=Mock()) as online,
                        patch.object(app, "_speech_config_container_tts", return_value=Mock()) as container,
                        patch.object(app.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
                    ):
                        rate, samples = app.synthesize_speech("Hello", voice, use_container)
                    self.assertEqual(rate, 24000)
                    self.assertEqual(samples.tolist(), [0])
                    self.assertEqual(online.call_count, int(not use_container))
                    if use_container:
                        container.assert_called_once_with(voice)
                    else:
                        container.assert_not_called()

    def test_missing_container_endpoint_reports_error_without_online_fallback(self):
        with (
            patch.object(app, "CONTAINER_TTS_SERAPHINA_ENDPOINT", ""),
            patch.object(app, "_speech_config") as online,
        ):
            with self.assertRaises(gr.Error) as error:
                app.synthesize_speech("Hello", app.DEFAULT_VOICE, True)
            self.assertIn("CONTAINER_TTS_SERAPHINA_ENDPOINT", str(error.exception))
            online.assert_not_called()
        with (
            patch.object(Path, "is_file", return_value=True),
            patch.object(app, "_prepare_stt_audio", return_value=(b"audio", 1, "audio")),
            patch.object(app, "CONTAINER_STT_EN_ENDPOINT", ""),
            patch.object(app, "_speech_config") as online,
        ):
            with self.assertRaises(gr.Error) as error:
                app.transcribe_audio("recording.wav", "en-US", True)
            self.assertIn("CONTAINER_STT_EN_ENDPOINT", str(error.exception))
            online.assert_not_called()

    def test_container_configs_select_configured_hosts(self):
        with (
            patch.object(app, "CONTAINER_STT_EN_ENDPOINT", "http://stt-en:5000"),
            patch.object(app, "CONTAINER_STT_SL_ENDPOINT", "http://stt-sl:5000"),
            patch.object(app, "CONTAINER_TTS_SERAPHINA_ENDPOINT", "http://seraphina:5000"),
            patch.object(app, "CONTAINER_TTS_ANDREW_ENDPOINT", "http://andrew:5000"),
            patch.object(app.speechsdk, "SpeechConfig") as config,
        ):
            for configure, host in (
                (app._speech_config_container_en_stt, "http://stt-en:5000"),
                (app._speech_config_container_sl_stt, "http://stt-sl:5000"),
                (lambda: app._speech_config_container_tts(app.AVAILABLE_VOICES[0]), "http://seraphina:5000"),
                (lambda: app._speech_config_container_tts(app.AVAILABLE_VOICES[1]), "http://andrew:5000"),
            ):
                with self.subTest(host=host):
                    configure()
                    config.assert_called_with(host=host)

    def test_respond_shows_transcript_before_calling_llm(self):
        history = [{"role": "user", "content": "Previous message"}]
        with (
            patch.object(app, "transcribe_audio", return_value="Hello"),
            patch.object(app, "generate_reply", return_value="Hi!") as chat,
            patch.object(app, "synthesize_speech") as tts,
        ):
            updates = app.respond("recording.wav", "en-US", history)
            first, state, audio_update = next(updates)
            self.assertEqual(audio_update, gr.skip())
            self.assertEqual(first[-1], {"role": "user", "content": "Hello"})
            chat.assert_not_called()
            final, final_state, audio_update = next(updates)
            self.assertIsNone(audio_update)
            chat.assert_called_once_with(state, "en-US")
            self.assertEqual(final[-1], {"role": "assistant", "content": "Hi!"})
            self.assertEqual(final, final_state)
            self.assertEqual(len(first), 2)
            self.assertEqual(len(history), 1)
            tts.assert_not_called()
            with self.assertRaises(StopIteration):
                next(updates)

    def test_llm_failure_does_not_remove_transcript(self):
        with (
            patch.object(app, "transcribe_audio", return_value="Hello"),
            patch.object(app, "generate_reply", side_effect=gr.Error("LLM failed")),
        ):
            updates = app.respond("recording.wav", "en-US", [])
            dialog, state, audio_update = next(updates)
            self.assertEqual(audio_update, gr.skip())
            with self.assertRaises(gr.Error):
                next(updates)
            self.assertEqual(dialog, [{"role": "user", "content": "Hello"}])
            self.assertEqual(dialog, state)

    def test_regional_endpoint_with_entra_reports_auth_configuration_error(self):
        with (
            patch.object(
                app, "AZURE_SPEECH_ENDPOINT",
                "https://northeurope.api.cognitive.microsoft.com/",
            ),
            patch.object(app, "AZURE_SPEECH_KEY", ""),
            patch.object(app, "ChainedTokenCredential") as credential,
        ):
            with self.assertRaises(gr.Error) as error:
                app._speech_config()
        self.assertIn("resource-specific custom endpoint", str(error.exception))
        credential.assert_not_called()

    def test_custom_endpoint_preserves_token_credential_auth(self):
        credential = Mock()
        with (
            patch.object(app, "AZURE_SPEECH_ENDPOINT", "https://speech.example.com/"),
            patch.object(app, "AZURE_SPEECH_KEY", ""),
            patch.object(app, "AzureCliCredential"),
            patch.object(app, "DefaultAzureCredential"),
            patch.object(app, "ChainedTokenCredential", return_value=credential),
            patch.object(app.speechsdk, "SpeechConfig") as config,
        ):
            app._speech_config()
        config.assert_called_once_with(
            token_credential=credential, endpoint="https://speech.example.com/"
        )

    def test_send_appends_user_message_without_chat_or_tts(self):
        history = [{"role": "user", "content": "Previous message"}]
        with (
            patch.object(app, "transcribe_audio", return_value="Hello world") as stt,
            patch.object(app, "generate_reply") as chat,
            patch.object(app, "synthesize_speech") as tts,
        ):
            dialog, state = app.send_to_stt("recording.wav", "en-US", history)
        stt.assert_called_once_with("recording.wav", "en-US", False)
        self.assertEqual(dialog[-1], {"role": "user", "content": "Hello world"})
        self.assertEqual(dialog, state)
        self.assertEqual(len(history), 1)
        chat.assert_not_called()
        tts.assert_not_called()

    def test_failed_recognition_preserves_history(self):
        history = [{"role": "user", "content": "Previous message"}]
        with patch.object(app, "transcribe_audio", side_effect=gr.Error("No speech")):
            with self.assertRaises(gr.Error):
                app.send_to_stt("recording.wav", "en-US", history)
        self.assertEqual(history, [{"role": "user", "content": "Previous message"}])

    def test_missing_recording_is_reported(self):
        with self.assertRaises(gr.Error):
            app.transcribe_audio("", "en-US")

    def test_session_histories_are_independent(self):
        with patch.object(app, "transcribe_audio", return_value="Hello"):
            first, _ = app.send_to_stt("recording.wav", "en-US", [])
            second, _ = app.send_to_stt("recording.wav", "en-US", [])
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertIsNot(first, second)

    def test_explicit_endpoint_used_for_key_auth(self):
        with (
            patch.object(app, "AZURE_SPEECH_ENDPOINT", "https://speech.example.com/"),
            patch.object(app, "AZURE_SPEECH_KEY", "test-key"),
            patch.object(app.speechsdk, "SpeechConfig") as config,
        ):
            app._speech_config()
        config.assert_called_once_with(
            subscription="test-key", endpoint="https://speech.example.com/"
        )

    def test_entra_token_includes_resource_id(self):
        credential = Mock()
        credential.get_token.return_value = SimpleNamespace(token="test-token")
        with (
            patch.object(app, "AZURE_SPEECH_ENDPOINT", ""),
            patch.object(app, "AZURE_SPEECH_REGION", "eastus"),
            patch.object(app, "AZURE_SPEECH_KEY", ""),
            patch.object(app, "AZURE_SPEECH_RESOURCE_ID", "/test/speech-resource"),
            patch.object(app, "AzureCliCredential"),
            patch.object(app, "DefaultAzureCredential"),
            patch.object(app, "ChainedTokenCredential", return_value=credential),
            patch.object(app.speechsdk, "SpeechConfig") as config,
        ):
            app._speech_config()
        config.assert_called_once_with(
            auth_token="aad#/test/speech-resource#test-token", region="eastus"
        )
        credential.close.assert_called_once()

    def test_all_sentences_are_transcribed_and_errors_surface(self):
        for outcome in ("success", "canceled", "empty", "no_match"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                audio_path = Path(directory) / "recording.wav"
                with wave.open(str(audio_path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(b"\x00\x10" * 160)

                recognizer = Mock(
                    recognized=Signal(), canceled=Signal(), session_stopped=Signal()
                )

                def start():
                    if outcome == "canceled":
                        recognizer.canceled.callback(
                            SimpleNamespace(
                                reason=app.speechsdk.CancellationReason.Error,
                                error_details="Speech endpoint rejected authentication",
                            )
                        )
                    else:
                        if outcome == "success":
                            for text in ("First sentence.", "Second sentence."):
                                recognizer.recognized.callback(
                                    SimpleNamespace(result=SimpleNamespace(
                                        reason=app.speechsdk.ResultReason.RecognizedSpeech,
                                        text=text,
                                    ))
                                )
                        elif outcome == "no_match":
                            recognizer.recognized.callback(
                                SimpleNamespace(result=SimpleNamespace(
                                    reason=app.speechsdk.ResultReason.NoMatch,
                                    no_match_details=SimpleNamespace(reason="NotRecognized"),
                                ))
                            )
                        recognizer.session_stopped.callback(None)

                recognizer.start_continuous_recognition_async.return_value.get.side_effect = start
                with (
                    patch.object(app, "_speech_config", return_value=Mock()),
                    patch.object(app.speechsdk, "AudioConfig"),
                    patch.object(app.speechsdk.audio, "PushAudioInputStream") as stream,
                    patch.object(app.speechsdk, "SpeechRecognizer", return_value=recognizer),
                ):
                    if outcome == "success":
                        self.assertEqual(
                            app.transcribe_audio(str(audio_path), "en-US"),
                            "First sentence. Second sentence.",
                        )
                    else:
                        with self.assertRaises(gr.Error) as error:
                            app.transcribe_audio(str(audio_path), "en-US")
                        if outcome == "no_match":
                            self.assertIn("NotRecognized", str(error.exception))
                            self.assertIn("16000 Hz mono PCM16", str(error.exception))
                            self.assertIn("en-US", str(error.exception))
                    stream.return_value.write.assert_called_once()
                    self.assertEqual(len(stream.return_value.write.call_args.args[0]), 320)
                recognizer.stop_continuous_recognition_async.assert_called_once()

    def test_normalizes_browser_audio_formats(self):
        for rate in (16000, 44100, 48000):
            for dtype in (np.int16, np.int32, np.float32, np.uint8):
                with self.subTest(rate=rate, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                    signal = 0.4 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
                    stereo = np.column_stack((signal, signal * 0.5))
                    if dtype == np.uint8:
                        source = (stereo * 128 + 128).astype(dtype)
                    elif np.issubdtype(dtype, np.integer):
                        source = (stereo * (2 ** (np.dtype(dtype).itemsize * 8 - 1))).astype(dtype)
                    else:
                        source = stereo.astype(dtype)
                    path = Path(directory) / "browser.wav"
                    wavfile.write(path, rate, source)
                    pcm, duration, details = app._prepare_stt_audio(str(path))
                    normalized = np.frombuffer(pcm, dtype="<i2")
                    self.assertEqual(len(pcm), 32000)
                    self.assertEqual(duration, 1)
                    self.assertIn("2 channel(s)", details)
                    expected = 0.3 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
                    np.testing.assert_allclose(
                        normalized[100:-100] / 32768, expected[100:-100], atol=0.01
                    )

    def test_silence_empty_invalid_and_non_wav_are_reported(self):
        cases = [
            ("silence", np.zeros(1600, dtype=np.int16), "only silence"),
            ("empty", np.array([], dtype=np.int16), "empty"),
            ("invalid", np.array([np.nan], dtype=np.float32), "invalid audio samples"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.wav"
            for name, samples, message in cases:
                with self.subTest(name=name):
                    wavfile.write(path, 16000, samples)
                    with self.assertRaises(gr.Error) as error:
                        app.transcribe_audio(str(path), "en-US")
                    self.assertIn(message, str(error.exception))
            path.write_bytes(b"not a wav file")
            with self.assertRaises(gr.Error) as error:
                app.transcribe_audio(str(path), "en-US")
            self.assertIn("Cannot read", str(error.exception))

    def test_ui_routes_send_to_stt_and_llm(self):
        ui = app.build_ui()
        send = next(fn for fn in ui.fns.values() if fn.fn is app.respond)
        self.assertEqual(len(send.inputs), 4)
        self.assertEqual(len(send.outputs), 3)
        self.assertIsInstance(send.outputs[0], gr.Chatbot)
        audio = send.inputs[0]
        self.assertIs(send.outputs[2], audio)
        self.assertIn("microphone", audio.sources)
        self.assertEqual(audio.format, "wav")


if __name__ == "__main__":
    unittest.main()
