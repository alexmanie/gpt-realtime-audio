import io
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import gradio as gr
import numpy as np
from scipy.io import wavfile

import app


class Signal:
    def connect(self, callback):
        self.callback = callback


class ContainerReadinessTests(unittest.TestCase):
    def test_ready_requires_exact_ready_field_in_json_object(self):
        cases = [
            (b'{"ready":"ready","message":"API key is valid"}', True, "API key is valid"),
            (b'{"ready":"starting","message":"Loading"}', False, "Loading"),
            (b'{"ready":true}', False, 'Response does not contain'),
            (b'{}', False, 'Response does not contain'),
            (b'[]', False, "expected a JSON object"),
            (b'not json', False, "Readiness check failed"),
        ]
        for body, expected, message in cases:
            with self.subTest(body=body):
                response = Mock()
                response.__enter__ = Mock(return_value=io.BytesIO(body))
                response.__exit__ = Mock(return_value=False)
                with patch.object(app, "urlopen", return_value=response) as request:
                    ready, detail = app._container_readiness("http://speech:5012/")
                self.assertEqual(ready, expected)
                self.assertIn(message, detail)
                request.assert_called_once_with("http://speech:5012/ready", timeout=3)

    def test_unavailable_or_unconfigured_container_is_not_ready(self):
        with patch.object(app, "urlopen") as request:
            self.assertEqual(
                app._container_readiness(""), (False, "Endpoint not configured")
            )
            request.assert_not_called()
        for error in (
            HTTPError("http://speech:5012/ready", 503, "Unavailable", {}, None),
            URLError("Connection refused"),
            TimeoutError("Timed out"),
        ):
            with self.subTest(error=error):
                with patch.object(app, "urlopen", side_effect=error):
                    ready, detail = app._container_readiness("http://speech:5012")
                self.assertFalse(ready)
                self.assertIn("Readiness check failed", detail)

    def test_indicators_show_each_service_and_escape_response_messages(self):
        with (
            patch.object(app, "CONTAINER_STT_EN_ENDPOINT", "http://en:5000"),
            patch.object(app, "CONTAINER_STT_SL_ENDPOINT", "http://sl:5000"),
            patch.object(app, "CONTAINER_TTS_SERAPHINA_ENDPOINT", "http://seraphina:5000"),
            patch.object(app, "CONTAINER_TTS_ANDREW_ENDPOINT", "http://andrew:5000"),
            patch.object(app, "_container_readiness", side_effect=[
                (True, "Available"), (False, "<script>bad</script>"),
                (False, "Loading"), (True, "Available"),
            ]) as check,
        ):
            html = app.check_container_status()
        self.assertEqual(
            [call.args[0] for call in check.call_args_list],
            ["http://en:5000", "http://sl:5000", "http://seraphina:5000", "http://andrew:5000"],
        )
        self.assertIn("STT English: Ready", html)
        self.assertIn("STT Slovenian: Not ready", html)
        self.assertIn("TTS Seraphina: Not ready", html)
        self.assertIn("TTS Andrew: Ready", html)
        self.assertEqual(html.count("background:#16a34a"), 2)
        self.assertEqual(html.count("background:#dc2626"), 2)
        self.assertEqual(html.count("background:#f0fdf4"), 2)
        self.assertEqual(html.count("background:#fef2f2"), 2)
        self.assertEqual(html.count("border-radius:6px"), 4)
        self.assertNotIn("<script>", html)
        self.assertIn('title="&lt;script&gt;bad&lt;/script&gt;"', html)

    def test_ui_checks_on_load_and_every_30_seconds_without_conversation_queue(self):
        with patch.object(app, "urlopen") as request:
            ui = app.build_ui()
            request.assert_not_called()
        checks = [fn for fn in ui.fns.values() if fn.fn is app.check_container_status]
        self.assertEqual(len(checks), 2)
        self.assertIs(checks[0].outputs[0], checks[1].outputs[0])
        self.assertIsInstance(checks[0].outputs[0], gr.HTML)
        targets = [target for fn in checks for target in fn.targets]
        self.assertIn((ui._id, "load"), targets)
        timer_id = next(component_id for component_id, event in targets if event == "tick")
        self.assertEqual(ui.blocks[timer_id].value, 30)
        for fn in checks:
            self.assertFalse(fn.queue)

    def test_status_indicators_are_in_header_next_to_title(self):
        ui = app.build_ui()
        check = next(fn for fn in ui.fns.values() if fn.fn is app.check_container_status)
        header = ui.children[0]
        self.assertIsInstance(header, gr.Row)
        self.assertEqual(
            header.children[0].children[0].value,
            "# Speech to Text and Text to Speech",
        )
        self.assertIs(header.children[1].children[0], check.outputs[0])


class SpeechToTextTests(unittest.TestCase):
    def test_speak_without_text_reports_error_before_contacting_service(self):
        for text in ("", " \n\t"):
            with self.subTest(text=text):
                with patch.object(app, "_speech_config") as config:
                    with self.assertRaises(gr.Error):
                        app.synthesize_speech(text, app.DEFAULT_VOICE)
                    config.assert_not_called()

    def test_ui_routes_tts_from_editable_transcript_without_conversation(self):
        ui = app.build_ui()
        speak = next(fn for fn in ui.fns.values() if fn.fn is app.synthesize_speech)
        send = next(fn for fn in ui.fns.values() if fn.fn is app.send_to_stt)
        self.assertIs(speak.inputs[0], send.outputs[0])
        self.assertIsInstance(speak.inputs[0], gr.Textbox)
        self.assertTrue(speak.inputs[0].interactive)
        self.assertGreater(speak.inputs[0].lines, 1)
        self.assertFalse(any(isinstance(block, (gr.Chatbot, gr.State)) for block in ui.blocks.values()))
        self.assertIsInstance(speak.outputs[0], gr.Audio)
        self.assertTrue(speak.outputs[0].autoplay)
        self.assertEqual(speak.inputs[1].value, app.DEFAULT_VOICE)
        self.assertIs(speak.inputs[2], send.inputs[2])
        self.assertIsInstance(send.inputs[2], gr.Checkbox)
        self.assertIs(send.outputs[2], speak.outputs[0])

    def test_clear_resets_text_and_both_audio_components(self):
        ui = app.build_ui()
        clear = next(
            fn for fn in ui.fns.values()
            if any(getattr(ui.blocks.get(target), "value", None) == "Clear" for target, _ in fn.targets)
        )
        send = next(fn for fn in ui.fns.values() if fn.fn is app.send_to_stt)
        self.assertEqual(clear.outputs, send.outputs)
        self.assertEqual(clear.fn(), ("", None, None))

    def test_selected_mode_is_forwarded_to_stt_without_automatic_tts(self):
        for use_container in (False, True):
            with self.subTest(use_container=use_container):
                with (
                    patch.object(app, "transcribe_audio", return_value="Hello") as stt,
                    patch.object(app, "synthesize_speech") as tts,
                ):
                    self.assertEqual(
                        app.send_to_stt("recording.wav", "en-US", use_container),
                        ("Hello", None, None),
                    )
                stt.assert_called_once_with("recording.wav", "en-US", use_container)
                tts.assert_not_called()

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
                    synthesizer.speak_text_async.assert_called_once_with("Hello")
                    self.assertEqual(online.call_count, int(not use_container))
                    if use_container:
                        container.assert_called_once_with(voice)
                    else:
                        container.assert_not_called()

    def test_long_text_is_preserved_in_bounded_chunks(self):
        for text in (
            "## Heading\n\n" + ("A sentence about audio sampling. " * 120),
            "a" * 2501,
            (" " * 1200) + "Text after leading whitespace.",
        ):
            with self.subTest(text=text[:30]):
                chunks = app._split_tts_text(text)
                self.assertEqual("".join(chunks), text)
                self.assertTrue(all(0 < len(chunk) <= 1000 for chunk in chunks))

    def test_long_synthesis_joins_pcm_in_order_and_rejects_partial_results(self):
        text = "Sentence about audio. " * 130
        chunks = app._split_tts_text(text)
        results = []
        for index in range(len(chunks)):
            wav = io.BytesIO()
            with wave.open(wav, "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                audio.writeframes(np.array([index, -index], dtype="<i2").tobytes())
            results.append(SimpleNamespace(
                reason=app.speechsdk.ResultReason.SynthesizingAudioCompleted,
                audio_data=wav.getvalue(),
            ))
        for fail in (False, True):
            with self.subTest(fail=fail):
                responses = list(results)
                if fail:
                    responses[1] = SimpleNamespace(
                        reason=app.speechsdk.ResultReason.Canceled,
                        cancellation_details=SimpleNamespace(
                            error_details="Internal server error", reason="Error"
                        ),
                    )
                synthesizer = Mock()
                synthesizer.speak_text_async.return_value.get.side_effect = responses
                with (
                    patch.object(app, "_speech_config_container_tts", return_value=Mock()),
                    patch.object(app.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
                ):
                    if fail:
                        with self.assertRaisesRegex(gr.Error, "part 2/.*Internal server error"):
                            app.synthesize_speech(text, app.DEFAULT_VOICE, True)
                        self.assertEqual(synthesizer.speak_text_async.call_count, 2)
                    else:
                        rate, samples = app.synthesize_speech(text, app.DEFAULT_VOICE, True)
                        self.assertEqual(rate, 24000)
                        self.assertEqual(
                            samples.tolist(),
                            [value for index in range(len(chunks)) for value in (index, -index)],
                        )
                        sent = [call.args[0] for call in synthesizer.speak_text_async.call_args_list]
                        self.assertEqual("".join(sent), text)
                        self.assertTrue(all(len(chunk) <= 1000 for chunk in sent))

    def test_synthesis_rejects_empty_or_inconsistent_audio(self):
        for rate, frames in ((24000, b""), (16000, b"\x00\x00")):
            with self.subTest(rate=rate, frames=frames):
                wav = io.BytesIO()
                with wave.open(wav, "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(rate)
                    audio.writeframes(frames)
                synthesizer = Mock()
                synthesizer.speak_text_async.return_value.get.return_value = SimpleNamespace(
                    reason=app.speechsdk.ResultReason.SynthesizingAudioCompleted,
                    audio_data=wav.getvalue(),
                )
                with (
                    patch.object(app, "_speech_config", return_value=Mock()),
                    patch.object(app.speechsdk, "SpeechSynthesizer", return_value=synthesizer),
                ):
                    with self.assertRaises(gr.Error):
                        app.synthesize_speech("Hello", app.DEFAULT_VOICE)

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

    def test_failed_recognition_surfaces_without_returning_clear_updates(self):
        with patch.object(app, "transcribe_audio", side_effect=gr.Error("No speech")):
            with self.assertRaisesRegex(gr.Error, "No speech"):
                app.send_to_stt("recording.wav", "en-US")

    def test_missing_recording_is_reported(self):
        with self.assertRaises(gr.Error):
            app.transcribe_audio("", "en-US")

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

    def test_ui_routes_send_to_stt_and_clears_recording_on_success(self):
        ui = app.build_ui()
        send = next(fn for fn in ui.fns.values() if fn.fn is app.send_to_stt)
        self.assertEqual(len(send.inputs), 3)
        self.assertEqual(len(send.outputs), 3)
        self.assertIsInstance(send.outputs[0], gr.Textbox)
        audio = send.inputs[0]
        self.assertIs(send.outputs[1], audio)
        self.assertIn("microphone", audio.sources)
        self.assertEqual(audio.format, "wav")


if __name__ == "__main__":
    unittest.main()
