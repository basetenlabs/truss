import asyncio
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from loguru import logger

# Reconfigure loguru: only show INFO+ (suppresses pipecat DEBUG spam)
logger.remove()
logger.add(sys.stderr, level="INFO")

# My test comment

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    OutputTransportMessageUrgentFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: F401
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    UserTurnStrategies,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.runner.types import RunnerArguments


def _resolve_project_root() -> Path:
    """Find the bundle root for config paths such as ${PROJECT_ROOT}/data/voices."""
    module_path = Path(__file__).resolve()
    candidates = [module_path.parent, *module_path.parents, Path("/packages"), Path("/app")]

    for candidate in candidates:
        if (candidate / "data" / "voices").is_dir():
            return candidate

    return module_path.parent


class FastVADTurnStopStrategy(BaseUserTurnStopStrategy):
    """Triggers end-of-turn as soon as VAD stops AND a transcript exists.

    Skips the default LocalSmartTurnAnalyzerV3 ML model (~0.86s) in favor
    of instant VAD+transcript gating.
    """

    def __init__(self, on_turn_stopped=None, **kwargs):
        super().__init__(**kwargs)
        self._has_transcript = False
        self._vad_stopped = False
        self._last_transcript = ""
        self._on_turn_stopped = on_turn_stopped

    async def reset(self):
        await super().reset()
        self._has_transcript = False
        self._vad_stopped = False
        self._last_transcript = ""

    async def process_frame(self, frame):
        await super().process_frame(frame)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._has_transcript = False
            self._vad_stopped = False
            self._last_transcript = ""
        elif isinstance(frame, TranscriptionFrame):
            self._has_transcript = True
            self._last_transcript = frame.text
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._vad_stopped = True

        if self._has_transcript and self._vad_stopped:
            if self._on_turn_stopped:
                await self._on_turn_stopped(self._last_transcript)
            await self.trigger_user_turn_stopped()

import httpx
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

# Import your chosen providers
from pipecat.services.openai.llm import OpenAILLMService
from .baseten_qwen_tts import BasetenQwenTTSService
from .baseten_stt import BasetenSTTService
from .baseten_tts import BasetenTTSService


class FastOpenAILLMService(OpenAILLMService):
    """OpenAILLMService with HTTP/2 and persistent connections."""

    _profiler = None

    def create_client(self, api_key=None, base_url=None, **kwargs):
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                http2=True,
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=20,
                    keepalive_expiry=None,
                ),
            ),
        )

    async def process_frame(self, frame, direction):
        fname = type(frame).__name__
        if ("Context" in fname or fname == "LLMRunFrame") and self._profiler:
            self._profiler.mark_llm_recv()
        await super().process_frame(frame, direction)

from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


async def _ping_endpoint(session: aiohttp.ClientSession, label: str, url: str, n: int = 3) -> dict:
    """Measure network RTT to a host via HTTPS HEAD requests."""
    parsed = urlparse(url)
    scheme = "https" if parsed.scheme in ("https", "wss") else "http"
    ping_url = f"{scheme}://{parsed.hostname}/"

    rtts = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            async with session.head(ping_url, timeout=aiohttp.ClientTimeout(total=5)):
                pass
        except Exception:
            pass
        rtts.append((time.perf_counter() - t0) * 1000)

    avg = sum(rtts) / len(rtts) if rtts else 0
    mn = min(rtts) if rtts else 0
    mx = max(rtts) if rtts else 0
    return {"label": label, "avg_ms": avg, "min_ms": mn, "max_ms": mx, "samples": rtts}


async def ping_models(stt_url: str, llm_url: str, tts_url: str, n: int = 3):
    """Ping all three model endpoints and log network RTT."""
    logger.info("Pinging model endpoints...")
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            _ping_endpoint(session, "STT", stt_url, n),
            _ping_endpoint(session, "LLM", llm_url, n),
            _ping_endpoint(session, "TTS", tts_url, n),
        )
    for r in results:
        logger.info(
            f"  {r['label']:>3} ping: avg={r['avg_ms']:.1f}ms  "
            f"min={r['min_ms']:.1f}ms  max={r['max_ms']:.1f}ms  ({n} samples)"
        )
    return results


async def run_bot(
    websocket,
    secrets: dict = None,
    stt_kwargs: dict = None,
    llm_kwargs: dict = None,
    tts_kwargs: dict = None,
    mcp_servers: list = None,
    handle_sigint: bool = True,
):
    """Main bot logic with STT, LLM, and TTS."""
    secrets = secrets or {}
    api_key = secrets.get("BASETEN_API_KEY", "")
    project_root = _resolve_project_root()

    await ping_models(
        stt_url=(stt_kwargs or {}).get("url", ""),
        llm_url=(llm_kwargs or {}).get("base_url", ""),
        tts_url=(tts_kwargs or {}).get("url", ""),
    )

    llm_kwargs = dict(llm_kwargs or {})
    system_prompt = llm_kwargs.pop("system_prompt", None)
    tts_kwargs = dict(tts_kwargs or {})
    tts_provider = tts_kwargs.pop("provider", "baseten_orpheus")
    for key in ("voice_audio_dir", "ref_audio"):
        value = tts_kwargs.get(key)
        if isinstance(value, str):
            tts_kwargs[key] = value.replace("${PROJECT_ROOT}", str(project_root))

    stt = BasetenSTTService(api_key=api_key, **(stt_kwargs or {}))
    llm = FastOpenAILLMService(api_key=api_key, **llm_kwargs)
    if tts_provider == "baseten_qwen":
        tts = BasetenQwenTTSService(api_key=api_key, **tts_kwargs)
    else:
        tts = BasetenTTSService(api_key=api_key, **tts_kwargs)

    # ── MCP tool discovery ─────────────────────────────────────
    from pipecat.services.mcp_service import MCPClient
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    mcp_clients = []
    all_tools = ToolsSchema(standard_tools=[])

    for server_cfg in (mcp_servers or []):
        server_type = server_cfg.get("type", "sse")
        try:
            if server_type == "sse":
                from mcp.client.session_group import SseServerParameters
                params = SseServerParameters(
                    url=server_cfg["url"],
                    headers=server_cfg.get("headers"),
                    timeout=server_cfg.get("timeout", 10),
                    sse_read_timeout=server_cfg.get("sse_read_timeout", 300),
                )
            elif server_type == "streamable_http":
                from mcp.client.session_group import StreamableHttpParameters
                params = StreamableHttpParameters(
                    url=server_cfg["url"],
                    headers=server_cfg.get("headers"),
                    timeout=server_cfg.get("timeout", 10),
                    sse_read_timeout=server_cfg.get("sse_read_timeout", 300),
                )
            elif server_type == "stdio":
                from mcp import StdioServerParameters
                params = StdioServerParameters(
                    command=server_cfg["command"],
                    args=server_cfg.get("args", []),
                    env=server_cfg.get("env"),
                )
            else:
                logger.warning(f"Unknown MCP server type: {server_type}, skipping")
                continue

            mcp_client = MCPClient(
                server_params=params,
                tools_filter=server_cfg.get("tools_filter"),
            )
            tools = await mcp_client.register_tools(llm)
            all_tools.standard_tools.extend(tools.standard_tools)
            mcp_clients.append(mcp_client)
            logger.info(f"MCP server ({server_type}) registered {len(tools.standard_tools)} tools")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server ({server_type}): {e}")

    if not all_tools.standard_tools:
        logger.info("No MCP tools registered, running without tools")
        all_tools = None

    # Profiler reference set after TurnProfiler is defined below
    
    # Set up conversation context
    default_system_prompt = (
        "You are a helpful, conversational AI speaking to a user in a live voice call. "
        "Speak naturally and keep your responses short and easy to understand."
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt or default_system_prompt,
        },
    ]

    # One-shot greeting instruction injected on connect (see on_client_connected).
    # It MUST be stripped after the greeting, otherwise it lingers in the
    # context and the model keeps re-introducing itself on every later turn.
    intro_instruction = "Please introduce yourself to the user."

    turn_boundary_queue = asyncio.Queue()

    context = LLMContext(messages, tools=all_tools) if all_tools else LLMContext(messages)

    def clear_intro_instruction():
        """Remove the one-shot bootstrap greeting instruction from the context."""
        msgs = context.get_messages()
        filtered = [
            m
            for m in msgs
            if not (
                isinstance(m, dict)
                and m.get("role") == "system"
                and m.get("content") == intro_instruction
            )
        ]
        if len(filtered) != len(msgs):
            context.set_messages(filtered)

    async def on_turn_stopped(transcript_text: str):
        # Drop the one-shot greeting instruction before the first real user
        # turn reaches the LLM, otherwise the model keeps re-introducing
        # itself instead of answering the user.
        clear_intro_instruction()
        await turn_boundary_queue.put(
            {
                "type": "turn_boundary",
                "text": transcript_text,
            }
        )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(start_secs=0.2, stop_secs=0.15, confidence=0.9)),
            user_turn_strategies=UserTurnStrategies(
                stop=[FastVADTurnStopStrategy(on_turn_stopped=on_turn_stopped)]
            ),
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )
    
    # ── Per-turn profiler ───────────────────────────────────────
    class TurnProfiler:
        """Per-turn timing. All timestamps are time.monotonic().

        STT:      vad_stopped → stt_done        (speech end to transcript)
        Turn:     stt_done → llm_start           (aggregator + pipeline overhead)
        LLM TTFT: llm_start → llm_first_token    (actual model API latency)
        TTS TTFB: llm_first_token → tts_first_audio
        E2E:      vad_stopped → tts_first_audio
        """
        def __init__(self):
            self.reset()

        def reset(self):
            self.stt_done = 0.0
            self.vad_stopped = 0.0
            self.context_ready = 0.0
            self.llm_recv = 0.0
            self.llm_start = 0.0
            self.llm_first_token = 0.0
            self.tts_first_audio = 0.0
            self._reported = False

        def mark_stt_done(self):
            self.stt_done = time.monotonic()

        def mark_vad_stopped(self):
            self.vad_stopped = time.monotonic()

        def mark_context_ready(self):
            if self.stt_done:
                self.context_ready = time.monotonic()

        def mark_llm_recv(self):
            if not self.llm_recv:
                self.llm_recv = time.monotonic()

        def mark_llm_start(self):
            self.llm_start = time.monotonic()

        def mark_llm_first_token(self):
            if not self.llm_first_token:
                self.llm_first_token = time.monotonic()

        def mark_tts_first_audio(self):
            """Returns breakdown string if this is the first audio, else None."""
            if self.tts_first_audio:
                return None
            self.tts_first_audio = time.monotonic()
            return self._get_breakdown()

        def _get_breakdown(self):
            if self._reported or not self.stt_done:
                return None
            self._reported = True

            anchor = self.vad_stopped if self.vad_stopped else self.stt_done
            stt = (self.stt_done - anchor)
            turn = (self.llm_start - self.stt_done) if self.llm_start else 0
            llm_ttft = (self.llm_first_token - self.llm_start) if (self.llm_first_token and self.llm_start) else 0
            tts_ttfb = (self.tts_first_audio - self.llm_first_token) if (self.tts_first_audio and self.llm_first_token) else 0
            e2e = (self.tts_first_audio - anchor) if self.tts_first_audio else 0

            msg = (
                f"[PROFILE] "
                f"STT: {stt:.3f}s | "
                f"Turn: {turn:.3f}s | "
                f"LLM TTFT: {llm_ttft:.3f}s | "
                f"TTS TTFB: {tts_ttfb:.3f}s | "
                f"E2E: {e2e:.3f}s"
            )
            logger.info(msg)
            return msg

    profiler = TurnProfiler()
    llm._profiler = profiler

    # Queue for forwarding TranscriptionFrame to client (for TTFA measurement)
    transcription_to_client_queue = asyncio.Queue()

    # After STT: log transcripts, forward to client, record timing
    class TranscriptLogger(FrameProcessor):
        def __init__(self, forward_queue, profiler):
            super().__init__()
            self._audio_count = 0
            self._forward_queue = forward_queue
            self._profiler = profiler

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, AudioRawFrame):
                self._audio_count += 1
                if self._audio_count == 1:
                    logger.info(f"Receiving audio from client ({len(frame.audio)} bytes, rate={frame.sample_rate})")
                elif self._audio_count % 100 == 0:
                    logger.info(f"Audio frames received: {self._audio_count}")
            elif isinstance(frame, TranscriptionFrame):
                self._profiler.mark_stt_done()
                self._forward_queue.put_nowait(frame)
                logger.info(f"STT: {frame.text}")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                self._profiler.mark_vad_stopped()
            await self.push_frame(frame, direction)

    transcript_logger = TranscriptLogger(transcription_to_client_queue, profiler)

    # Queue for forwarding bot response text to client
    bot_text_to_client_queue = asyncio.Queue()

    # After LLM: log text, record LLM timing, queue bot text for client
    class LLMResponseLogger(FrameProcessor):
        def __init__(self, profiler, bot_text_queue):
            super().__init__()
            self._current_response = ""
            self._profiler = profiler
            self._bot_text_queue = bot_text_queue

        def _clear_bot_queue(self):
            while not self._bot_text_queue.empty():
                try:
                    self._bot_text_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMFullResponseStartFrame):
                self._profiler.mark_llm_start()
            elif isinstance(frame, TextFrame):
                self._profiler.mark_llm_first_token()
                self._current_response += frame.text
                if frame.text and frame.text[-1] in ".!?\n":
                    if self._current_response.strip():
                        text = self._current_response.strip()
                        logger.info(f"LLM: {text}")
                        self._bot_text_queue.put_nowait(text)
                        self._current_response = ""
            elif isinstance(frame, LLMFullResponseEndFrame):
                if self._current_response.strip():
                    text = self._current_response.strip()
                    logger.info(f"LLM: {text}")
                    self._bot_text_queue.put_nowait(text)
                    self._current_response = ""
            elif isinstance(frame, InterruptionFrame):
                self._current_response = ""
                self._clear_bot_queue()
            await self.push_frame(frame, direction)

    llm_response_logger = LLMResponseLogger(profiler, bot_text_to_client_queue)

    class ToolCallForwarder(FrameProcessor):
        """Intercepts function-call frames and forwards them to the client."""

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, FunctionCallInProgressFrame):
                args = frame.arguments
                if not isinstance(args, (dict, list, str)):
                    args = str(args)
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(
                        message={
                            "type": "tool_call_start",
                            "function_name": frame.function_name,
                            "tool_call_id": frame.tool_call_id,
                            "arguments": args,
                        }
                    ),
                    direction,
                )
            elif isinstance(frame, FunctionCallResultFrame):
                args = frame.arguments
                if not isinstance(args, (dict, list, str)):
                    args = str(args)
                result = frame.result
                if not isinstance(result, (dict, list, str)):
                    result = str(result)
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(
                        message={
                            "type": "tool_call_result",
                            "function_name": frame.function_name,
                            "tool_call_id": frame.tool_call_id,
                            "arguments": args,
                            "result": result,
                        }
                    ),
                    direction,
                )
            await self.push_frame(frame, direction)

    tool_call_forwarder = ToolCallForwarder()

    class TurnBoundaryForwarder(FrameProcessor):
        def __init__(self, turn_queue):
            super().__init__()
            self._turn_queue = turn_queue

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            while not self._turn_queue.empty():
                try:
                    turn = self._turn_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message=turn),
                    direction,
                )
            await self.push_frame(frame, direction)

    turn_boundary_forwarder = TurnBoundaryForwarder(turn_boundary_queue)

    # Between user_aggregator and LLM: detect when context exits the aggregator
    class PreLLMProbe(FrameProcessor):
        def __init__(self, profiler):
            super().__init__()
            self._profiler = profiler
            self._seen_types = set()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            fname = type(frame).__name__
            if not isinstance(frame, AudioRawFrame) and fname not in self._seen_types:
                self._seen_types.add(fname)
                logger.info(f"[PreLLM] frame: {fname}")
            if "Context" in fname or fname == "LLMRunFrame":
                self._profiler.mark_context_ready()
            await self.push_frame(frame, direction)

    pre_llm_probe = PreLLMProbe(profiler)

    # After TTS: detect first audio, log breakdown, forward transcript + profile + bot text to client
    # Uses OutputTransportMessageUrgentFrame so the transport actually sends them over the WebSocket
    # (the transport drops raw TextFrame/TranscriptionFrame — only audio and message frames are sent).
    class PostTTSProfiler(FrameProcessor):
        def __init__(self, profiler, transcript_queue, bot_text_queue):
            super().__init__()
            self._profiler = profiler
            self._transcript_queue = transcript_queue
            self._bot_text_queue = bot_text_queue

        def _drain_queue(self, queue):
            items = []
            while not queue.empty():
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return items

        def _clear_queue(self, queue):
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        async def _send_client_message(self, msg_type, text, direction):
            await self.push_frame(
                OutputTransportMessageUrgentFrame(
                    message={"type": msg_type, "text": text}
                ),
                direction,
            )

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TTSAudioRawFrame):
                breakdown = self._profiler.mark_tts_first_audio()
                if breakdown:
                    await self._send_client_message("profile", breakdown, direction)
                for tf in self._drain_queue(self._transcript_queue):
                    await self._send_client_message("transcript", tf.text, direction)
                for text in self._drain_queue(self._bot_text_queue):
                    await self._send_client_message("bot_text", text, direction)
            elif isinstance(frame, LLMFullResponseEndFrame):
                for text in self._drain_queue(self._bot_text_queue):
                    await self._send_client_message("bot_text", text, direction)
                self._profiler.reset()
            elif isinstance(frame, TTSStoppedFrame):
                await self._send_client_message("agent_turn_done", "", direction)
            elif isinstance(frame, InterruptionFrame):
                self._clear_queue(self._transcript_queue)
                self._clear_queue(self._bot_text_queue)
                self._profiler.reset()

            await self.push_frame(frame, direction)

    output_transport = transport.output()
    post_tts_profiler = PostTTSProfiler(profiler, transcription_to_client_queue, bot_text_to_client_queue)

    pipeline = Pipeline([
        transport.input(),     # WebSocket input (VAD emits speaking frames)
        stt,                   # Speech-to-Text
        transcript_logger,     # Log STT transcriptions + timing
        user_aggregator,       # Turn management, interruptions, context building
        turn_boundary_forwarder,  # Forward explicit user turn-stop markers
        pre_llm_probe,         # Timestamp when context exits aggregator
        llm,                   # Language Model
        llm_response_logger,   # Log LLM responses + timing
        tool_call_forwarder,   # Forward tool call events to client
        tts,                   # Text-to-Speech
        assistant_aggregator,  # Aggregate assistant text and clear on interruptions
        post_tts_profiler,     # Profile TTS + forward transcription
        output_transport,      # WebSocket output
    ])
    
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    # Client connection handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Kick off the first reply explicitly through the context pipeline.
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "system",
                            "content": intro_instruction,
                        }
                    ]
                ),
                LLMRunFrame(),
            ]
        )
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()
    
    # Run the bot
    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point compatible with Pipecat Cloud."""
    await run_bot(
        runner_args.websocket,
        handle_sigint=runner_args.handle_sigint,
    )


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()