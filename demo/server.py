import os
import io
import time
import json
import threading
import uuid
import torch
import uvicorn
import logging
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Literal, Iterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel
from huggingface_hub import snapshot_download

# --- 1. Environment Configuration ---
# Accept either HF cache root (e.g. ~/.cache/huggingface) or hub dir (.../huggingface/hub).
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
HF_CACHE_HOME = (
    os.getenv("VIBEVOICE_HF_CACHE")
    or os.getenv("HF_HOME")
    or os.getenv("HF_HUB_CACHE")
    or DEFAULT_HF_CACHE
)
hf_cache_path = Path(HF_CACHE_HOME).expanduser()
if hf_cache_path.name == "hub":
    hf_cache_path = hf_cache_path.parent
HF_CACHE_HOME = str(hf_cache_path)

os.environ["HF_HOME"] = HF_CACHE_HOME
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --- Import VibeVoice Modules ---
try:
    from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
    from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor
except ImportError:
    print("Error: Could not import 'vibevoice'. Run this from the VibeVoice project root.")
    exit(1)

# --- Configuration ---
MODEL_ID = os.getenv("VIBEVOICE_MODEL", "microsoft/VibeVoice-ASR")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PORT = 8000

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VibeVoice-Server")

class VibeVoiceEngine:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self.lock = asyncio.Lock()
        self.model_input_device = torch.device("cpu")

    def _to_torch_device(self, value) -> Optional[torch.device]:
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str):
            if value.startswith("cuda") or value == "cpu":
                return torch.device(value)
        return None

    def _pick_input_device_from_hf_map(self, hf_device_map: dict) -> Optional[torch.device]:
        # Prefer device hosting speech encoders/tokenizers.
        preferred_keys = [
            "model.acoustic_tokenizer",
            "model.semantic_tokenizer",
            "acoustic_tokenizer",
            "semantic_tokenizer",
            "model.speech",
            "speech",
        ]
        for key_hint in preferred_keys:
            for module_name, device_value in hf_device_map.items():
                if key_hint in module_name:
                    device = self._to_torch_device(device_value)
                    if device is not None and device.type == "cuda":
                        return device

        for device_value in hf_device_map.values():
            device = self._to_torch_device(device_value)
            if device is not None and device.type == "cuda":
                return device
        return None

    def _is_cross_device_index_error(self, exc: Exception) -> bool:
        return "indices should be either on cpu or on the same device as the indexed tensor" in str(exc)

    def _is_input_weight_device_mismatch_error(self, exc: Exception) -> bool:
        msg = str(exc)
        return "Input type (" in msg and "weight type (" in msg and "should be the same" in msg

    def _alternate_cuda_device(self, current: torch.device) -> Optional[torch.device]:
        if current.type != "cuda":
            return None
        count = torch.cuda.device_count()
        if count <= 1:
            return None
        idx = current.index if current.index is not None else 0
        return torch.device(f"cuda:{(idx + 1) % count}")

    def _get_speech_modules(self) -> list[torch.nn.Module]:
        modules: list[torch.nn.Module] = []
        candidates = []
        model_root = getattr(self.model, "model", None)
        if model_root is not None:
            candidates.extend([
                getattr(model_root, "acoustic_tokenizer", None),
                getattr(model_root, "semantic_tokenizer", None),
            ])
        candidates.extend([
            getattr(self.model, "acoustic_tokenizer", None),
            getattr(self.model, "semantic_tokenizer", None),
        ])
        seen_ids = set()
        for module in candidates:
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen_ids:
                continue
            seen_ids.add(module_id)
            modules.append(module)
        return modules

    def _move_speech_modules(self, target_device: torch.device) -> bool:
        moved = False
        for module in self._get_speech_modules():
            module.to(target_device)
            moved = True
        return moved

    def _retry_with_speech_modules_on_cuda(self, audio, sr, gen_config):
        candidate_devices: list[torch.device] = []
        if self.model_input_device.type == "cuda":
            candidate_devices.append(self.model_input_device)
        for idx in range(torch.cuda.device_count()):
            dev = torch.device(f"cuda:{idx}")
            if all(dev != existing for existing in candidate_devices):
                candidate_devices.append(dev)

        last_exc: Optional[Exception] = None
        for device in candidate_devices:
            try:
                logger.warning(f"Retrying with speech modules and inputs on {device}.")
                self._move_speech_modules(device)
                self.model_input_device = device
                inputs = self._prepare_inputs(audio, sr)
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, **gen_config)
                return output_ids, inputs
            except RuntimeError as exc:
                if self._is_cross_device_index_error(exc) or self._is_input_weight_device_mismatch_error(exc):
                    last_exc = exc
                    continue
                raise

        raise RuntimeError(
            "Unable to align speech module/input devices across available CUDA devices."
        ) from last_exc

    def _prepare_inputs(self, audio, sr):
        # Newer VibeVoice processor builds expect `audio=[np.ndarray]`;
        # older ones may still accept `[(audio, sr)]`. Try modern first.
        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1).astype(np.float32)
        audio_array = np.ascontiguousarray(audio_array)

        common_kwargs = {
            "sampling_rate": int(sr),
            "return_tensors": "pt",
            "padding": True,
            "add_generation_prompt": True,
        }
        try:
            batch = self.processor(audio=[audio_array], **common_kwargs)
        except Exception as primary_exc:
            logger.warning(
                f"Processor audio=[array] failed ({primary_exc}). Trying legacy tuple format."
            )
            batch = self.processor(audio=[(audio_array, int(sr))], **common_kwargs)

        return batch.to(self.model_input_device)

    def _build_gen_config(self, temperature: float):
        gen_config = {
            "max_new_tokens": 2048,
            "pad_token_id": self.processor.pad_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_config["temperature"] = temperature
        return gen_config

    def _fallback_stream_chunks(self, text: str, target_chunk_size: int = 24) -> Iterator[str]:
        words = text.split()
        if not words:
            return
        chunk = ""
        for word in words:
            candidate = f"{chunk} {word}".strip() if chunk else word
            if len(candidate) >= target_chunk_size and chunk:
                yield chunk + " "
                chunk = word
            else:
                chunk = candidate
        if chunk:
            yield chunk

    def _resolve_model_path(self) -> str:
        # Support a direct local model path, in addition to HF repo IDs.
        model_path = Path(self.model_id).expanduser()
        if model_path.exists():
            return str(model_path.resolve())

        cache_home = Path(HF_CACHE_HOME)
        cache_dirs = [cache_home, cache_home / "hub"]
        checked_dirs: list[Path] = []
        for cache_dir in cache_dirs:
            if cache_dir not in checked_dirs:
                checked_dirs.append(cache_dir)

        last_error: Optional[Exception] = None
        for cache_dir in checked_dirs:
            try:
                return snapshot_download(
                    repo_id=self.model_id,
                    cache_dir=str(cache_dir),
                    local_files_only=True,
                )
            except Exception as exc:
                last_error = exc

        # Fallback: manually find the latest downloaded snapshot in the cache.
        repo_cache_name = f"models--{self.model_id.replace('/', '--')}"
        snapshot_candidates: list[Path] = []
        for cache_dir in checked_dirs:
            snapshots_dir = cache_dir / repo_cache_name / "snapshots"
            if snapshots_dir.exists():
                snapshot_candidates.extend(p for p in snapshots_dir.iterdir() if p.is_dir())

        if snapshot_candidates:
            latest_snapshot = max(snapshot_candidates, key=lambda p: p.stat().st_mtime)
            return str(latest_snapshot.resolve())

        raise RuntimeError(
            f"Could not find '{self.model_id}' in local HF cache. "
            f"Checked: {', '.join(str(p) for p in checked_dirs)}. "
            "Set VIBEVOICE_HF_CACHE to your Hugging Face cache root or pass a direct model path."
        ) from last_error

    def load(self):
        logger.info(f"Searching for {self.model_id} in HF cache home: {HF_CACHE_HOME}...")
        
        try:
            resolved_path = self._resolve_model_path()
            logger.info(f"✅ Success! Found model at: {resolved_path}")
        except Exception as e:
            logger.error(f"❌ Failed to find model in local cache. Error: {e}")
            raise

        # Precision settings
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn_impl = "flash_attention_2" if self.device == "cuda" else "sdpa"
        split_model = os.getenv("VIBEVOICE_SPLIT_MODEL", "1") == "1"
        gpu_count = torch.cuda.device_count() if self.device == "cuda" else 0
        use_sharded_load = split_model and gpu_count > 1
        device_map_policy = os.getenv("VIBEVOICE_DEVICE_MAP", "balanced")
        enforce_no_offload = os.getenv("VIBEVOICE_NO_OFFLOAD", "1") == "1"

        # Keep some VRAM headroom for activations and runtime allocations.
        max_memory = None
        if use_sharded_load:
            mem_ratio = float(os.getenv("VIBEVOICE_GPU_MEM_RATIO", "0.98"))
            reserve_mib = int(os.getenv("VIBEVOICE_GPU_RESERVE_MIB", "128"))
            max_memory = {}
            for idx in range(gpu_count):
                total_bytes = torch.cuda.get_device_properties(idx).total_memory
                total_mib = total_bytes // (1024 ** 2)
                alloc_mib = max(1024, int(total_mib * mem_ratio) - reserve_mib)
                max_memory[idx] = f"{alloc_mib}MiB"

            # Default offload is disabled because some VibeVoice modules read
            # parameters directly (outside module forward hooks), which can
            # trip on meta tensors when CPU offload is active.
            cpu_offload_gib = int(os.getenv("VIBEVOICE_CPU_OFFLOAD_GIB", "0"))
            if cpu_offload_gib > 0:
                max_memory["cpu"] = f"{cpu_offload_gib}GiB"
                logger.warning(
                    "CPU offload is enabled; if you hit 'Tensor on device meta' "
                    "during inference, set VIBEVOICE_CPU_OFFLOAD_GIB=0."
                )

            logger.info(
                f"Using multi-GPU sharded load across {gpu_count} GPUs "
                f"with device_map={device_map_policy} and max_memory={max_memory}"
            )
        elif self.device == "cuda":
            logger.info("Using single-GPU load on cuda:0")

        try:
            logger.info("Initializing Processor...")
            self.processor = VibeVoiceASRProcessor.from_pretrained(
                resolved_path,
                language_model_pretrained_name="Qwen/Qwen2.5-7B"
            )
            
            logger.info("Initializing Model...")
            load_kwargs = {
                "attn_implementation": attn_impl,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if self.device == "cuda":
                if use_sharded_load:
                    load_kwargs["device_map"] = device_map_policy
                    load_kwargs["max_memory"] = max_memory
                    if enforce_no_offload:
                        # Avoid meta/disk/CPU offload with custom VibeVoice layers.
                        load_kwargs["offload_folder"] = None
                        load_kwargs["offload_state_dict"] = False
                else:
                    load_kwargs["device_map"] = "cuda:0"

            try:
                self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    resolved_path,
                    dtype=dtype,
                    **load_kwargs,
                )
            except TypeError:
                # Backward compatibility for older transformers.
                self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                    resolved_path,
                    torch_dtype=dtype,
                    **load_kwargs,
                )

            if use_sharded_load and enforce_no_offload:
                meta_params = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
                if meta_params:
                    logger.warning(
                        f"Detected {len(meta_params)} meta parameters after load with "
                        f"device_map={device_map_policy}. Retrying with strict balanced placement."
                    )
                    del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    retry_kwargs = dict(load_kwargs)
                    retry_kwargs["device_map"] = "balanced"
                    retry_kwargs.pop("max_memory", None)
                    retry_kwargs["offload_folder"] = None
                    retry_kwargs["offload_state_dict"] = False

                    try:
                        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                            resolved_path,
                            dtype=dtype,
                            **retry_kwargs,
                        )
                    except TypeError:
                        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                            resolved_path,
                            torch_dtype=dtype,
                            **retry_kwargs,
                        )

                    meta_params = [n for n, p in self.model.named_parameters() if p.device.type == "meta"]
                    if meta_params:
                        preview = ", ".join(meta_params[:5])
                        raise RuntimeError(
                            "Model still has meta tensors after strict GPU-only load. "
                            "This build cannot run with offload for VibeVoice custom layers. "
                            f"Example meta params: {preview}. "
                            "Use bigger VRAM, quantization, or vLLM server for this model."
                        )

            if self.device == "cuda" and not use_sharded_load:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            if use_sharded_load and hasattr(self.model, "hf_device_map"):
                override_input_device = os.getenv("VIBEVOICE_INPUT_DEVICE")
                if override_input_device:
                    self.model_input_device = torch.device(override_input_device)
                    logger.info(
                        f"Model sharded. Using VIBEVOICE_INPUT_DEVICE override: {self.model_input_device}."
                    )
                else:
                    picked_device = self._pick_input_device_from_hf_map(self.model.hf_device_map)
                    self.model_input_device = picked_device or torch.device("cuda:0")
                    logger.info(
                        f"Model sharded. Input tensors will be sent to {self.model_input_device} "
                        "based on hf_device_map."
                    )
            else:
                self.model_input_device = torch.device(self.device if self.device == "cuda" else "cpu")

            # Optional override to co-locate speech tokenizers and speech masks.
            speech_device_override = os.getenv("VIBEVOICE_SPEECH_DEVICE")
            if speech_device_override:
                speech_device = torch.device(speech_device_override)
                try:
                    if self._move_speech_modules(speech_device):
                        self.model_input_device = speech_device
                        logger.info(
                            f"Moved speech tokenizer modules to {speech_device} and "
                            f"set model_input_device={self.model_input_device}."
                        )
                except Exception as exc:
                    logger.warning(f"Failed moving speech tokenizer modules to {speech_device}: {exc}")
            logger.info("🚀 VibeVoice is ready for inference.")
        except Exception as e:
            logger.error(f"Inference engine startup failed: {e}")
            raise e

    def predict(self, audio, sr, temperature=0.0):
        inputs = self._prepare_inputs(audio, sr)
        gen_config = self._build_gen_config(temperature)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_config)
        except RuntimeError as exc:
            if "Tensor on device meta" in str(exc):
                raise RuntimeError(
                    "Model contains meta tensors at runtime (usually from CPU offload). "
                    "Set VIBEVOICE_CPU_OFFLOAD_GIB=0 and retry, or increase "
                    "VIBEVOICE_GPU_MEM_RATIO / reduce VIBEVOICE_GPU_RESERVE_MIB."
                ) from exc
            if self._is_cross_device_index_error(exc):
                original_exc = exc

                # Attempt 1: same model, alternate input GPU.
                alt_device = self._alternate_cuda_device(self.model_input_device)
                if alt_device is not None and alt_device != self.model_input_device:
                    logger.warning(
                        f"Cross-device index mismatch on {self.model_input_device}; "
                        f"retrying with input tensors on {alt_device}."
                    )
                    try:
                        self.model_input_device = alt_device
                        inputs = self._prepare_inputs(audio, sr)
                        with torch.no_grad():
                            output_ids = self.model.generate(**inputs, **gen_config)
                        original_exc = None
                    except RuntimeError as alt_exc:
                        if not self._is_cross_device_index_error(alt_exc):
                            raise
                        original_exc = alt_exc

                # Attempt 2: align speech tokenizer modules to current input device.
                if original_exc is not None:
                    try_align = os.getenv("VIBEVOICE_TRY_ALIGN_SPEECH_TO_INPUT", "1") == "1"
                    if try_align:
                        logger.warning(
                            f"Retrying by moving speech tokenizer modules to {self.model_input_device}."
                        )
                        try:
                            if self._move_speech_modules(self.model_input_device):
                                inputs = self._prepare_inputs(audio, sr)
                                with torch.no_grad():
                                    output_ids = self.model.generate(**inputs, **gen_config)
                                original_exc = None
                            else:
                                logger.warning("No speech tokenizer modules found to move.")
                        except RuntimeError as align_exc:
                            if not self._is_cross_device_index_error(align_exc):
                                raise
                            original_exc = align_exc

                # Attempt 3: move speech path and inputs to CPU.
                if original_exc is not None:
                    cpu_fallback_enabled = os.getenv("VIBEVOICE_TRY_CPU_SPEECH_FALLBACK", "1") == "1"
                    if cpu_fallback_enabled:
                        logger.warning(
                            "Cross-device indexing persists. Trying CPU speech-module fallback "
                            "(set VIBEVOICE_TRY_CPU_SPEECH_FALLBACK=0 to disable)."
                        )
                        try:
                            moved = self._move_speech_modules(torch.device("cpu"))
                            if moved:
                                self.model_input_device = torch.device("cpu")
                                inputs = self._prepare_inputs(audio, sr)
                                with torch.no_grad():
                                    output_ids = self.model.generate(**inputs, **gen_config)
                                original_exc = None
                            else:
                                raise RuntimeError("No speech tokenizer modules found to move.")
                        except RuntimeError as cpu_exc:
                            if not self._is_cross_device_index_error(cpu_exc):
                                raise
                            original_exc = cpu_exc

                if original_exc is not None:
                    raise RuntimeError(
                        "Cross-device indexing error in sharded mode after retries. "
                        "Set VIBEVOICE_INPUT_DEVICE to the speech shard (cuda:0/cuda:1), or "
                        "force VIBEVOICE_SPEECH_DEVICE=cpu and VIBEVOICE_INPUT_DEVICE=cpu."
                    ) from original_exc
            elif self._is_input_weight_device_mismatch_error(exc):
                output_ids, inputs = self._retry_with_speech_modules_on_cuda(audio, sr, gen_config)
            else:
                raise

        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0, input_len:]
        
        text = self.processor.decode(generated_ids, skip_special_tokens=True)
        try:
            segments = self.processor.post_process_transcription(text)
        except:
            segments = []
        return text, segments

    def predict_stream(self, audio, sr, temperature=0.0) -> Iterator[str]:
        inputs = self._prepare_inputs(audio, sr)
        gen_config = self._build_gen_config(temperature)
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            text, _ = self.predict(audio, sr, temperature)
            yield from self._fallback_stream_chunks(text)
            return

        try:
            from transformers import TextIteratorStreamer
        except Exception:
            text, _ = self.predict(audio, sr, temperature)
            yield from self._fallback_stream_chunks(text)
            return

        streamer = TextIteratorStreamer(
            tokenizer=tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=1.0,
        )
        gen_error: list[Exception] = []

        def _worker():
            try:
                with torch.no_grad():
                    self.model.generate(**inputs, **gen_config, streamer=streamer)
            except RuntimeError as exc:
                if "Tensor on device meta" in str(exc):
                    exc = RuntimeError(
                        "Model contains meta tensors at runtime (usually from CPU offload). "
                        "Set VIBEVOICE_CPU_OFFLOAD_GIB=0 and retry, or increase "
                        "VIBEVOICE_GPU_MEM_RATIO / reduce VIBEVOICE_GPU_RESERVE_MIB."
                    )
                gen_error.append(exc)
            except Exception as exc:
                gen_error.append(exc)

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        except Exception:
            # If the streamer times out or fails, prefer surfacing the model error if any.
            if gen_error:
                raise gen_error[0]
            raise
        finally:
            worker.join()

        if gen_error:
            raise gen_error[0]

# --- FastAPI Integration ---

engine = VibeVoiceEngine(MODEL_ID, DEVICE)

@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="VibeVoice ASR OpenAI API", lifespan=lifespan)

def sse(data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"

async def _transcribe_impl(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
):
    content = await file.read()
    with io.BytesIO(content) as buf:
        audio, sr = sf.read(buf)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    accept_header = request.headers.get("accept", "")
    wants_stream = stream or ("text/event-stream" in accept_header.lower())

    if wants_stream:
        async def event_generator():
            queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
            loop = asyncio.get_running_loop()
            full_text = ""
            failed = False
            event_id = f"asr-{uuid.uuid4().hex}"
            created = int(time.time())
            model_name = "vibevoice"

            def _worker():
                def _push(kind: str, payload: str):
                    asyncio.run_coroutine_threadsafe(queue.put((kind, payload)), loop).result()

                try:
                    for chunk in engine.predict_stream(audio, sr, temperature):
                        _push("chunk", chunk)
                except Exception as exc:
                    _push("error", str(exc))
                finally:
                    _push("done", "")

            async with engine.lock:
                worker = threading.Thread(target=_worker, daemon=True)
                worker.start()
                # Immediate signal so clients know stream started.
                yield sse({
                    "id": event_id,
                    "object": "transcription.chunk",
                    "created": created,
                    "model": model_name,
                    "type": "transcript.text.start",
                    "delta": "",
                })

                while True:
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        # Keep connection alive through proxies/load balancers.
                        yield ": ping\n\n"
                        continue

                    if kind == "chunk":
                        full_text += payload
                        yield sse({
                            "id": event_id,
                            "object": "transcription.chunk",
                            "created": created,
                            "model": model_name,
                            "type": "transcript.text.delta",
                            "delta": payload,
                            "text": full_text,
                        })
                    elif kind == "error":
                        failed = True
                        logger.error(f"Streaming transcription failed: {payload}")
                        yield sse({
                            "id": event_id,
                            "object": "error",
                            "type": "error",
                            "message": payload,
                        })
                    elif kind == "done":
                        break

            if not failed:
                yield sse({
                    "id": event_id,
                    "object": "transcription.completed",
                    "created": int(time.time()),
                    "model": model_name,
                    "type": "transcript.text.done",
                    "text": full_text,
                })
            yield sse("[DONE]")

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async with engine.lock:
        text, segments = await asyncio.to_thread(engine.predict, audio, sr, temperature)

    if response_format == "text":
        return PlainTextResponse(text)
    
    return {"text": text}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
):
    return await _transcribe_impl(request, file, response_format, temperature, stream)

@app.post("/v1/audio/translations")
async def translate(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
):
    # Alias for OpenAI-compatible clients that call /translations first.
    return await _transcribe_impl(request, file, response_format, temperature, stream)

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "vibevoice", "object": "model"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
