import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sounddevice as sd
import numpy as np
import time
import queue
import threading
import json
import logging
import torch
import mlx.core as mx

from pathlib import Path
from collections import deque
import re
import gc
import multiprocessing as mp
from pynput import keyboard
from typing import List, Dict, Coroutine
import asyncio

# --- ASR and VLM/LLM Imports ---
from parakeet_mlx import from_pretrained
from mlx_vlm import load as gemma_vlm_load, stream_generate as gemma_vlm_generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template as gemma_vlm_apply_chat_template
from mlx_lm import load as llm_load, stream_generate as llm_stream_generate
from mlx_lm.models.cache import make_prompt_cache

# --- PIPER TTS IMPORT ---
from piper import PiperVoice, SynthesisConfig

# =================================================================
# START: BROWSER-USE INTEGRATION IMPORTS
# =================================================================
from browser_use import Agent, Controller, BrowserSession, ActionResult
from browser_use.browser.types import Page
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
# =================================================================
# END: BROWSER-USE INTEGRATION IMPORTS
# =================================================================


# =================================================================
# 0. SETUP AND CONFIGURATION
# =================================================================
# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - (%(threadName)-10s) - %(message)s')
root = logging.getLogger()
for h in root.handlers:
    h.setLevel(logging.INFO) # Set to INFO to see the new monitor logs

# --- System Prompt for Intent Routing ---
SYSTEM_PROMPT = r"""You are a highly specialized AI assistant that functions as an intent router. Your primary task is to analyze the user's prompt and respond with a single JSON object that classifies the user's intent and contains the appropriate response.

You must follow these rules strictly. Your output MUST ALWAYS be a single, valid JSON object and nothing else. Do not include any introductory text, explanations, or markdown formatting like ```json.

Rule 1: Browser Use Intent (browseruse)
Condition: If the user's prompt contains explicit instructions to navigate, open, search, or go to a website (e.g., "go to", "open", "search for", "look up on").
Action: If this condition is met, your ONLY output must be a single JSON object in the following format, containing the original user prompt:
{"intent": "browseruse", "output": "The user's original and complete prompt"}

Rule 2: General Chat Intent (generalchat)
Condition: If the user's prompt is a general question, a request for information, a statement, or any other conversational input that does not meet the criteria for the browseruse intent.
Action: If this condition is met, you must first formulate a direct, helpful answer to the user's question. Then, your ONLY output must be a single JSON object in the following format, containing that answer:
{"intent": "generalchat", "output": "Your direct and helpful answer to the user's prompt"}

Examples of Correct Output:

User Prompt: go to youtube, search for videos about gravitational waves?
Your Response:
{"intent": "browseruse", "output": "go to youtube, search for videos about gravitational waves?"}

User Prompt: What is the capital of Canada?
Your Response:
{"intent": "generalchat", "output": "The capital of Canada is Ottawa."}

Now, analyze the following user prompt and respond with the correct JSON object according to the rules above.
"""

class TermColors:
    ASSISTANT = '\033[94m'
    USER = '\033[92m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    SYSTEM = '\033[91m'

class Config:
    # --- ASR Configuration ---
    STREAMING_ASR_SAMPLE_RATE = 16000
    STREAMING_ASR_BLOCK_SIZE = 512
    ASR_MODEL_REPO = "mlx-community/parakeet-tdt_ctc-110m"

    # --- VAD Configuration ---
    VAD_CHUNK_SIZE = 512
    VAD_SAMPLE_RATE = 16000
    VAD_THRESHOLD = 0.5
    SILENCE_THRESHOLD_SECONDS = 1.0
    _VAD_CHUNKS_PER_SECOND = VAD_SAMPLE_RATE / VAD_CHUNK_SIZE
    VAD_SILENCE_THRESHOLD_CHUNKS = int(SILENCE_THRESHOLD_SECONDS * _VAD_CHUNKS_PER_SECOND)

    # --- General/Other ---
    CHANNELS = 1
    DTYPE = 'float32'

    # --- TTS Configuration ---
    PIPER_MODEL_PATH = "/Users/zx/Documents/projects/gemma-v/models/en_US-lessac-low.onnx" # CHANGE THIS PATH
    PIPER_LENGTH_SCALE = 1.0
    PIPER_NOISE_SCALE = 0.667
    PIPER_NOISE_W_SCALE = 0.8

    # --- NLU/LLM Configuration ---
    NLU_MODEL_PATH = "mlx-community/gemma-3n-E2B-it-lm-4bit"
    MAX_KV_SIZE = 4096 * 4

def common_prefix_len(list1: List, list2: List) -> int:
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] != list2[i]:
            return i
    return min_len

# =================================================================
# 1. CORE COMPONENTS
# =================================================================

class TTSProcessWorker(mp.Process):
    def __init__(self, sentence_queue: mp.JoinableQueue, config_dict: dict, interrupt_event: mp.Event, tts_is_speaking_event: mp.Event):
        super().__init__()
        self.sentence_queue = sentence_queue
        self.interrupt_event = interrupt_event
        self.tts_is_speaking_event = tts_is_speaking_event
        self.model_path = config_dict.get('PIPER_MODEL_PATH')
        self.length_scale = config_dict.get('PIPER_LENGTH_SCALE')
        self.noise_scale = config_dict.get('PIPER_NOISE_SCALE')
        self.noise_w_scale = config_dict.get('PIPER_NOISE_W_SCALE')
        logging.info(f"[TTS Process] Initialized with Piper model: '{self.model_path}'")

    def run(self):
        try:
            if not self.model_path or not Path(self.model_path).exists():
                logging.critical(f"[TTS Process] Piper model not found at path: {self.model_path}. Exiting.")
                return
            logging.info("[TTS Process] Loading Piper voice model...")
            voice = PiperVoice.load(self.model_path)
            syn_config = SynthesisConfig(length_scale=self.length_scale, noise_scale=self.noise_scale, noise_w_scale=self.noise_w_scale)
            logging.info("[TTS Process] Piper model loaded successfully.")
        except Exception as e:
            logging.critical(f"[TTS Process] Failed to load Piper model: {e}", exc_info=True)
            return

        while True:
            try:
                if self.interrupt_event.is_set():
                    while not self.sentence_queue.empty():
                        try: self.sentence_queue.get_nowait(); self.sentence_queue.task_done()
                        except queue.Empty: break
                    time.sleep(0.1)
                    continue
                sentence = self.sentence_queue.get()
                sentence = sentence.rstrip('"}')
                if sentence is None: break
                if self.interrupt_event.is_set() or not sentence.strip():
                    self.sentence_queue.task_done()
                    continue
                try:
                    self.tts_is_speaking_event.set()
                    with sd.RawOutputStream(samplerate=voice.config.sample_rate, channels=1, dtype="int16") as stream:
                        logging.info(f"[TTS Process] Synthesizing sentence: '{sentence}'")
                        for chunk in voice.synthesize(sentence, syn_config=syn_config):
                            if self.interrupt_event.is_set():
                                stream.abort(ignore_errors=True); break
                            stream.write(chunk.audio_int16_bytes)
                except Exception as e: logging.error(f"[TTS Process] Error during synthesis: {e}")
                finally: self.tts_is_speaking_event.clear()
                self.sentence_queue.task_done()
            except queue.Empty: continue
            except Exception as e:
                logging.error(f"[TTS Process] Error in run loop: {e}", exc_info=True)
                self.tts_is_speaking_event.clear()

class StreamingTTSEngine:
    def __init__(self, config: Config, tts_is_speaking_event: mp.Event):
        self.config = config; self.sentence_queue = None; self.tts_process = None
        self.interrupt_event = mp.Event(); self.tts_is_speaking_event = tts_is_speaking_event

    def _start_worker_if_needed(self):
        if self.tts_process is None or not self.tts_process.is_alive():
            logging.info("Starting new TTS worker process...")
            if self.tts_process: self.tts_process.join(timeout=0.1)
            self.sentence_queue = mp.JoinableQueue(); self.interrupt_event.clear(); self.tts_is_speaking_event.clear()
            config_dict = {k: v for k, v in self.config.__class__.__dict__.items() if not k.startswith('_') and not callable(v)}
            if not config_dict.get('PIPER_MODEL_PATH') or not Path(config_dict['PIPER_MODEL_PATH']).exists():
                logging.critical("Cannot start TTS worker: PIPER_MODEL_PATH is not set or file does not exist.")
                self.tts_process = None; return
            self.tts_process = TTSProcessWorker(self.sentence_queue, config_dict, self.interrupt_event, self.tts_is_speaking_event)
            self.tts_process.start()

    def interrupt(self):
        logging.info("[TTS Controller] INTERRUPT: Terminating TTS worker process.")
        if self.tts_process and self.tts_process.is_alive():
            self.tts_process.terminate(); self.tts_process.join(timeout=1.0)
        self.tts_process = None; self.sentence_queue = None; self.tts_is_speaking_event.clear()

    def speak(self, text_generator, barge_in_event: threading.Event, generation_done_event: threading.Event):
        self._start_worker_if_needed()
        if not self.tts_process:
            print("\nAssistant: [TTS Error]")
            for _ in text_generator: pass
            generation_done_event.set()
            return

        def llm_to_tts_feeder():
            sentence_buffer = ""
            sentence_end_pattern = re.compile(r'(?<=[.!?,\n])\s*')
            print(f"{TermColors.BOLD}{TermColors.ASSISTANT}Assistant:{TermColors.ENDC} ", end="", flush=True)

            unwanted_suffix = '"}\n```'
            print_buffer = deque()
            PRINT_BUFFER_SIZE = 10

            try:
                for token in text_generator:
                    clean_token = token.replace("<eos>", "").replace("*", "")
                    sentence_buffer += clean_token
                    print_buffer.append(clean_token)
                    if len(print_buffer) > PRINT_BUFFER_SIZE:
                        print(print_buffer.popleft(), end="", flush=True)

                    sentences = sentence_end_pattern.split(sentence_buffer)
                    if len(sentences) > 1:
                        complete, sentence_buffer = sentences[:-1], sentences[-1]
                        for s in complete:
                            if s.strip():
                                if self.sentence_queue and self.tts_process and self.tts_process.is_alive():
                                    try:
                                        self.sentence_queue.put(s.strip(), timeout=0.5)
                                    except queue.Full:
                                        logging.warning("TTS queue full, likely due to barge-in. Halting feeder.")
                                        barge_in_event.set()
                                        break
                                else:
                                    barge_in_event.set()
                                    break
                        if barge_in_event.is_set():
                            break

                if not barge_in_event.is_set():
                    final_text_to_print = "".join(print_buffer).rstrip(unwanted_suffix)
                    print(final_text_to_print, end="", flush=True)

                if not barge_in_event.is_set() and (last_sentence := sentence_buffer.strip()):
                    if self.sentence_queue and self.tts_process and self.tts_process.is_alive():
                        clean_last_sentence = last_sentence.rstrip('"}\n```')
                        self.sentence_queue.put(clean_last_sentence)

            except Exception as e:
                logging.error(f"Error in feeder: {e}", exc_info=True)
            finally:
                generation_done_event.set()
                if barge_in_event.is_set():
                    self.interrupt()
                print()
                logging.info("="*20 + " WAITING FOR NEXT COMMAND " + "="*20)

        threading.Thread(target=llm_to_tts_feeder, daemon=True, name="LLM-TTS-Feeder").start()

    def shutdown(self): self.interrupt()

    def play_greeting(self, text: str):
        self._start_worker_if_needed()
        if not self.tts_process:
            print(f"\nAssistant: {text}\n")
            return
        if self.sentence_queue:
            self.sentence_queue.put(text)

class StreamingTranscriber:
    def __init__(self, config: Config, tts_is_speaking_event: mp.Event, barge_in_event: threading.Event, video_is_playing_event: threading.Event):
        logging.info("Initializing Streaming Transcriber with Parakeet...")
        self.config = config
        self.tts_is_speaking_event = tts_is_speaking_event
        self.barge_in_event = barge_in_event
        self.video_is_playing_event = video_is_playing_event

        self._load_asr_model()
        self.transcriber = None

        logging.info("Loading Silero VAD model...")
        self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=True, verbose=False)

        self.audio_stream = None
        self.audio_queue = queue.Queue()
        self.final_transcript_queue = queue.Queue()
        self._is_running = False
        self._worker_thread = None
        self.speech_triggered = False
        self.silence_chunks_counter = 0
        self.was_tts_speaking_prev_iter = False
        self.was_video_playing_prev_iter = False

    def _load_asr_model(self):
        logging.info(f"Loading Parakeet ASR model: {self.config.ASR_MODEL_REPO}")
        try:
            self.model = from_pretrained(self.config.ASR_MODEL_REPO)
            if self.config.STREAMING_ASR_SAMPLE_RATE != self.model.preprocessor_config.sample_rate:
                 logging.warning(
                     f"Config sample rate ({self.config.STREAMING_ASR_SAMPLE_RATE}) "
                     f"mismatches model's ({self.model.preprocessor_config.sample_rate}). "
                     "Please update your Config class."
                 )
            self.model.eval()
            logging.info("Parakeet ASR model loaded successfully.")
        except Exception as e:
            logging.critical(f"Failed to load Parakeet ASR model: {e}", exc_info=True)
            self.model = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Audio callback status: {status}")
        if self._is_running:
            self.audio_queue.put(indata.copy())

    def _reset_asr_state(self):
        logging.info("ASR: Resetting state for new utterance.")
        self.speech_triggered = False
        self.silence_chunks_counter = 0
        if self.model:
            self.transcriber = self.model.transcribe_stream(context_size=(256, 256))
        else:
            self.transcriber = None

    def _transcription_worker(self):
        if not self.model:
            logging.error("ASR model not loaded. Transcription worker cannot start.")
            return

        self._reset_asr_state()
        vad_process_buffer = np.array([], dtype=np.float32)
        asr_audio_buffer = np.array([], dtype=np.float32)
        last_printed_text = ""
        ASR_FEED_CHUNK_SIZE = 16000

        while self._is_running:
            try:
                # Pause ASR if a video is playing in the browser
                if self.video_is_playing_event.is_set():
                    if not self.was_video_playing_prev_iter:
                        logging.info("ASR paused for video playback.")
                        with self.audio_queue.mutex: self.audio_queue.queue.clear()
                        vad_process_buffer = np.array([], dtype=np.float32)
                        asr_audio_buffer = np.array([], dtype=np.float32)
                        self._reset_asr_state()
                    self.was_video_playing_prev_iter = True
                    time.sleep(0.1)
                    continue
                if self.was_video_playing_prev_iter:
                    logging.info("ASR resumed after video playback.")
                self.was_video_playing_prev_iter = False

                if self.barge_in_event.is_set():
                    with self.audio_queue.mutex: self.audio_queue.queue.clear()
                    vad_process_buffer = np.array([], dtype=np.float32)
                    asr_audio_buffer = np.array([], dtype=np.float32)
                    self._reset_asr_state()
                    last_printed_text = ""
                    logging.info("ASR worker state reset due to barge-in.")
                    while self.barge_in_event.is_set():
                        time.sleep(0.1)
                    logging.info("ASR worker resuming after barge-in signal cleared.")
                    continue

                if self.tts_is_speaking_event.is_set():
                    if not self.was_tts_speaking_prev_iter: logging.info("ASR/VAD paused for TTS.")
                    self.was_tts_speaking_prev_iter = True
                    with self.audio_queue.mutex: self.audio_queue.queue.clear()
                    vad_process_buffer = np.array([], dtype=np.float32)
                    asr_audio_buffer = np.array([], dtype=np.float32)
                    time.sleep(0.1)
                    continue

                if self.was_tts_speaking_prev_iter:
                    logging.info("ASR/VAD resumed.")
                    self._reset_asr_state()
                    last_printed_text = ""
                self.was_tts_speaking_prev_iter = False

                audio_chunk = self.audio_queue.get(timeout=0.1)
                vad_process_buffer = np.concatenate([vad_process_buffer, audio_chunk[:, 0]])

                while len(vad_process_buffer) >= self.config.VAD_CHUNK_SIZE:
                    chunk_to_process = vad_process_buffer[:self.config.VAD_CHUNK_SIZE]
                    vad_process_buffer = vad_process_buffer[self.config.VAD_CHUNK_SIZE:]
                    is_speech = self.vad_model(torch.from_numpy(chunk_to_process), self.config.VAD_SAMPLE_RATE).item() > self.config.VAD_THRESHOLD

                    if is_speech:
                        if not self.speech_triggered:
                            print(f"\n{TermColors.BOLD}{TermColors.USER}User:{TermColors.ENDC} ", end="", flush=True)
                            self.speech_triggered = True
                            logging.info("VAD: Speech started.")
                        self.silence_chunks_counter = 0
                        asr_audio_buffer = np.concatenate([asr_audio_buffer, chunk_to_process])
                    elif self.speech_triggered:
                        self.silence_chunks_counter += 1
                        asr_audio_buffer = np.concatenate([asr_audio_buffer, chunk_to_process])

                    if len(asr_audio_buffer) >= ASR_FEED_CHUNK_SIZE:
                        logging.info(f"ASR buffer full ({len(asr_audio_buffer)} samples), feeding to transcriber.")
                        self.transcriber.add_audio(mx.array(asr_audio_buffer))
                        asr_audio_buffer = np.array([], dtype=np.float32)

                    if self.speech_triggered:
                        current_text = self.transcriber.result.text.strip()
                        if current_text != last_printed_text:
                            print(f"\r{TermColors.BOLD}{TermColors.USER}User:{TermColors.ENDC} {current_text}", end="", flush=True)
                            last_printed_text = current_text

                    if self.speech_triggered and self.silence_chunks_counter > self.config.VAD_SILENCE_THRESHOLD_CHUNKS:
                        logging.info("VAD: Silence threshold exceeded. Finalizing utterance.")
                        if len(asr_audio_buffer) > 0:
                            logging.info(f"Feeding final {len(asr_audio_buffer)} audio samples to transcriber.")
                            self.transcriber.add_audio(mx.array(asr_audio_buffer))
                            asr_audio_buffer = np.array([], dtype=np.float32)

                        final_text = last_printed_text.strip()

                        if final_text:
                            print()
                            logging.info(f"Final transcript: '{final_text}'")
                            self.final_transcript_queue.put(final_text)

                        self._reset_asr_state()
                        last_printed_text = ""
                        break

            except queue.Empty: continue
            except Exception as e:
                logging.error(f"Error in ASR worker: {e}", exc_info=True)
                self._reset_asr_state()
                last_printed_text = ""; vad_process_buffer = np.array([], dtype=np.float32); asr_audio_buffer = np.array([], dtype=np.float32)

    def start(self):
        if self._is_running: return
        logging.info("Starting continuous transcription...")
        self._is_running = True; self.audio_queue.queue.clear(); self.final_transcript_queue.queue.clear()
        self.audio_stream = sd.InputStream(samplerate=self.config.STREAMING_ASR_SAMPLE_RATE, blocksize=self.config.STREAMING_ASR_BLOCK_SIZE, channels=self.config.CHANNELS, dtype=self.config.DTYPE, callback=self._audio_callback)
        self.audio_stream.start()
        self._worker_thread = threading.Thread(target=self._transcription_worker, name="ASR-Worker"); self._worker_thread.start()
        logging.info("Microphone is now LIVE.")

    def stop(self):
        if not self._is_running: return
        logging.info("Stopping transcription...")
        self._is_running = False
        if self._worker_thread: self._worker_thread.join(timeout=1.0)
        if self.audio_stream: self.audio_stream.stop(); self.audio_stream.close()
        self.audio_stream = None
        logging.info("Transcription stopped.")

class GemmaChatEngine:
    def __init__(self, config: Config):
        logging.info("Initializing Gemma Chat Engine...")
        self.config = config
        model_path = config.NLU_MODEL_PATH
        self.is_llm = "-lm-" in model_path.lower()
        self.messages: List[Dict[str, str]] = []
        if self.is_llm:
            logging.info(f"Loading LLM model: {model_path}")
            self.model, self.tokenizer = llm_load(model_path)
            self.processed_tokens: List[int] = []; self.prompt_cache = None
        else:
            logging.info(f"Loading VLM model: {model_path}")
            self.model, self.processor = gemma_vlm_load(model_path)
            self.tokenizer = self.processor; self.model_config = load_config(model_path)
        self.reset()
        logging.info(f"Gemma model loaded. Cache size set to {config.MAX_KV_SIZE} tokens.")

    def reset(self):
        logging.info("Resetting conversation history and KV cache.")
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.is_llm:
            self.processed_tokens = []; self.prompt_cache = make_prompt_cache(self.model, self.config.MAX_KV_SIZE)

    def stream_intent_and_response(self, transcript: str, barge_in_event: threading.Event):
        if not transcript: return
        self.messages.append({"role": "user", "content": transcript})
        if self.is_llm:
            target_tokens = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True)
            prefix_len = common_prefix_len(self.processed_tokens, target_tokens)
            tokens_to_process = target_tokens[prefix_len:]
            logging.info(f"KV Cache: Total tokens: {len(target_tokens)}, Cached: {prefix_len}, New: {len(tokens_to_process)}")
            token_stream_generator = llm_stream_generate(self.model, self.tokenizer, tokens_to_process, max_tokens=2048, prompt_cache=self.prompt_cache)
            generated_tokens = []
            def text_generator_wrapper():
                nonlocal generated_tokens
                for response in token_stream_generator:
                    generated_tokens.append(response.token)
                    yield response.text
            token_generator = text_generator_wrapper()
        else:
            prompt = gemma_vlm_apply_chat_template(self.processor, self.model_config, self.messages)
            token_generator = gemma_vlm_generate(self.model, self.processor, prompt, max_tokens=2048)
        buffer = ""; intent = None; payload_started = False; browser_payload_parts = []; full_response_parts = []
        payload_marker_re = re.compile(r'"output"\s*:\s*"')
        try:
            for token_text in token_generator:
                if barge_in_event.is_set():
                    logging.info("Barge-in detected in Gemma Engine, halting generation."); break
                full_response_parts.append(token_text)
                if not payload_started:
                    buffer += token_text
                    if intent is None:
                        if '"intent": "generalchat"' in buffer: intent = "generalchat"; logging.info("Intent detected early: generalchat")
                        elif '"intent": "browseruse"' in buffer: intent = "browseruse"; logging.info("Intent detected early: browseruse")
                    if intent is not None:
                        match = payload_marker_re.search(buffer)
                        if match:
                            payload_started = True
                            initial_payload = buffer[match.end():]
                            if intent == "generalchat": yield initial_payload
                            elif intent == "browseruse": browser_payload_parts.append(initial_payload)
                else:
                    if intent == "generalchat": yield token_text
                    elif intent == "browseruse": browser_payload_parts.append(token_text)
        finally:
            gc.collect()
            full_response_text = "".join(full_response_parts).replace("<eos>", "").strip()
            if barge_in_event.is_set():
                logging.info("Barge-in: Not saving partial model response. Removing user prompt from history.")
                if self.messages and self.messages[-1]["role"] == "user": self.messages.pop()
            elif full_response_text:
                self.messages.append({"role": "model", "content": full_response_text})
                if self.is_llm: self.processed_tokens = target_tokens + generated_tokens
            if intent == "browseruse":
                final_payload = "".join(browser_payload_parts).replace("<eos>", "").strip().rstrip('"}\n```')
                return {"intent": "browseruse", "payload": final_payload}

class KeyPressListener:
    def __init__(self, is_replying_event: threading.Event, barge_in_event: threading.Event, video_is_playing_event: threading.Event, go_home_event: threading.Event):
        self.is_replying_event = is_replying_event
        self.barge_in_event = barge_in_event
        self.video_is_playing_event = video_is_playing_event
        self.go_home_event = go_home_event
        self.listener = None

    def _on_press(self, key):
        if key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            # If a video is playing, the Command key's job is to trigger the "Reset & Ready" action.
            if self.video_is_playing_event.is_set():
                logging.info("Command key pressed during video playback. Signaling 'Reset & Ready'.")
                print(f"\n{TermColors.SYSTEM}Reset command received. Stopping task and returning to home...{TermColors.ENDC}")
                self.go_home_event.set()

            # If the assistant is speaking (but not in a video task), Command key is a barge-in.
            elif self.is_replying_event.is_set():
                logging.info("Command key pressed during TTS. Signaling BARGE-IN.")
                print(f"\n{TermColors.SYSTEM}Barge-in activated. Silenced.{TermColors.ENDC}")
                self.barge_in_event.set()

    def start(self):
        if self.listener is None:
            self.listener = keyboard.Listener(on_press=self._on_press)
            self.listener.start()
            logging.info("Keypress listener started. Press 'Command' to interrupt.")

    def stop(self):
        if self.listener and self.listener.is_alive():
            self.listener.stop(); self.listener.join()
        self.listener = None

class GeneratorWithReturnValue:
    def __init__(self, gen):
        self.gen = gen
        self.return_value = None
    def __iter__(self):
        self.return_value = yield from self.gen

# =================================================================
# 2. BROWSER-USE INTEGRATION
# =================================================================
class BrowserController:
    def __init__(self, tts_engine: StreamingTTSEngine, video_is_playing_event: threading.Event):
        self.tts = tts_engine
        self.controller = Controller()
        self.browser_session: BrowserSession | None = None
        self.video_is_playing_event = video_is_playing_event
        
        self.monitor_task: asyncio.Task | None = None
        self._monitor_stop_event = asyncio.Event()

        if not os.getenv("OPENROUTER_API_KEY"):
            logging.critical("API key OPENROUTER environment variable is not set. Browser-use functionality will fail.")
            self.tts.play_greeting("Error: An API key is not set. Browser functions are disabled.")

    YOUTUBE_VIDEO_URL_PATTERNS = [
        "youtube.com/watch",
        "youtube.com/shorts/"
    ]

    async def initialize_session(self):
        if self.browser_session is None or not await self.browser_session.is_connected():
            logging.info("Initializing persistent browser session.")
            self.browser_session = BrowserSession(
                keep_alive=True,
                headless=False,
                executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                user_data_dir='~/.config/browseruse/profiles/default',
                window_size={"width": 1280, "height": 1420},
            )
            await self.browser_session.start()
            self.start_monitoring()

    def start_monitoring(self):
        if self.monitor_task and not self.monitor_task.done():
            logging.warning("[URL Monitor] Monitor task is already running.")
            return
        logging.info("[URL Monitor] Starting persistent URL monitor task.")
        self._monitor_stop_event.clear()
        self.monitor_task = asyncio.create_task(self._monitor_video_state(), name="PersistentURLMonitor")

    async def stop_monitoring(self):
        if self.monitor_task and not self.monitor_task.done():
            logging.info("[URL Monitor] Stopping persistent URL monitor task.")
            self._monitor_stop_event.set()
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                logging.info("[URL Monitor] Monitor task was successfully cancelled.")
        self.monitor_task = None

    # --- MODIFIED: The monitor is now resilient to initialization race conditions ---
    async def _monitor_video_state(self):
        logging.info("[URL Monitor] Persistent monitor started.")
        while not self._monitor_stop_event.is_set():
            try:
                # Check for connection readiness
                if self.browser_session and await self.browser_session.is_connected():
                    page = await self.browser_session.get_current_page()
                    url = page.url
                    is_video_page = any(pattern in url for pattern in self.YOUTUBE_VIDEO_URL_PATTERNS)
                    event_is_set = self.video_is_playing_event.is_set()

                    if is_video_page and not event_is_set:
                        logging.info(f"[URL Monitor] YouTube video page detected ({url}). Setting video_is_playing_event.")
                        self.video_is_playing_event.set()

                    elif not is_video_page and event_is_set:
                        logging.info(f"[URL Monitor] No longer on a video page. Clearing video_is_playing_event.")
                        self.video_is_playing_event.clear()
                else:
                    logging.warning("[URL Monitor] Browser session not ready yet. Waiting...")
                    await asyncio.sleep(1.0) # Wait for 1 second before checking again
                    continue # Continue to the next loop iteration

                await asyncio.sleep(1.0) # Check every second

            except asyncio.CancelledError:
                logging.info("[URL Monitor] Monitor loop was cancelled.")
                break 
            except Exception as e:
                logging.error(f"[URL Monitor] Exception during check: {e}", exc_info=True)
                await asyncio.sleep(2.0)
        
        if self.video_is_playing_event.is_set():
            self.video_is_playing_event.clear()
            logging.info("[URL Monitor] Monitor stopping, ensuring video_is_playing_event is cleared.")
        logging.info("[URL Monitor] Persistent monitor stopped.")


    async def _navigate_to_home(self):
        """Navigates the current browser page to the site's base URL."""
        if self.browser_session and await self.browser_session.is_connected():
            try:
                logging.info("Navigating to base URL.")
                page = await self.browser_session.get_current_page()
                current_url = page.url
                base_url = "https://www.google.com"
                match = re.search(r"https?://(www\.)?([^/]+)", current_url)
                if match:
                    base_url = f"https://{match.group(2)}"
                await page.goto(base_url)
                logging.info(f"Navigation to {base_url} successful.")
            except Exception as e:
                logging.error(f"Failed to navigate to home: {e}")
        else:
            logging.warning("Cannot navigate to home, browser session is not active.")

    async def run_browser_task(self, task: str, go_home_event: threading.Event):
        logging.info(f"Starting browser task: {task}")
        self.tts.play_greeting("Okay, on it. This might take a moment.")
        
        agent_task = None
        watcher_task = None

        try:
            await self.initialize_session()

            llm = ChatOpenAI(
                model='google/gemini-2.5-flash-lite',
                base_url='https://openrouter.ai/api/v1',
                api_key=os.getenv('OPENROUTER_API_KEY'),
            )

            agent = Agent(
                task=task,
                llm=llm,
                controller=self.controller,
                browser_session=self.browser_session,
                extend_system_message="You are an expert web assistant. Be concise."
            )

            async def watch_for_go_home():
                while not go_home_event.is_set():
                    await asyncio.sleep(0.1)
                logging.info("Go home event detected by watcher task.")

            agent_task = asyncio.create_task(agent.run(), name="AgentRun")
            watcher_task = asyncio.create_task(watch_for_go_home(), name="GoHomeWatcher")

            done, pending = await asyncio.wait(
                {agent_task, watcher_task},
                return_when=asyncio.FIRST_COMPLETED
            )
            
            if watcher_task in done:
                logging.info("'Reset & Ready' signal received. Stopping agent and navigating home.")
                await self._navigate_to_home()
                self.tts.play_greeting("Okay, I've stopped that. What's next?")
            else: 
                history = await agent_task
                final_result = history.final_result()

                if self.video_is_playing_event.is_set():
                    logging.info("Agent task finished, and a video is playing. Handing off to monitor.")
                    self.tts.play_greeting("The video is now playing. To stop and reset, press the Command key.")
                elif final_result:
                    logging.info(f"Browser task finished. Final result: {final_result}")
                    self.tts.play_greeting(f"Task complete. Here is the result: {final_result}")
                else:
                    logging.info("Browser task finished with no final result.")
                    self.tts.play_greeting("I've completed the browser task.")
            
        except Exception as e:
            logging.error(f"An error occurred during the browser task: {e}", exc_info=True)
            if not self.video_is_playing_event.is_set():
                self.tts.play_greeting(f"Sorry, an error occurred: {str(e)}")
        finally:
            for t in {agent_task, watcher_task}:
                if t and not t.done():
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        logging.info(f"Task '{t.get_name()}' was cancelled successfully.")
            logging.info("Agent task lifecycle has ended.")

    async def shutdown(self):
        await self.stop_monitoring() 
        if self.browser_session and await self.browser_session.is_connected():
            logging.info("Shutting down browser session as part of application exit.")
            await self.browser_session.kill()
            self.browser_session = None

class VoiceAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.tts_is_speaking_event = mp.Event()
        self.barge_in_event = threading.Event()
        self.is_replying_event = threading.Event()
        self.video_is_playing_event = threading.Event()
        self.go_home_event = threading.Event()

        self.gemma = GemmaChatEngine(config)
        self.tts = StreamingTTSEngine(config, self.tts_is_speaking_event)
        self.transcriber = StreamingTranscriber(config, self.tts_is_speaking_event, self.barge_in_event, self.video_is_playing_event)
        self.key_listener = KeyPressListener(self.is_replying_event, self.barge_in_event, self.video_is_playing_event, self.go_home_event)
        self.browser_controller = BrowserController(self.tts, self.video_is_playing_event)
        self.browser_task_thread: threading.Thread | None = None
        
        self.asyncio_loop: asyncio.AbstractEventLoop | None = None
        self.asyncio_thread: threading.Thread | None = None
        self.loop_is_ready = threading.Event()

    def _start_asyncio_loop(self):
        logging.info("Starting asyncio event loop thread.")
        self.asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.asyncio_loop)
        self.loop_is_ready.set()
        self.asyncio_loop.run_forever()
        self.asyncio_loop.close()
        logging.info("Asyncio event loop has been closed.")

    def _submit_coro_and_wait(self, coro: Coroutine):
        if not self.asyncio_loop or not self.asyncio_loop.is_running():
            logging.error("Cannot submit coroutine, asyncio loop is not running.")
            return

        future = asyncio.run_coroutine_threadsafe(coro, self.asyncio_loop)
        try:
            return future.result()
        except Exception as e:
            logging.error(f"Error in coroutine execution: {e}", exc_info=True)
            return e

    def _handle_browser_task(self, task: str):
        try:
            self._submit_coro_and_wait(
                self.browser_controller.run_browser_task(task, self.go_home_event)
            )

            if self.go_home_event.is_set():
                return

            if self.video_is_playing_event.is_set():
                logging.info("Browser thread entered post-task monitoring. Waiting for 'Go Home' signal or navigation away.")
                # This loop keeps the thread alive, polling for two conditions to exit:
                # 1. The user presses Command (go_home_event is set).
                # 2. The user manually navigates away from the video (video_is_playing_event is cleared by the URL monitor).
                while self.video_is_playing_event.is_set():
                    if self.go_home_event.is_set():
                        logging.info("Go home event received during post-task monitoring. Navigating home.")
                        # Since we are in the correct thread, we can now safely issue the command to go home.
                        self._submit_coro_and_wait(self.browser_controller._navigate_to_home())
                        self.tts.play_greeting("Okay, I've reset the browser.")
                        break  # Exit the monitoring loop to allow the thread to finish.
                    time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error running browser task thread: {e}", exc_info=True)
        finally:
            if self.video_is_playing_event.is_set():
                self.video_is_playing_event.clear()

            self.go_home_event.clear()
            logging.info("Browser task thread finished. Assistant is ready for new commands.")

    def run(self):
        self.asyncio_thread = threading.Thread(target=self._start_asyncio_loop, name="Asyncio-Loop-Thread", daemon=True)
        self.asyncio_thread.start()
        self.loop_is_ready.wait()
        
        self.key_listener.start()
        self.transcriber.start()
        self.tts.play_greeting("Gemma V is ready. How can I help?")
        print(f"\n{TermColors.BOLD}{TermColors.SYSTEM}Press the Command key to interrupt the assistant or reset a browser task.{TermColors.ENDC}\n", flush=True)

        try:
            while True:
                if self.browser_task_thread and not self.browser_task_thread.is_alive():
                    logging.info("Detected finished browser thread. Joining it to clean up.")
                    self.browser_task_thread.join()
                    self.browser_task_thread = None
                    logging.info("Browser thread joined. State is now free.")

                try:
                    final_transcript = self.transcriber.final_transcript_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                logging.info(f"Orchestrator received: '{final_transcript}'")

                if self.browser_task_thread and self.browser_task_thread.is_alive():
                    if not self.video_is_playing_event.is_set():
                        self.tts.play_greeting("Please wait, I'm currently busy with a browser task.")
                    continue

                self.is_replying_event.set()
                try:
                    if "goodbye" in final_transcript.lower():
                        self.tts.play_greeting("Goodbye!"); time.sleep(2); break
                    if "reset conversation" in final_transcript.lower():
                        self.gemma.reset(); self.tts.play_greeting("History cleared."); continue

                    generation_done_event = threading.Event()
                    response_generator = GeneratorWithReturnValue(self.gemma.stream_intent_and_response(final_transcript, self.barge_in_event))
                    self.tts.speak(iter(response_generator), self.barge_in_event, generation_done_event)
                    generation_done_event.wait()
                    
                    while self.tts.sentence_queue and not self.tts.sentence_queue.empty(): time.sleep(0.1)
                    while self.tts_is_speaking_event.is_set(): time.sleep(0.1)

                    if self.barge_in_event.is_set(): self.tts.interrupt()

                    result = response_generator.return_value
                    if result and result.get("intent") == "browseruse":
                        payload = result.get("payload", "")
                        self.browser_task_thread = threading.Thread(
                            target=self._handle_browser_task, args=(payload,), name="Browser-Task-Thread"
                        )
                        self.browser_task_thread.start()
                finally:
                    self.barge_in_event.clear()
                    self.is_replying_event.clear()

        except KeyboardInterrupt: logging.info("Caught KeyboardInterrupt.")
        finally:
            self.shutdown()

    def shutdown(self):
        print("\nShutting down...")
        self.barge_in_event.set()
        self.go_home_event.set()
        
        self.key_listener.stop()
        self.transcriber.stop()
        self.tts.shutdown()
        
        if self.browser_task_thread and self.browser_task_thread.is_alive():
            logging.info("Main loop exiting. Waiting for browser thread to finish...")
            self.browser_task_thread.join(timeout=10)

        if self.asyncio_loop and self.asyncio_loop.is_running():
            logging.info("Shutting down async components...")
            self._submit_coro_and_wait(self.browser_controller.shutdown())
            
            self.asyncio_loop.call_soon_threadsafe(self.asyncio_loop.stop)
        
        if self.asyncio_thread and self.asyncio_thread.is_alive():
            logging.info("Waiting for asyncio thread to terminate...")
            self.asyncio_thread.join(timeout=5)
            
        logging.info("Assistant shut down completely.")
            
if __name__ == "__main__":
    try:
        if not Path(Config.PIPER_MODEL_PATH).exists():
            print(f"{TermColors.SYSTEM}Error: Piper TTS model not found at '{Config.PIPER_MODEL_PATH}'. Please check the path in the Config class.{TermColors.ENDC}")
        elif not os.getenv("OPENROUTER_API_KEY"):
             print(f"{TermColors.SYSTEM}Error: OPENROUTER_API_KEY not found in environment variables. Please set one for browser tasks.{TermColors.ENDC}")
        else:
            config = Config()
            assistant = VoiceAssistant(config)
            assistant.run()
    except Exception as e:
        logging.critical(f"Fatal error in main execution block: {e}", exc_info=True)