# Gemma V: Your Voice is the Browser

We built a fully conversational AI assistant on a laptop, using Gemma 3n to give the visually impaired voice-driven access to the entire web.

Demo: https://www.youtube.com/watch?v=e6HVVWYjdtw

## Env

- Apple M1 16GB


```
conda create -n gemmav python=3.11.8
conda activate gemmav
conda install -c conda-forge torchaudio ffmpeg librosa numpy pynput

pip install torch mlx mlx-lm mlx-vlm sounddevice 
pip install piper-tts rustymimi moshi-mlx huggingface-hub sentencepiece 
pip install --upgrade huggingface-hub
pip install parakeet-mlx -U
```


```
mkdir -p models
wget 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx' -O 'models/en_US-lessac-low.onnx'
wget 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json' -O 'models/en_US-lessac-low.onnx.json'
```

```
pip install browser-use==0.5.5
playwright install --with-deps
# create a .env file in project's root, and add your OPENROUTER_API_KEY
OPENROUTER_API_KEY=""
```

# Run

```
python gemma_v.py

# Say "Go to YouTube and play a video about quantum computing, and skip advertisement if there is one"
```



## References

- https://huggingface.co/mlx-community/gemma-3n-E4B-it-4bit/tree/main

