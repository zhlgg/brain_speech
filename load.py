

import torchaudio
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Available backends: {torchaudio.list_audio_backends()}")

if 'ffmpeg' in torchaudio.list_audio_backends():
    try:
        torchaudio.set_audio_backend("ffmpeg")
        print("Successfully set backend to FFmpeg.")
        # Replace item["audio"] with your actual MP3 file path
        # audio, sr = torchaudio.load(item["audio"])
        # print("MP3 loaded successfully with FFmpeg!")
    except Exception as e:
        print(f"Error using FFmpeg backend: {e}")
else:
    print("FFmpeg backend is still not available to torchaudio.")
    print("Please ensure FFmpeg is installed and correctly added to your system PATH.")
    print("You might need to reinstall torchaudio after configuring FFmpeg.")


path = "/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/minmax_Vicuna_fast/ques_6591.flac"
info = torchaudio.info(path)  # 不会把音频整段读入
print(info)  # sample_rate=24000, num_channels=1, num_frames=...

# 确保使用 sox_io 后端，FLAC 兼容好
try:
    torchaudio.set_audio_backend("ffmpeg")
except Exception:
    pass
waveform, sr = torchaudio.load(path)  # shape: [channels, num_frames], dtype=float32
# 若需单声道（你的是1声道，可省略）
waveform = waveform.mean(dim=0, keepdim=True)

# 若需重采样到 16k（示例）
target_sr = 16000
if sr != target_sr:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
    sr = target_sr

print(waveform.shape, sr)