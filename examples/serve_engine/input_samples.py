import base64
import os
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def get_interleaved_dialogue_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: vocal fry;moderate pitch;monotone;masculine;young adult;slightly fast\n"
        # "SPEAKER1: masculine;moderate;moderate pitch;monotone;mature\n\n"
        # "In this scene, a group of adventurers is debating whether to investigate a potentially dangerous situation.\n"
        "In this scene, introduce yourself in the first person.\n"
        "<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 20 seconds.\n<|generation_instruction_end|>",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample

def get_interleaved_chat_input_sample(response_type="ta"):
    # system_prompt = (
    #     "Generate audio following instruction.\n\n"
    #     "<|scene_desc_start|>\n"
    #     "Audio is recorded from a quiet room.\n"
    #     "<|scene_desc_end|>"
    # )

    system_prompt = (
        "Answer the question of the user!"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            # content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/11/2_1_d11.wav"),
            # content="Why so late? Didn’t she want to get married this October?",

            # content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/181/0_1_d181.wav"),
            # content="Say, what's your favorite sport?",

            # content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/923/0_1_d923.wav"),
            # content="What can I do for you, sir?",

            # content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/512/0_1_d512.wav"),
            # content="How about going to the cinema tonight?",

            # content="How many years of history does China have?",
            # content="How many countries are there in the world?",
            # content = "What is the capital of France?",
            # content = "Which river is the longest in South America?",
            # content = "What is the highest mountain peak in North America?",
            # content = "Who was the first president of the United States?",
            # content = "Which city is located at the intersection of the Tigris and Euphrates rivers?",
            content = AudioContent(audio_url="/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/VoiceBench/processed/llama-questions/audio/2.wav"),


            response_type=response_type,
            # content="<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 20 seconds.\n<|generation_instruction_end|>",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample

def get_interleaved_chat_input_sample_ref(response_type="ta"):
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "Audio is recorded from a quiet room.\n"
        "<|scene_desc_end|>"
    )

    speaker = "zh_man_sichuan"
    audio_file_path = f"/apdcephfs_cq10/share_1297902/user/nenali/project/wanghualei/code/train-higgs-audio/examples/voice_prompts/{speaker}.wav"
    text_file_path = audio_file_path[:-3] + "txt"
    with open(text_file_path, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        # Message(
        #     role="user", 
        #     content=f"[SPEAKER0] {transcript}"
        # ),
        Message(
            role="user", 
            content=f"[SPEAKER0]"
        ),
        Message(
            role="assistant", 
            content=AudioContent(audio_url=audio_file_path)
        ),
        Message(
            role="user",
            # content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/11/2_1_d11.wav"),
            content=AudioContent(audio_url="/apdcephfs_cq10/share_1297902/data/speech_data/qa_data/dailytalk/data/23/0_1_d23.wav"),
            response_type=response_type,
            # content="<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 20 seconds.\n<|generation_instruction_end|>",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages, start_index=3)
    return chat_ml_sample

def get_zero_shot_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n<|scene_desc_start|>\nSPEAKER0: british accent\n<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


def get_voice_clone_input_sample():
    reference_text = "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic."
    reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "voice_examples/old_man.wav")
    )
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    return ChatMLSample(messages=messages)


INPUT_SAMPLES = {
    "interleaved_dialogue": get_interleaved_dialogue_input_sample,
    "zero_shot": get_zero_shot_input_sample,
    "voice_clone": get_voice_clone_input_sample,
    "chat":get_interleaved_chat_input_sample,
    "chat_ref":get_interleaved_chat_input_sample_ref,
}
