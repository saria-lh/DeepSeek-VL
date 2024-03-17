import argparse
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from read_vid import *

# Set up argument parser
parser = argparse.ArgumentParser(description="Process a video and generate descriptions.")
parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
args = parser.parse_args()

# Use the provided video path
path = args.video_path
path_without_extension = path.rsplit('.', 1)[0]
audio_path = path_without_extension + '.mp3'
txt_file = path_without_extension + '.txt'

selected_frames = select_active_frames_and_timestamps_every_second(path, 5)
print('Number of frames', len(selected_frames))
frames_only = [f[1] for f in selected_frames]
save_images_to_folder(frames_only, path_without_extension + '_images')
extract_audio_from_video(path, audio_path)
transcribed_text = transcribe(audio_path)
model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "<image_placeholder> Describe this image, OBJECTS list briefly, and the text in it briefly.",
    },
    {"role": "Assistant", "content": ""},
]

for i, pil_im in enumerate(selected_frames):
    ts, im = pil_im
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[im], force_batchify=True
    ).to(vl_gpt.device)

    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=256,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{ts} \n {prepare_inputs['sft_format'][0]}", answer)
    with open(txt_file, 'a') as f:
        f.write(f'TS: {ts}\n')
        f.write(answer)
        f.write('\n\n')

with open(txt_file, 'a') as f:
    f.write(f'\nTranscribed_text:\n')
    f.write(transcribed_text)
