import argparse
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from read_vid import *
from database import *

def process_video(video_path: str, database_path: str) -> None:
    path_without_extension = video_path.rsplit('.', 1)[0]
    audio_path = f"{path_without_extension}.mp3"
    txt_file = f"{path_without_extension}.txt"
    images_folder_path = f"{path_without_extension}_images"

    selected_frames = extract_active_frames_ts(video_path, 5)
    print('Number of frames', len(selected_frames))
    frames_only = [f[1] for f in selected_frames]
    save_images_to_folder(frames_only, images_folder_path)
    
    if extract_audio_from_video(video_path, audio_path) is None:
        transcribed_text = ""  # No audio in video, so transcribe an empty string
    else:
        transcribed_text = transcribe(audio_path)

    model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    ).to(torch.bfloat16).cuda().eval()

    conversation = [
        {"role": "User", "content": "<image_placeholder> Describe this image, OBJECTS list briefly, and the text in it briefly."},
        {"role": "Assistant", "content": ""},
    ]

    with open(txt_file, 'a') as f:
        for i, (ts, im) in enumerate(selected_frames):
            prepare_inputs = vl_chat_processor(conversations=conversation, images=[im], force_batchify=True).to(vl_gpt.device)
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
            f.write(f'TS: {ts}\n{answer}\n\n')
        f.write(f'\nTranscribed_text:\n{transcribed_text}')
    video_data = (path_without_extension, txt_file, audio_path, images_folder_path)
    print(video_data)
    conn = create_connection(database_path)
    if conn is not None:
        create_table(conn)
        insert_video_data(conn, video_data)
        conn.close()
    else:
        print("Error! Cannot create the database connection.")

def main():
    parser = argparse.ArgumentParser(description="Process a video and generate descriptions.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--database_path", type=str, default="simple.db", help="Path to the database")
    args = parser.parse_args()

    process_video(args.video_path, args.database_path)

if __name__ == "__main__":
    main()
