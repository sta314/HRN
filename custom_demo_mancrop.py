import os
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, clips_array
import imageio
from models.hrn import Reconstructor
import cv2
from tqdm import tqdm
import shutil
import numpy as np
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def video_extract_cropped_frames_and_audio(video_path, boxes, output_frames_dir, output_audio_dir):
    frames_dir = output_frames_dir
    audio_dir = output_audio_dir
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    video_clip = VideoFileClip(video_path)

    frames = []
    for _, frame in enumerate(video_clip.iter_frames()):
        frames.append(frame)

    length_diff = len(frames) - len(boxes)

    fps = video_clip.fps

    import concurrent.futures

    def process_frame(frame, box, frames_dir, i):
        x1, y1, x2, y2 = box
        cropped_frame = frame[y1:y2, x1:x2] if x1 != -1 else np.zeros((224, 224, 3), dtype=np.uint8)
        frame_path = os.path.join(frames_dir, f'{i}.png')
        imageio.imwrite(frame_path, cropped_frame)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for i in range(len(frames)):
            if i == (len(frames) - 1): # skip the frame at the end
                break
            if i < length_diff - 1: # skip remaining frames from beginning
                continue
            frame = frames[i]
            box = boxes[i - (length_diff - 1)]
            future = executor.submit(process_frame, frame, box, frames_dir, i - (length_diff - 1))
            futures.append(future)
        concurrent.futures.wait(futures)

    audio_path = os.path.join(audio_dir, 'audio.mp3')
    video_clip.audio.write_audiofile(audio_path, logger=None)

    video_clip.close()

    return fps

def recreate_video_from_frames(fps, frames_dir, audio_path, output_path):
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]

    frame_files.sort(key=lambda x: int(x.split('.')[0]))

    frames = []

    for _, frame_file in enumerate(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        frame = imageio.imread(frame_path)
        frames.append(frame)

    video_clip = ImageSequenceClip(frames, fps=fps)
    video_clip = video_clip.set_audio(AudioFileClip(audio_path))
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', logger=None)

def merge_original_and_reconstructed_videos(og_path, recon_path, out_path):

    cropped_clip = VideoFileClip(og_path)
    output_clip = VideoFileClip(recon_path)

    cropped_clip_resized = cropped_clip.resize(height=output_clip.h)
    output_clip_resized = output_clip

    merged_clip = clips_array([[cropped_clip_resized, output_clip_resized]])
    merged_clip.write_videofile(out_path, codec="libx264", logger=None)

    cropped_clip.close()
    output_clip.close()

def run_hrn(in_dir, out_frames_dir, out_extras_dir):
    args = {
        'checkpoints_dir': 'assets/pretrained_models',
        'name': 'hrn_v1.1',
        'epoch': '10',
        'input_type': 'single_view',
        'input_root': in_dir,
        'output_frames': out_frames_dir,
        'output_extras': out_extras_dir,
    }
    params = [
        '--checkpoints_dir', args['checkpoints_dir'],
        '--name', args['name'],
        '--epoch', args['epoch'],
    ]

    reconstructor = Reconstructor(params)
    names = [name for name in os.listdir(args['input_root']) if '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name]
    names = sorted(names, key=lambda f: int(f.split('.')[0]))

    images = []
    for name in names:
        img = cv2.imread(os.path.join(args['input_root'], name))
        images.append(img)

    for i, img in enumerate(tqdm(images)):
        save_name = os.path.splitext(names[i])[0]
        out_frames_dir = args['output_frames']
        out_extras_dir = args['output_extras']
        os.makedirs(out_frames_dir, exist_ok=True)
        os.makedirs(out_extras_dir, exist_ok=True)
        reconstructor.predict(img, visualize=True, save_name=save_name, out_frames_dir=out_frames_dir, out_extras_dir=out_extras_dir, use_threshold=True)

parser = argparse.ArgumentParser()
parser.add_argument('--media_dir', default='media_mancrop', help='Directory for input files')
parser.add_argument('--output_dir', default='output', help='Directory for outputs')
args = parser.parse_args()

media_dir = args.media_dir
input_dir = os.path.join(media_dir, 'input') # folder that contains videos to be processed
input_videos_dir = os.path.join(input_dir, 'videos')
input_frameinfo_dir = os.path.join(input_dir, 'faces')

output_dir = os.path.join(media_dir, args.output_dir) # folder to output results

temporary_dir = os.path.join(media_dir, 'temporary') # folder for storing temporary files

frames_dir = os.path.join(temporary_dir, 'frames') # folder for storing temporary frames
audio_dir = os.path.join(temporary_dir, 'audio')   # folder for storing temporary audio

video_list = os.listdir(input_videos_dir)
for i, video_name in enumerate(video_list):

    video_name_base = os.path.splitext(video_name)[0]
    instance_frameinfo_dir = os.path.join(input_frameinfo_dir, video_name_base)
    instance_output_dir = os.path.join(output_dir, video_name_base)
    instance_output_frames_dir = os.path.join(instance_output_dir, 'frames')
    instance_output_extras_dir = os.path.join(instance_output_dir, 'extras')
    instance_video_path = os.path.join(input_videos_dir, video_name)

    frame_infos = list()

    txt_files = [f for f in os.listdir(instance_frameinfo_dir) if f.endswith('.txt')]
    txt_files = sorted(txt_files, key=lambda f: int(f.split('.')[0]))
    
    for txt_file in txt_files:
        txt_path = os.path.join(instance_frameinfo_dir, txt_file)
        with open(txt_path, 'r') as file:
            first_line = file.readline().strip()
            frame_info = [int(x) for x in first_line.split()]
            frame_infos.append(frame_info)

    if os.path.exists(temporary_dir) and os.path.isdir(temporary_dir):
        shutil.rmtree(temporary_dir)

    instance_output_dir = os.path.join(output_dir, video_name_base)
    instance_output_frames_dir = os.path.join(instance_output_dir, 'frames')
    print('{}/{} - Starting processing for the video {}'.format(i+1, len(video_list), video_name))
    print('Extracting frames from the video and cropping')
    fps = video_extract_cropped_frames_and_audio(instance_video_path, frame_infos, frames_dir, audio_dir)
    print('Executing the pipeline for every frame')
    run_hrn(frames_dir, instance_output_frames_dir, instance_output_extras_dir)
    print('Merging resulting frames for recreating the video')
    recreate_video_from_frames(fps, instance_output_frames_dir, os.path.join(audio_dir, 'audio.mp3'), os.path.join(instance_output_dir, 'reconstruction.mp4'))
    print('Merging original video with recreated video for visualization purposes')
    merge_original_and_reconstructed_videos(instance_video_path, os.path.join(instance_output_dir, 'reconstruction.mp4'), os.path.join(instance_output_dir, 'merged.mp4'))

    # remove the instance
    if os.path.exists(instance_video_path):
        os.remove(instance_video_path)
    if os.path.exists(instance_frameinfo_dir) and os.path.isdir(instance_frameinfo_dir):
        shutil.rmtree(instance_frameinfo_dir)

    if os.path.exists(temporary_dir) and os.path.isdir(temporary_dir):
        shutil.rmtree(temporary_dir)

    print('{}/{} - Finished processing video {}'.format(i+1, len(video_list), video_name))
    print('-' * 30)