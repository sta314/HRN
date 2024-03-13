import os
from moviepy.editor import VideoFileClip, AudioFileClip, ImageSequenceClip, clips_array
import imageio
from models.hrn import Reconstructor
import cv2
from tqdm import tqdm
import shutil

def video_extract_frames_and_audio(video_path, output_dir):
    frames_dir = os.path.join(output_dir, 'frames')
    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    video_clip = VideoFileClip(video_path)

    fps = video_clip.fps

    for i, frame in tqdm(enumerate(video_clip.iter_frames())):
        frame_path = os.path.join(frames_dir, f'{i}.png')
        imageio.imwrite(frame_path, frame)

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

def run_hrn(in_dir, out_dir):
    args = {
        'checkpoints_dir': 'assets/pretrained_models',
        'name': 'hrn_v1.1',
        'epoch': '10',
        'input_type': 'single_view',
        'input_root': in_dir,
        'output_root': out_dir,
    }
    params = [
        '--checkpoints_dir', args['checkpoints_dir'],
        '--name', args['name'],
        '--epoch', args['epoch'],
    ]

    reconstructor = Reconstructor(params)

    names = sorted([name for name in os.listdir(args['input_root']) if '.jpg' in name or '.png' in name or '.jpeg' in name or '.PNG' in name or '.JPG' in name or '.JPEG' in name])

    for _, name in enumerate(tqdm(names)):
        save_name = os.path.splitext(name)[0]
        out_dir = args['output_root']
        os.makedirs(out_dir, exist_ok=True)
        img = cv2.imread(os.path.join(args['input_root'], name))
        reconstructor.predict(img, visualize=True, save_name=save_name, out_dir=out_dir, use_threshold=True)

media_dir = 'media'
input_dir = os.path.join(media_dir, 'input') # folder that contains videos to be processed
output_dir = os.path.join(media_dir, 'output') # folder to output results

temporary_dir = os.path.join(media_dir, 'temporary') # folder for storing temporary files
frames_dir = os.path.join(temporary_dir, 'frames') # folder for storing temporary frames
audio_dir = os.path.join(temporary_dir, 'audio') # folder for storing temporary audio

video_list = os.listdir(input_dir)

for i, video_name in enumerate(video_list):
    instance_output_dir = os.path.join(output_dir, os.path.splitext(video_name)[0])
    instance_output_frames_dir = os.path.join(instance_output_dir, 'frames')
    print('{}/{} - Starting processing for the video {}'.format(i+1, len(video_list), video_name))
    print('Extracting frames from the video')
    fps = video_extract_frames_and_audio(os.path.join(input_dir, video_name), temporary_dir)
    print('Executing the pipeline for every frame')
    run_hrn(frames_dir, instance_output_frames_dir)
    print('Merging resulting frames for recreating the video')
    recreate_video_from_frames(fps, instance_output_frames_dir, os.path.join(audio_dir, 'audio.mp3'), os.path.join(instance_output_dir, 'reconstruction.mp4'))
    print('Merging original video with recreated video for visualization purposes')
    merge_original_and_reconstructed_videos(os.path.join(input_dir, video_name), os.path.join(instance_output_dir, 'reconstruction.mp4'), os.path.join(instance_output_dir, 'merged.mp4'))
    shutil.rmtree(temporary_dir)
    print('{}/{} - Finished processing video {}'.format(i+1, len(video_list), video_name))
    print('-' * 30)