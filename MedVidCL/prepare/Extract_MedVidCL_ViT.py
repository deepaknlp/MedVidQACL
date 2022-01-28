import os
import cv2
import glob
import json
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import videotransforms
from feature_extractor import InceptionI3d
from torchvision import transforms
from torch.autograd import Variable
from pytube import YouTube
from transformers import ViTModel


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="gpu index")
parser.add_argument("--dataset_dir", type=str, default='data/text', help="dataset path")
parser.add_argument("--dataset_file_format", type=str, default='json', help="dataset file format")

parser.add_argument("--video_dir", type=str, default='data/videos/medvidcl', help="where to download the videos")
parser.add_argument("--images_dir", type=str, default='data/images/medvidcl', help="where to save extracted images")
parser.add_argument("--save_dir", type=str, default='data/features/ViT', help="where to save extracted features")
parser.add_argument("--fps", type=float, default=16, help="frames per second")  # TACoS's default fps is 29.4
parser.add_argument("--fpps", type=float, default=2000, help="frames processing per second")
parser.add_argument("--video_format", type=str, default=".avi", help="video format")
parser.add_argument("--strides", type=int, default=16, help="window size")
parser.add_argument("--remove_images", default=True, help="whether remove extract images to release space")
args = parser.parse_args()


is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    print("using GPU...")


def load_images(img_dir, vid, start_frame, lengths):
    img_frames, raw_height, raw_width = [], None, None
    for x in range(start_frame, start_frame + lengths):
        image = cv2.imread(os.path.join(img_dir, "{}-{}.jpg".format(vid, str(x).zfill(6))))[:, :, [2, 1, 0]]
        width, height, channel = image.shape
        raw_width, raw_height = width, height
        # resize image
        scale = 1 + (224.0 - min(width, height)) / min(width, height)
        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        # normalize image to [0, 1]
        image = (image / 255.0) * 2 - 1
        img_frames.append(image)
    return img_frames, raw_width, raw_height


def extract_features(image_tensor, model, strides):
    # Image Tensor is 1 x channels (3 - RGB) x frames x height x width
    
    # Only feed image tensor of frames x channel x height x width into model
    outputs = model(torch.permute(image_tensor[0], (1, 0, 2, 3)))

    # Get the embedding tensor which is frames x cls+196 patches x out_channel
    last_hidden_states = outputs.last_hidden_state

    # Transform into numpy array
    last_hidden_state_numpy = last_hidden_states.detach().numpy()

    # Only return frames x out_channel of the 2D cls array
    return last_hidden_state_numpy[:, 0, :]

def download_all_videos(video_links):
    print(f'Downloading video...')
    for link in tqdm(video_links):
        download_video(link, args.video_dir)
    print('Videos have been downloaded.')

def download_video(link, path):
    def select_stream(streams):
        for stream in streams:
            if stream.resolution == '360p':
                return stream
        return streams.first()

    if link.startswith('https:') and '/?start=' in link and 'embed' in link:
        fname = link.split('/?start=')[0]
        video_id = fname.split('embed/')[1]
        video_link = 'https://www.youtube.com/watch?v=' + video_id
    elif link.startswith('https:') and 'watch?v=' in link:
        video_id = link.split('watch?v=')[1]
        video_link = link
    else:
        video_id = link
        video_link = 'https://www.youtube.com/watch?v=' + video_id

    if os.path.exists(os.path.join(path, video_id + args.video_format)):
        return
    else:
        try:
            # object creation using YouTube
            # which was imported in the beginning
            yt = YouTube(video_link)
            streams = yt.streams.filter(progressive=True)
            stream = select_stream(streams)
        except Exception as e:
            # to handle exception
            print(e)
            print(f"Connection Error while downloading video {video_link}")
        try:
            # downloading the video
            stream.download(path, filename=video_id+args.video_format)
            # print(f"Downloaded... video from {video_link}")
        except Exception as e:
            # to handle exception
            print(e)
            print(f"Some Error while downloading video {video_link}")



def read_json_dataset_files(dataset_dir):
    '''
    read all the video link from the training, test and val set
    '''

    videos_list = []
    dataset_files = glob.glob(os.path.join(dataset_dir, "*{}".format(args.dataset_file_format)))
    for file in dataset_files:
        print(f"Reading file: {file}")
        with open(file, 'r') as read_file:
            data_items = json.load(read_file)
        for data_item in tqdm(data_items):
            videos_list.append(data_item['video_id'])

    print(f"Total video clips: {len(list(set(videos_list)))}")
    return videos_list

def read_dataset_file(dataset_dir):
    videos_list = []
    dataset_files = glob.glob(os.path.join(dataset_dir, "*{}".format(args.dataset_file_format)))
    for file in dataset_files:
        print(f"Reading file: {file}")
        with open(file, 'r') as read_file:
            lines = read_file.readlines()
        for line in tqdm(lines[1:]): ### skipping header
            items = line.split('\t')
            assert len(items) == 12
            video_link = items[2]
            videos_list.append(video_link)
    print(f"Total video clips: {len(list(set(videos_list)))}")
    return videos_list

def read_dataset_test_file(dataset_dir):
    videos_list = []
    dataset_files = glob.glob(os.path.join(dataset_dir, "*{}".format(args.dataset_file_format)))
    for file in dataset_files:
        print(f"Reading file: {file}")
        with open(file, 'r') as read_file:
            lines = read_file.readlines()
        for line in tqdm(lines[1:]): ### skipping header
            items = line.split('\t')
            assert len(items) == 5
            video_link = items[0]
            videos_list.append(video_link)
    print(f"Total video clips: {len(list(set(videos_list)))}")
    return videos_list

def get_length(filename): # New
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


if not os.path.exists(args.video_dir):
    os.makedirs(args.video_dir)

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

#### download videos if required
video_link_list= read_json_dataset_files(args.dataset_dir)
download_all_videos(video_links=video_link_list)


# Create and load pre-trained ViT model 
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
video_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

# extract images and features
feature_shape_path=os.path.join(args.save_dir, "feature_shapes.json")

do_compute_feature_shape=True

feature_shapes = dict()
video_paths = glob.glob(os.path.join(args.video_dir, "*{}".format(args.video_format)))
for idx, video_path in enumerate(video_paths):
    video_id = os.path.basename(video_path).replace(args.video_format, '') #### getting video ids from the video file name

    image_dir = os.path.join(args.images_dir, video_id)
    print("{} / {}: extract features for video {}".format(idx + 1, len(video_paths), video_id), flush=True)

    if os.path.exists(os.path.join(args.save_dir, "{}.npy".format(video_id))):
        print("the visual features for video {} are exist in {}...".format(video_id, args.save_dir), flush=True)
        if do_compute_feature_shape:
            loaded_feature=np.load(os.path.join(args.save_dir, "{}.npy".format(video_id)))
            print("extracted features shape: {}".format(loaded_feature.shape), flush=True)
            feature_shapes[video_id] = loaded_feature.shape[0]
        continue

    # extract images
    if os.path.exists(image_dir):
        print("the images for video {} already are exist in {}...".format(video_id, args.images_dir))
    else:
        os.makedirs(image_dir)
        print("extract images with fps={}...".format(args.fps), flush=True)
        video_dur_seconds = get_length(video_path)
        frames_per_second_rate = 18/video_dur_seconds
        subprocess.call("ffmpeg -i {} -r {} -f image2 {}/{}-%6d.jpg".format(video_path, frames_per_second_rate, image_dir,
                                                                                         video_id), shell=True)
        

    # process extracted images
    print("load RGB frames...", flush=True)
    num_frames = len(os.listdir(image_dir))

    if num_frames < args.fpps:
        frames, raw_w, raw_h = load_images(image_dir, video_id, 1, num_frames)
        frames = np.asarray(frames, dtype=np.float32)
        imgs = video_transforms(frames)
        img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
        print("process images:", (frames.shape[0], raw_w, raw_h, frames.shape[-1]), "-->", frames.shape, "-->",
              imgs.shape, "-->", tuple(img_tensor.size()), flush=True)

        print("extract visual features...", flush=True)
        features = extract_features(img_tensor, model, args.strides)
        np.save(os.path.join(args.save_dir, video_id), arr=features)
        print("extracted features shape: {}".format(features.shape), flush=True)
        feature_shapes[video_id] = features.shape[0]

    else:
        all_features = []
        for start_idx in range(1, num_frames, args.fpps):
            end_idx = min(start_idx + args.fpps, num_frames + 1)
            cur_num_frames = end_idx - start_idx
            if cur_num_frames < args.strides:
                cur_num_frames = args.strides
                start_idx = end_idx - cur_num_frames
            frames, raw_w, raw_h = load_images(image_dir, video_id, start_idx, cur_num_frames)
            frames = np.asarray(frames, dtype=np.float32)
            imgs = video_transforms(frames)
            img_tensor = torch.from_numpy(np.expand_dims(imgs.transpose([3, 0, 1, 2]), axis=0))
            print("process images:", (frames.shape[0], raw_w, raw_h, frames.shape[-1]), "-->", frames.shape, "-->",
                  imgs.shape, "-->", tuple(img_tensor.size()), flush=True)
            print("extract visual features...", flush=True)
            features = extract_features(img_tensor, model, args.strides)
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        np.save(os.path.join(args.save_dir, video_id), arr=all_features)
        print("extracted features shape: {}".format(all_features.shape), flush=True)
        feature_shapes[video_id] = all_features.shape[0]

    if args.remove_images:
        # remove extract images to release memory space
        subprocess.call("rm -rf {}".format(image_dir), shell=True)

with open(feature_shape_path, mode="w", encoding="utf-8") as f:
    json.dump(feature_shapes, f)
