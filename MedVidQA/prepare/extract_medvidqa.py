import os
import cv2
import glob
import json
import torch
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
from prepare import videotransforms
from prepare.feature_extractor import InceptionI3d
from torchvision import transforms
from pytube import YouTube


parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx", type=str, default="0", help="gpu index")
parser.add_argument("--load_model", type=str, default='data/rgb_imagenet.pt', help="pre-trained model")
parser.add_argument("--dataset_dir", type=str, default='data/dataset/medvidqa', help="dataset path")
parser.add_argument("--dataset_file_format", type=str, default='json', help="dataset file format")

parser.add_argument("--video_dir", type=str, default='data/videos/medvidqa', help="where to download the videos")
parser.add_argument("--images_dir", type=str, default='data/images/medvidqa', help="where to save extracted images")
parser.add_argument("--save_dir", type=str, default='data/features/medvidqa', help="where to save extracted features")
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


def get_video_length(video_url):
    yt = YouTube(video_url)
    return yt.length




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
    #### extract the per window/stride wise features
    # import pdb
    # pdb.set_trace()
    b, c, t, h, w = image_tensor.shape
    extracted_features = []
    for start in range(0, t, strides):
        end = min(t - 1, start + strides)
        if end - start < strides:
            start = max(0, end - strides)
        ip = torch.from_numpy(image_tensor.numpy()[:, :, start:end])
        if is_cuda_available:
            ip = ip.cuda()
        with torch.no_grad():
            # feature = model.extract_features(ip).data.cpu().numpy() ## very slow https://discuss.pytorch.org/t/cant-convert-cuda-tensor-to-numpy-use-tensor-cpu-to-copy-the-tensor-to-host-memory-first/38301/3
            feature = model.extract_features(ip).detach().cpu().numpy()
        extracted_features.append(feature)
    extracted_features = np.concatenate(extracted_features, axis=0)
    return extracted_features

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
            videos_list.append(data_item['video_url'])

    print(f"Total video clips: {len(list(set(videos_list)))}")
    return videos_list




if not os.path.exists(args.video_dir):
    os.makedirs(args.video_dir)

if not os.path.exists(args.images_dir):
    os.makedirs(args.images_dir)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

#### download videos if required
video_link_list= read_json_dataset_files(args.dataset_dir)
download_all_videos(video_links=video_link_list)


# create I3D model and load pre-trained model
i3d_model = InceptionI3d(400, in_channels=3)
try:
    i3d_model.load_state_dict(torch.load(args.load_model))
except Exception as e:
    print(e)
print("Model has been initialized with pre-trained weights.")
if is_cuda_available:
    i3d_model.cuda()
i3d_model.train(False)
video_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

# extract images and features


feature_shape_path=os.path.join(args.save_dir, "feature_shapes.json")

do_compute_feature_shape=True
# if not os.path.exists(feature_shape_path):
#     do_compute_feature_shape=True

feature_shapes = dict()
video_paths = glob.glob(os.path.join(args.video_dir, "*{}".format(args.video_format)))
# print(video_paths)
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
        subprocess.call("ffmpeg -hide_banner -loglevel panic -i {} -r {} {}/{}-%6d.jpg".format(video_path, args.fps, image_dir,
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
        features = extract_features(img_tensor, i3d_model, args.strides)
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
            features = extract_features(img_tensor, i3d_model, args.strides)
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
