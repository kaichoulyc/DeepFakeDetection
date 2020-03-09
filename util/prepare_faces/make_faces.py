import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm
import skvideo.io
from facenet_pytorch import MTCNN
from PIL import Image

from aligner import Alinger


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='Data path',
                    default=None)
parser.add_argument('--ex_path', type=str, help='New data path',
                    default=None)
parser.add_argument('--gpu', type=int, help='Gpu number',
                    default=None)
parser.add_argument('--am_frames', type=int, help='Amount of faces',
                    default=None)
parser.add_argument('--folders_on_gpu', nargs='+', help='Names of datasets')
parser.add_argument('--batch_size', type=int, help='Amount of frames to gpu',
                    default=None)
parser.add_argument('--start', type=int, help='Start video',
                    default=None)


class FastMTCNN(object):
    """Fast MTCNN implementation."""
    
    def __init__(self, stride, resize=1, *args, **kwargs):
        """Constructor for FastMTCNN class.
        
        Arguments:
            stride (int): The detection stride. Faces will be detected every `stride` frames
                and remembered for `stride-1` frames.
        
        Keyword arguments:
            resize (float): Fractional frame scaling. [default: {1}]
            *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
            **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)
        
    def __call__(self, frames):
        """Detect faces in frames using strided MTCNN."""
        if self.resize != 1:
            frames = [f.resize([int(d * self.resize) for d in f.size]) for f in frames]
                      
        boxes, probs, landamrks = self.mtcnn.detect(frames[::self.stride], landmarks=True)

        lands = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for land in landamrks[box_ind]:
                lands.append(land)

        del(frames)
        del(probs)
        del(boxes)
        del(landamrks)
        
        return lands


def rework_landmarks(landamrks):
    
    landamrks = landamrks.astype(int)
    new_lands = list(landamrks[:, 0])
    new_lands.extend(list(landamrks[:, 1]))
    return np.array([new_lands])

def extract_faces(data_path, ex_path, detector, aligner, am_frames, folders_on_gpu, batch_size, start):
    parts = os.listdir(data_path)
    print(folders_on_gpu)
    k = False
    for part in tqdm(parts[int(folders_on_gpu[0]):int(folders_on_gpu[1])]):
        if k:
            start = 0
        videos_path = os.path.join(data_path, part)
        for video_name in tqdm(os.listdir(videos_path)[start:]):
            newp_path = os.path.join(ex_path, video_name[:-4])
            if not os.path.exists(newp_path):
                os.makedirs(newp_path)
            try:
                frames_all = skvideo.io.vread(os.path.join(videos_path, video_name))
            except ValueError:
                continue
            len_frames = len(frames_all)
            chips_num = 0
            frames = []
            real_frames = []
            chipss = []
            for j, image in enumerate(frames_all):
                real_frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                frame = Image.fromarray(image)
                frames.append(frame)
                if len(frames) >= batch_size or j == len_frames - 1:
                    landmarkss = detector(frames)
                    for landmarks, image in zip(landmarkss, real_frames):
                        if len(landmarks) != 0:
                            landmarks = rework_landmarks(landmarks)
                            chips = aligner.extract_image_chips(image, landmarks)
                            chipss.append(chips)
                    del(frames)
                    del(real_frames)
                    frames = []
                    real_frames = []
            for chips in chipss:
                if chips:
                    cv2.imwrite(os.path.join(newp_path, f'{chips_num}.jpg'), chips[0])
                    chips_num += 1
            del(chipss)
        k = True
            # needed_frames = int(chips_num / am_frames) + 1
            # for i in range(chips_num):
            #     if (i % needed_frames) == 0:
            #         continue
            #     os.remove(os.path.join(newp_path, f'{i}.jpg'))


def main(args):

    detector = FastMTCNN(
        stride=4,
        resize=1,
        margin=14,
        factor=0.6,
        keep_all=True,
        device=f'cuda:{args.gpu}',

    )
    aligner = Alinger(300, 0.3)
    extract_faces(args.data_path, args.ex_path, detector, aligner, args.am_frames, args.folders_on_gpu, args.batch_size, args.start)

if __name__ == '__main__':

    args = parser.parse_args()
    main(args)
