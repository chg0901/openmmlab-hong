import argparse
import numpy as np
import cv2
import mmcv

from mmdet.apis import inference_detector, init_detector

# 　https://mmdetection.readthedocs.io/en/3.x/user_guides/inference.html
#   https://github.com/open-mmlab/mmcv/blob/master/mmcv/video/io.py
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection balloon color splash video demo')
    parser.add_argument('video_path', help='path of video file ')
    parser.add_argument('model_config', help='path of config file ')
    parser.add_argument('checkpoint_path', help='path of fine-tuned checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score_thr', type=float, default=0.8, help='Bbox score threshold')
    parser.add_argument('--output_path', type=str, help='path of processed video file')
    parser.add_argument('--display_video', action='store_true', help='display the processed video')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of display (s), 0 is block, default is 1')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.output_path or args.display_video, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--output_path" or "--display_video"')
    # print(args.model_config,args.checkpoint_path)
    model = init_detector(args.model_config,
                          args.checkpoint_path, device=args.device)

    video_reader = mmcv.VideoReader(args.video_path)
    video_writer = None
    if args.output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.output_path, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):

        # frame read in bgr not rgb
        result = inference_detector(model, frame)
        mask = None
        masks = result[1][0]

        # take mask of balloon for compute
        for i in range(len(masks)):
            if result[0][0][i][-1] >= args.score_thr:
                if not mask is None:
                    mask = mask | masks[i]
                else:
                    mask = masks[i]

        # change color of background and balloon
        masked_b = frame[:, :, 0] * mask
        masked_g = frame[:, :, 1] * mask
        masked_r = frame[:, :, 2] * mask
        masked = np.concatenate([masked_b[:, :, None], masked_g[:, :, None], masked_r[:, :, None]], axis=2)

        un_mask = 1 - mask
        frame_b = frame[:, :, 0] * un_mask
        frame_g = frame[:, :, 1] * un_mask
        frame_r = frame[:, :, 2] * un_mask
        frame = np.concatenate([frame_b[:, :, None], frame_g[:, :, None], frame_r[:, :, None]], axis=2).astype(np.uint8)

        # background gray color with 3 channels
        frame = mmcv.bgr2gray(frame, keepdim=True)
        frame = np.concatenate([frame, frame, frame], axis=2)

        frame += masked

        if args.display_video:
            cv2.namedWindow('The processed video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)

        if args.output_path:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()