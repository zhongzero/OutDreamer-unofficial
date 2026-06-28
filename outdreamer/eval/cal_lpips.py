import decord
import numpy as np
import torch
from tqdm import tqdm
import math

import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass
import lpips
import torch.nn.functional as F

spatial = True         # Return a spatial map of perceptual distance.

# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # value range [0, 1] -> [-1, 1]
    x = x / 255.0
    x = x * 2 - 1

    return x

def calculate_lpips(videos1, videos2, mask, device):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    print("calculate_lpips...")

    assert videos1.shape == videos2.shape

    # videos [batch_size, timestamps, channel, h, w]

    # support grayscale input, if grayscale -> channel*3
    # value range [0, 1] -> [-1, 1]
    
    videos2 = videos2 * (1 - mask) + videos1 * mask
    
    videos1 = trans(videos1)
    videos2 = trans(videos2)

    lpips_results = []

    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        
        # video1 = F.interpolate(video1, (256,256), align_corners=False, mode='bilinear')
        # video2 = F.interpolate(video2, (256,256), align_corners=False, mode='bilinear')

        lpips_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] tensor

            img1 = video1[clip_timestamp].unsqueeze(0).to(device)
            img2 = video2[clip_timestamp].unsqueeze(0).to(device)
            
            loss_fn.to(device)

            # calculate lpips of a video
            lpips_results_of_a_video.append(loss_fn.forward(img1, img2).mean().detach().cpu().tolist())
        lpips_results.append(lpips_results_of_a_video)
    
    lpips_results = np.array(lpips_results)
    
    lpips = {}
    lpips_std = {}

    for clip_timestamp in range(len(video1)):
        lpips[clip_timestamp] = np.mean(lpips_results[:,clip_timestamp])
        lpips_std[clip_timestamp] = np.std(lpips_results[:,clip_timestamp])


    result = {
        "value": np.mean(list(lpips.values())),
        # "value": lpips,
        # "value_std": lpips_std,
        "video_setting": video1.shape,
        "video_setting_name": "time, channel, heigth, width",
    }

    return result

# test code / using example

def readVideo(video_path):
    decord_vr = decord.VideoReader(video_path)
    print(f"len: {len(decord_vr)}")
    video_data = decord_vr.get_batch([i for i in range(len(decord_vr))]).asnumpy()
    video_data = torch.from_numpy(video_data)
    print(video_data.shape)
    video_data = video_data.permute(0,3,1,2).unsqueeze(0) # (T,H,W,C) -> (B(1),T,C,H,W)
    print(video_data.shape)
    return video_data

def cal_lpips(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width):
    videos1 = readVideo(GT_video_path)
    videos2 = readVideo(gen_video_path)
    
    mask = torch.zeros_like(videos1)
    mask[:,:,:,start_h:start_h+target_height,start_w:start_w+target_width] = 1
    print("videos1.shape:", videos1.shape)
    print("videos2.shape:", videos2.shape)
    print("mask_ratio:", (1 - mask).sum() / torch.ones_like(videos1).sum())

    import json
    device = torch.device("cuda")
    result = calculate_lpips(videos1, videos2, mask, device)
    print(json.dumps(result, indent=4))
    return result["value"]

if __name__ == "__main__":
    GT_video_path = "/path/GT.mp4"
    gen_video_path = "/path/result.mp4"
    W = 640
    H = 480
    mask_ratio_l = 0.125
    mask_ratio_r = 0.125
    mask_ratio_u = 0.0
    mask_ratio_d = 0.0
    start_h = math.floor(H * mask_ratio_u)
    start_w = math.floor(W * mask_ratio_l)
    end_h = math.floor(H * mask_ratio_d)
    end_w = math.floor(W * mask_ratio_r)
    target_height = H - start_h - end_h
    target_width = W - start_w - end_w
    print(f"start_h={start_h}, target_height={target_height}, start_w={start_w}, target_width={target_width}")
    cal_lpips(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width)