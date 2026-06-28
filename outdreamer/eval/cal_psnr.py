import decord
import numpy as np
import torch
from tqdm import tqdm
import math
import torch.nn.functional as F

def img_psnr(img1, img2):
    # [0,1]
    # compute mse
    # mse = np.mean((img1-img2)**2)
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

def trans(x):
    return x / 255.0

def calculate_psnr(videos1, videos2, mask):
    print("calculate_psnr...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape
    
    videos2 = videos2 * (1 - mask) + videos1 * mask

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    psnr_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        
        # video1 = F.interpolate(video1, (256,256), align_corners=False, mode='bilinear')
        # video2 = F.interpolate(video2, (256,256), align_corners=False, mode='bilinear')

        psnr_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()
            
            # calculate psnr of a video
            psnr_results_of_a_video.append(img_psnr(img1, img2))

        psnr_results.append(psnr_results_of_a_video)
    
    psnr_results = np.array(psnr_results) # [batch_size, num_frames]
    psnr = {}
    psnr_std = {}

    for clip_timestamp in range(len(video1)):
        psnr[clip_timestamp] = np.mean(psnr_results[:,clip_timestamp])
        psnr_std[clip_timestamp] = np.std(psnr_results[:,clip_timestamp])

    result = {
        "value": np.mean(list(psnr.values())),
        # "value": psnr,
        # "value_std": psnr_std,
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

def cal_psnr(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width):
    videos1 = readVideo(GT_video_path)
    videos2 = readVideo(gen_video_path)
    
    mask = torch.zeros_like(videos1)
    mask[:,:,:,start_h:start_h+target_height,start_w:start_w+target_width] = 1
    print("videos1.shape:", videos1.shape)
    print("videos2.shape:", videos2.shape)
    print("mask_ratio:", (1 - mask).sum() / torch.ones_like(videos1).sum())
    
    import json
    result = calculate_psnr(videos1, videos2, mask)
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
    cal_psnr(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width)