import decord
import numpy as np
import torch
from tqdm import tqdm
import cv2
import math
import torch.nn.functional as F
 
def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
 
 
def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w 
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()                   
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def trans(x):
    return x / 255.0

def calculate_ssim(videos1, videos2, mask):
    print("calculate_ssim...")

    # videos [batch_size, timestamps, channel, h, w]

    assert videos1.shape == videos2.shape
    
    videos2 = videos2 * (1 - mask) + videos1 * mask

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    ssim_results = []
    
    for video_num in tqdm(range(videos1.shape[0])):
        # get a video
        # video [timestamps, channel, h, w]
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        
        # video1 = F.interpolate(video1, (256,256), align_corners=False, mode='bilinear')
        # video2 = F.interpolate(video2, (256,256), align_corners=False, mode='bilinear')

        ssim_results_of_a_video = []
        for clip_timestamp in range(len(video1)):
            # get a img
            # img [timestamps[x], channel, h, w]
            # img [channel, h, w] numpy

            img1 = video1[clip_timestamp].numpy()
            img2 = video2[clip_timestamp].numpy()
            
            # calculate ssim of a video
            ssim_results_of_a_video.append(calculate_ssim_function(img1, img2))

        ssim_results.append(ssim_results_of_a_video)

    ssim_results = np.array(ssim_results)

    ssim = {}
    ssim_std = {}

    for clip_timestamp in range(len(video1)):
        ssim[clip_timestamp] = np.mean(ssim_results[:,clip_timestamp])
        ssim_std[clip_timestamp] = np.std(ssim_results[:,clip_timestamp])

    result = {
        "value": np.mean(list(ssim.values())),
        # "value": ssim,
        # "value_std": ssim_std,
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

def cal_ssim(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width):
    videos1 = readVideo(GT_video_path)
    videos2 = readVideo(gen_video_path)
    
    mask = torch.zeros_like(videos1)
    mask[:,:,:,start_h:start_h+target_height,start_w:start_w+target_width] = 1
    print("videos1.shape:", videos1.shape)
    print("videos2.shape:", videos2.shape)
    print("mask_ratio:", (1 - mask).sum() / torch.ones_like(videos1).sum())
    
    import json
    result = calculate_ssim(videos1, videos2, mask)
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
    cal_ssim(GT_video_path, gen_video_path, start_h, target_height, start_w, target_width)