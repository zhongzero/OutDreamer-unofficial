import decord
import os
import math
from functools import wraps
import time

import cv2
import numpy as np
import torch
import imageio

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        est = te - ts
        print(f"\t <{f.__name__}> time = {est:.3f} sec")
        return result
    return wrap

def get_contours(mask):
    contours, _ = cv2.findContours(mask[:, :, 0].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:-1]
    x_min, x_max, y_min, y_max = w, 0, h, 0
    for c in contours:
        x_min = min(c[:, :, 0].min(), x_min)
        x_max = max(c[:, :, 0].max(), x_max)
        y_min = min(c[:, :, 1].min(), y_min)
        y_max = max(c[:, :, 1].max(), y_max)
    return contours, ((x_min, y_min), (x_max, y_max))

@timing
def get_blurred_mask(mask, contours = None, strength = 60):
    result = mask.copy().astype(np.float32)
    merged = mask.copy()
    if not contours:
        contours, _ = get_contours(mask)
    for i in range(2, strength):
        k = (strength - i) / strength
        cv2.drawContours(mask, contours, -1, (255, ), i * 2)
        diff = mask - merged
        merged = (mask - merged) | merged
        result = result + diff.astype(np.float32) * k
    return result / 255.

def blend_with_mask(image, mask, original_image):
    img = image.astype(np.float32) * mask + original_image.astype(np.float32) * (1 - mask)
    return img.astype(np.uint8)

def video_stats(video, m):
    # r, g, b = video[...,0][m[...,0]], video[...,1][m[...,1]], video[...,2][m[...,2]]
    r, g, b = video[...,0], video[...,1], video[...,2]
    rMean, rStd = r.mean(), r.std()
    gMean, gStd = g.mean(), g.std()
    bMean, bStd = b.mean(), b.std()
    return rMean, rStd, gMean, gStd, bMean, bStd

@timing
def align_colors(source_video, template_video, target_video, mask):
    m = mask.astype(bool)
    m = np.repeat(np.expand_dims(m,0), 3, axis=0)
    template_video = template_video.astype("float32")
    source_video = source_video.astype("float32")
    
    rMeanSrc, rStdSrc, gMeanSrc, gStdSrc, bMeanSrc, bStdSrc = video_stats(source_video, m)
    rMeanTmpl, rStdTmpl, gMeanTmpl, gStdTmpl, bMeanTmpl, bStdTmpl = video_stats(template_video, m)
    
    ans1 = np.zeros_like(source_video).astype("uint8")
    for i in range(len(source_video)):
        r, g, b = cv2.split(source_video[i])
        r -= rMeanSrc
        g -= gMeanSrc
        b -= bMeanSrc
        
        r = (rStdTmpl / rStdSrc) * r
        g = (gStdTmpl / gStdSrc) * g
        b = (bStdTmpl / bStdSrc) * b
        
        r += rMeanTmpl
        g += gMeanTmpl
        b += bMeanTmpl
        
        r = np.clip(r, 0, 255)
        g = np.clip(g, 0, 255)
        b = np.clip(b, 0, 255)
        
        ans1[i] = cv2.merge([r, g, b]).astype("uint8")
    ans2 = np.zeros_like(target_video)
    for i in range(len(target_video)):
        tmp = target_video[i].astype("float32")
        r, g, b = cv2.split(tmp)
        r -= rMeanSrc
        g -= gMeanSrc
        b -= bMeanSrc
        
        r = (rStdTmpl / rStdSrc) * r
        g = (gStdTmpl / gStdSrc) * g
        b = (bStdTmpl / bStdSrc) * b
        
        r += rMeanTmpl
        g += gMeanTmpl
        b += bMeanTmpl
        
        r = np.clip(r, 0, 255)
        g = np.clip(g, 0, 255)
        b = np.clip(b, 0, 255)
        
        ans2[i] = cv2.merge([r, g, b]).astype("uint8")
    return ans1, ans2

def match_histograms(source_video, template_video, target_video, mask = None, histSize = 256, accumulate = False):
    mask = mask.astype(bool)
    mask = np.repeat(np.expand_dims(mask, 0), 3, axis=0)
    mask = mask[...,0]
    ans2 = target_video.copy()
    for i in range(3):
        # print(source_video.shape)
        # src_hist = cv2.calcHist([source_video[..., i][mask].reshape(-1)], [0], None, [histSize], (0, histSize), accumulate=accumulate)[:, 0].astype(np.int32)
        # tmpl_hist = cv2.calcHist([template_video[..., i][mask].reshape(-1)], [0], None, [histSize], (0, histSize), accumulate=accumulate)[:, 0].astype(np.int32)
        src_hist = cv2.calcHist([source_video[..., i].reshape(-1)], [0], None, [histSize], (0, histSize), accumulate=accumulate)[:, 0].astype(np.int32)
        tmpl_hist = cv2.calcHist([template_video[..., i].reshape(-1)], [0], None, [histSize], (0, histSize), accumulate=accumulate)[:, 0].astype(np.int32)
        
        src_quantiles = np.cumsum(src_hist) / sum(src_hist)
        tmpl_quantiles = np.cumsum(tmpl_hist) / sum(tmpl_hist)
        
        tmpl_values = np.arange(0, histSize)
        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
        for j in range(len(source_video)):
            src_lookup = source_video[j][:,:,i].reshape(-1)
            interp = interp_a_values[src_lookup]
            source_video[j][:,:,i] = interp.reshape(source_video[j][:,:,i].shape).astype("uint8")
        for j in range(len(ans2)):
            src_lookup = ans2[j][...,i].reshape(-1)
            interp = interp_a_values[src_lookup]
            ans2[j][...,i] = interp.reshape(ans2[j][...,i].shape).astype("uint8")
    return source_video, ans2


@timing
def align_match(template_video, mask, source_video, target_video, blur_strength = 40, align_color = True, match_hist = True):
    source_video_result = source_video.copy()
    target_video_result = target_video.copy()
    
    contours, min_max_coords = get_contours(mask.copy())
    
    msk2 = 255 - mask
    if align_color:
        source_video_result, target_video_result = align_colors(source_video.copy(), template_video.copy(), target_video_result, msk2 + 1)
    
    if match_hist:
        source_video_result, target_video_result = match_histograms(source_video_result, template_video.copy(), target_video_result, msk2 + 1)
    
    blurred_mask = get_blurred_mask(mask[:, :, 0].copy(), contours, strength=blur_strength)[..., None]
    blurred_mask = np.repeat(blurred_mask, 3, axis=-1)
    
    for i in range(len(target_video_result)):
        result = blend_with_mask(target_video_result[i], blurred_mask, target_video[i].copy())
        target_video_result[i] = result
    return target_video_result


def readVideo(video_path):
    decord_vr = decord.VideoReader(video_path)
    # print("video length:", len(decord_vr))
    video_data = decord_vr.get_batch([i for i in range(0, len(decord_vr))]).asnumpy()
    return video_data # T H W C

def create_mask(H, W, mask_ratio_l, mask_ratio_r, mask_ratio_u, mask_ratio_d):
    start_h = math.floor(H * mask_ratio_u)
    start_w = math.floor(W * mask_ratio_l)
    end_h = math.floor(H * mask_ratio_d)
    end_w = math.floor(W * mask_ratio_r)
    target_height = H - start_h - end_h
    target_width = W - start_w - end_w
    mask = np.ones((H, W), dtype=np.uint8) * 255
    print(f"start_h={start_h}, target_height={target_height}, start_w={start_w}, target_width={target_width}")
    mask[start_h:start_h+target_height, start_w:start_w+target_width] = 0
    # (H, W) -> (H, W, 3)
    mask = np.repeat(mask[..., None], 3, axis=-1)
    
    return mask

def refine_video_tensor(video, result_video, mask, align_color = True, match_hist = True):
    # video: (T1,H,W,C)
    # result_video: (T2,H,W,C)
    # mask: (H,W,C) unmask:0/mask:255
    # video is last T2 frames from previous frames, result_video is current frames, align result_video with video
    video = video.numpy()
    result_video = result_video.numpy()
    assert video[0].shape == result_video[0].shape
    T = result_video.shape[0]
    T2 = video.shape[0]
    print(f"T: {T}, T2: {T2}")
    template_video = video[0:T2]
    source_video = result_video[0:T2]
    target_video = result_video
    refine_video = align_match(template_video, mask, source_video, target_video, align_color = align_color, match_hist = match_hist) # result: (H,W,C)
    return torch.from_numpy(refine_video)


def refine_video(video_path, result_video_path, save_path, mask, align_color = True, match_hist = True, compare_frame = 3):
    video = readVideo(video_path)
    result_video = readVideo(result_video_path)
    # print(video.shape)
    # print(result_video.shape)
    # assert video.shape == result_video.shape
    # video/result_video: (T,H,W,C)
    # mask: (H,W,C) unmask:0/mask:255
    T = result_video.shape[0]
    T2 = compare_frame
    video = video[-T2:]
    template_video = video[0:T2]
    source_video = result_video[0:T2]
    target_video = result_video
    refine_video = align_match(template_video, mask, source_video, target_video, align_color = align_color, match_hist = match_hist) # result: (H,W,C)
    
    save_dir = save_path[0:save_path.rfind('/')]
    os.makedirs(save_dir, exist_ok=True)
    imageio.mimwrite(save_path, refine_video, fps=24, quality=6)
    print(f"video save in: {save_path}")
    output_video = torch.tensor(np.array(refine_video)).unsqueeze(0) # B(1) T H W C
    # print(output_video.shape)
    return output_video

if __name__ == "__main__":
    W = 640
    H = 480
    mask_ratio_l = 0.33
    mask_ratio_r = 0.33
    mask_ratio_u = 0.0
    mask_ratio_d = 0.0
    start_h = math.floor(H * mask_ratio_u)
    start_w = math.floor(W * mask_ratio_l)
    end_h = math.floor(H * mask_ratio_d)
    end_w = math.floor(W * mask_ratio_r)
    target_height = H - start_h - end_h
    target_width = W - start_w - end_w
    
    mask = create_mask(H, W, mask_ratio_l, mask_ratio_r, mask_ratio_u, mask_ratio_d)
    save_path_list = ['/path/refine.mp4']
    video_path_list = ['/path/pre_video.mp4']
    result_video_path_list = ['/path/current_video.mp4']
    
    # video_grids = []
    for (video_path, result_video_path, save_path) in zip(video_path_list, result_video_path_list, save_path_list):
        if not os.path.exists(video_path) or not os.path.exists(result_video_path):
            print(video_path, result_video_path, "not exist")
            continue
        video = refine_video(video_path, result_video_path, save_path, mask, align_color = True, match_hist = True, compare_frame = 3)
