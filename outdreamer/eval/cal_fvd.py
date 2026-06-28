import decord
import numpy as np
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    pass
import math
import torch.nn.functional as F

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x / 255.0

def calculate_fvd(videos1, videos2, mask, device, method='styleganv'):

    if method == 'styleganv':
        from outdreamer.eval.fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from outdreamer.eval.fvd.videogpt.fvd import load_i3d_pretrained
        from outdreamer.eval.fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats
        from outdreamer.eval.fvd.videogpt.fvd import frechet_distance

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]
    
    videos2 = videos2 * (1 - mask) + videos1 * mask

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = {}
    
    # videos1_tmp_list = []
    # videos2_tmp_list = []
    # for i in range(videos1.shape[0]):
    #     videos1_tmp_list.append(F.interpolate(videos1[i], (256,256), align_corners=False, mode='bilinear'))
    # for i in range(videos2.shape[0]):
    #     videos2_tmp_list.append(F.interpolate(videos2[i], (256,256), align_corners=False, mode='bilinear'))
    # videos1 = torch.stack(videos1_tmp_list)
    # videos2 = torch.stack(videos2_tmp_list)
    
    # get FVD features
    feats1 = get_fvd_feats(videos1, i3d=i3d, device=device)
    feats2 = get_fvd_feats(videos2, i3d=i3d, device=device)
    
    # calculate FVD when timestamps[:clip]
    fvd_results = frechet_distance(feats1, feats2)

    result = {
        "value": fvd_results,
        "video_setting": videos1.shape,
        "video_setting_name": "batch_size, channel, time, heigth, width",
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

def cal_fvd(GT_video_path_list, gen_video_path_list, start_h, target_height, start_w, target_width):
    videos1 = []
    videos2 = []
    clip1_map = {}
    clip2_map = {}
    
    for i,(GT_path,output_path) in enumerate(zip(GT_video_path_list,gen_video_path_list)):
        videos1.append(readVideo(GT_path))
        videos2.append(readVideo(output_path))
    videos1 = torch.concat(videos1, dim = 0)
    videos2 = torch.concat(videos2, dim = 0)
    
    mask = torch.zeros_like(videos1)
    mask[:,:,:,start_h:start_h+target_height,start_w:start_w+target_width] = 1
    print("videos1.shape:", videos1.shape)
    print("videos2.shape:", videos2.shape)
    print("mask_ratio:", (1 - mask).sum() / torch.ones_like(videos1).sum())
    
    import json
    device = torch.device("cuda")
    # device = torch.device("cpu")
    
    result = calculate_fvd(videos1, videos2, mask, device, method='videogpt')
    print(json.dumps(result, indent=4))
    
    return result["value"]

if __name__ == "__main__":
    GT_video_path_list = ["/path/GT.mp4"]
    gen_video_path_list = ["/path/result.mp4"]
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
    cal_fvd(GT_video_path_list, gen_video_path_list, start_h, target_height, start_w, target_width)