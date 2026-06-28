# import decord
import math
import os
import torch
import argparse
import torchvision
import decord

from diffusers.schedulers import (DDIMScheduler, DDPMScheduler, PNDMScheduler,
                                  EulerDiscreteScheduler, DPMSolverMultistepScheduler,
                                  HeunDiscreteScheduler, EulerAncestralDiscreteScheduler,
                                  DEISMultistepScheduler, KDPM2AncestralDiscreteScheduler)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from transformers import MT5EncoderModel, AutoTokenizer

from outdreamer.models.causalvideovae import ae_stride_config, CausalVAEModelWrapper

from outdreamer.sample.pipeline_outpaint import OutpaintPipeline

from outdreamer.models.diffusion.outdreamer.modeling_cnext import OpenSoraCNext

from outdreamer.refine.refine import create_mask, refine_video_tensor

import imageio


def main(args):
    weight_dtype = torch.bfloat16
    device = torch.device(args.device)

    vae = CausalVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]
    
    transformer_model = OpenSoraCNext.from_pretrained(args.model_path, cache_dir=args.cache_dir, 
                                                        low_cpu_mem_usage=False, device_map=None, torch_dtype=weight_dtype)
    
    text_encoder = MT5EncoderModel.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir, low_cpu_mem_usage=True, torch_dtype=weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)
    
    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == 'DDIM':  #########
        scheduler = DDIMScheduler(clip_sample=False)
    elif args.sample_method == 'EulerDiscrete':
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == 'DDPM':  #############
        scheduler = DDPMScheduler(clip_sample=False)
    elif args.sample_method == 'DPMSolverMultistep':
        '''
        DPM++ 2M	        DPMSolverMultistepScheduler	
        DPM++ 2M Karras	    DPMSolverMultistepScheduler	init with use_karras_sigmas=True
        DPM++ 2M SDE	    DPMSolverMultistepScheduler	init with algorithm_type="sde-dpmsolver++"
        DPM++ 2M SDE Karras	DPMSolverMultistepScheduler	init with use_karras_sigmas=True and algorithm_type="sde-dpmsolver++"
        
        DPM++ SDE	        DPMSolverSinglestepScheduler	
        DPM++ SDE Karras	DPMSolverSinglestepScheduler	init with use_karras_sigmas=True
        DPM2	            KDPM2DiscreteScheduler	
        DPM2 Karras	        KDPM2DiscreteScheduler	init with use_karras_sigmas=True
        DPM2 a	            KDPM2AncestralDiscreteScheduler	
        DPM2 a Karras	    KDPM2AncestralDiscreteScheduler	init with use_karras_sigmas=True
        '''
        # scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
        scheduler = DPMSolverMultistepScheduler()
    elif args.sample_method == 'DPMSolverSinglestep':
        scheduler = DPMSolverSinglestepScheduler()
    elif args.sample_method == 'PNDM':
        scheduler = PNDMScheduler()
    elif args.sample_method == 'HeunDiscrete':  ########
        scheduler = HeunDiscreteScheduler()
    elif args.sample_method == 'EulerAncestralDiscrete':
        scheduler = EulerAncestralDiscreteScheduler()
    elif args.sample_method == 'DEISMultistep':
        scheduler = DEISMultistepScheduler()
    elif args.sample_method == 'KDPM2AncestralDiscrete':  #########
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif args.sample_method == 'EulerDiscreteSVD':
        scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-video-diffusion-img2vid", 
                                                        subfolder="scheduler", cache_dir=args.cache_dir)
    pipeline = OutpaintPipeline(vae=vae,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                scheduler=scheduler,
                                transformer=transformer_model)
    pipeline.to(device)
    pipeline.set_aeType(args.ae)
        
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith('txt'):
        text_prompt = open(args.text_prompt[0], 'r').readlines()
        text_prompt = [i.strip() for i in text_prompt]
    
    def readVideo(video_path):
        decord_vr = decord.VideoReader(video_path)
        video_data = decord_vr.get_batch([i for i in range(len(decord_vr))]).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
        return video_data
    
    if not isinstance(args.origin_video, list):
        args.origin_video = [args.origin_video]
    if len(args.origin_video) == 1 and args.origin_video[0].endswith('txt'):
        origin_video_paths = open(args.origin_video[0], 'r').readlines()
        origin_video_paths = [i.strip() for i in origin_video_paths]
        origin_videos = [readVideo(origin_video_path) for origin_video_path in origin_video_paths]
    
    positive_prompt = """
    (masterpiece), (best quality), (ultra-detailed), 
    {}. 
    """
    
    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """

    assert args.given_frame_num < args.num_frames, f"args.given_frame_num {args.given_frame_num} should < args.num_frames {args.num_frames}"
    for idx, (prompt, video) in enumerate(zip(text_prompt, origin_videos)):
        # video: (T,C,H',W')
        assert video.shape[0] >= args.num_frames, f"video frames length {video.shape[0]} should >= setting num_frames {args.num_frames}"
        total_T, C = video.shape[0], video.shape[1]
        beginFrame = 0
        preGen_video = None
        total_video = torch.zeros((total_T, args.height, args.width, C)).to(torch.uint8)
        count = 0
        # total_video: (T,H,W,C)
        firstGen = None
        while beginFrame < total_T:
            if preGen_video == None:
                maskLen = args.num_frames
                unmaskLen = 0
            else:
                maskLen = args.num_frames - args.given_frame_num
                unmaskLen = args.given_frame_num
            if total_T - beginFrame < maskLen:
                beginFrame = total_T - maskLen
                preGen_video = total_video[beginFrame-unmaskLen:beginFrame].permute(0, 3, 1, 2)
            current_video = video[beginFrame:beginFrame+maskLen]
            # current_video: (T',C,H',W')
            if unmaskLen != 0:
                preGen_video = preGen_video[-unmaskLen:]
            print(f"start generate frames: {beginFrame}-{beginFrame+maskLen-1}, using previous generated frames number: {unmaskLen}")
            gen_video = pipeline(given_video=current_video,
                            preGen_video=preGen_video,
                            mask_ratio_l=args.mask_ratio_l,
                            mask_ratio_r=args.mask_ratio_r,
                            mask_ratio_u=args.mask_ratio_u,
                            mask_ratio_d=args.mask_ratio_d,
                            prompt=positive_prompt.format(prompt),
                            negative_prompt=negative_prompt, 
                            num_frames=args.num_frames,
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.num_sampling_steps,
                            guidance_scale=args.guidance_scale,
                            attention_scale = args.attention_scale,
                            num_images_per_prompt=1,
                            mask_feature=True,
                            device=args.device, 
                            max_sequence_length=args.max_sequence_length, 
                            ).images[0]
            # gen_video: (T',H,W,C)
            mask = create_mask(H = args.height, W = args.width, 
                               mask_ratio_l = args.mask_ratio_l, mask_ratio_r = args.mask_ratio_r, 
                               mask_ratio_u = args.mask_ratio_u, mask_ratio_d = args.mask_ratio_d)
            if firstGen == None:
                firstGen = gen_video
            # mask: (H,W,C)
            if unmaskLen != 0:
                # refine_video = refine_video_tensor(preGen_video.permute(0,2,3,1), gen_video, mask, align_color = True, match_hist = True)[-maskLen:]
                refine_video = refine_video_tensor(preGen_video.permute(0,2,3,1)[-1:], gen_video[-(maskLen+1):], mask, align_color = True, match_hist = True)[-maskLen:]
            else:
                refine_video = gen_video[-maskLen:]
            # refine_video: (T',H,W,C)
            preGen_video = refine_video.permute(0,3,1,2)
            # preGen_video: (T',C,H,W)
            total_video[beginFrame:beginFrame+maskLen] = refine_video
            beginFrame += maskLen
            count += 1
        try:
            imageio.mimwrite(
                args.save_video_path, total_video, fps=args.fps, quality=6)  # highest quality is 10, lowest is 0
        except:
            print('Error when saving {}'.format(prompt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_video", type=str, required=True)
    parser.add_argument("--model_path", type=str, default='LanguageBind/Open-Sora-Plan-v1.0.0')
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cache_dir", type=str, default='./cache_dir')
    parser.add_argument("--ae", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--ae_path", type=str, default='CausalVAEModel_4x8x8')
    parser.add_argument("--text_encoder_name", type=str, default='DeepFloyd/t5-v1_1-xxl')
    parser.add_argument("--save_video_path", type=str, default=None, required=True)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--attention_scale", type=float, default=0.1)
    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--max_sequence_length", type=int, default=300)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--text_prompt", nargs='+')
    parser.add_argument('--tile_overlap_factor', type=float, default=0.125)
    parser.add_argument('--enable_tiling', action='store_true')
    parser.add_argument('--save_memory', action='store_true')
    parser.add_argument('--given_frame_num', type=int, default=0)
    parser.add_argument('--mask_ratio_l', type=float, default=0.125) # 0.33
    parser.add_argument('--mask_ratio_r', type=float, default=0.125) # 0.33
    parser.add_argument('--mask_ratio_u', type=float, default=0.0)
    parser.add_argument('--mask_ratio_d', type=float, default=0.0)
    args = parser.parse_args()

    main(args)