import os, sys, glob
import numpy as np
from collections import OrderedDict
from decord import VideoReader, cpu
import cv2

import torch
import torchvision
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from einops import rearrange
from tqdm import tqdm

###############################
# Upsampling function
def upsample_noise(X, N, device = None):
    b, c, h, w = X.shape
    Z = torch.randn(b, c, N * h, N * w, device=device)
    Z_mean = Z.unfold(2, N, N).unfold(3, N, N).mean(dim=(4, 5))
    Z_mean = F.interpolate(Z_mean, scale_factor=N, mode='nearest')
    X = F.interpolate(X, scale_factor=N, mode='nearest')
    return X / N + Z - Z_mean

# Vectorized triangulate_area function
def triangulate_area_batch(A_batch, s):
    """
    A_batch: Tensor of shape (N, 2, 2), where N = H * W (number of pixels)
    s: Number of subdivisions
    Returns:
    - V_batch: Tensor of shape (N, (s+1)*(s+1), 2)
    - F: Tensor of shape (2*s*s, 3)
    """
    N = A_batch.shape[0]
    device = A_batch.device

    # Generate subdivision points for one pixel
    lin = torch.linspace(0, 1, s + 1, device=device)
    grid_x, grid_y = torch.meshgrid(lin, lin, indexing='ij')
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # Shape: ((s+1)*(s+1), 2)

    # Compute vertices for all pixels
    A_min = A_batch[:, 0].unsqueeze(1)  # Shape: (N, 1, 2)
    A_max = A_batch[:, 1].unsqueeze(1)  # Shape: (N, 1, 2)
    V_batch = A_min + grid.unsqueeze(0) * (A_max - A_min)  # Shape: (N, (s+1)*(s+1), 2)

    # Faces are the same for all pixels
    F = []
    for i in range(s):
        for j in range(s):
            idx = i * (s + 1) + j
            F.append([idx, idx + 1, idx + s + 1])
            F.append([idx + 1, idx + s + 2, idx + s + 1])
    F = torch.tensor(F, device=device)  # Shape: (2*s*s, 3)

    return V_batch, F

# Vectorized warp_vertices function
def warp_vertices_batch(vertices_batch, T_func):
    """
    vertices_batch: Tensor of shape (N, M, 2)
    Returns:
    - warped_vertices_batch: Tensor of shape (N, M, 2)
    """
    N, M, _ = vertices_batch.shape
    vertices_flat = vertices_batch.view(-1, 2)  # Shape: (N*M, 2)
    warped_vertices_flat = T_func(vertices_flat)
    warped_vertices_batch = warped_vertices_flat.view(N, M, 2)
    return warped_vertices_batch

# Vectorized rasterize function
def rasterize_batch(vertices_batch, shape):
    """
    vertices_batch: Tensor of shape (N, M, 2)
    shape: Output shape (B, C, Hk, Wk)
    Returns:
    - mask: Tensor of shape (N, Hk, Wk)
    """
    N, M, _ = vertices_batch.shape
    _, _, Hk, Wk = shape
    device = vertices_batch.device

    # Initialize masks
    mask = torch.zeros(N, Hk, Wk, dtype=torch.bool, device=device)

    # Flatten coordinates
    x = vertices_batch[..., 0].contiguous().view(-1).long()
    y = vertices_batch[..., 1].contiguous().view(-1).long()
    n_indices = torch.arange(N, device=device).repeat_interleave(M)

    # Filter valid indices
    valid = (x >= 0) & (x < Wk) & (y >= 0) & (y < Hk)
    x = x[valid]
    y = y[valid]
    n_indices = n_indices[valid]

    mask[n_indices, y, x] = True

    return mask

# Main function with vectorized operations
def distribution_preserving_noise_warping_full_frame_parallel(G, T_func, k, s):
    B, C, H, W = G.shape
    device = G.device

    # Upsample the noise
    G_infinity = upsample_noise(G, k, device)  # Shape: (B, C, Hk, Wk)
    Hk, Wk = H * k, W * k

    # Generate grid of pixel positions
    i_grid, j_grid = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    i_grid = i_grid.flatten()
    j_grid = j_grid.flatten()
    N = H * W  # Number of pixels

    # Define pixel areas for all pixels
    A_min = torch.stack([j_grid.float(), i_grid.float()], dim=1)  # Shape: (N, 2)
    A_max = A_min + 1  # Shape: (N, 2)
    A_batch = torch.stack([A_min, A_max], dim=1)  # Shape: (N, 2, 2), N = H*W

    # Triangulate areas
    V_batch, Faces = triangulate_area_batch(A_batch, s)  # V_batch: (N, M, 2)

    # Warp vertices
    warped_V_batch = warp_vertices_batch(V_batch, T_func)  # Shape: (N, M, 2)

    # Rasterize to get masks
    masks = rasterize_batch(warped_V_batch, G_infinity.shape)  # Shape: (N, Hk, Wk)

    # Compute pixel values
    warped_G = torch.zeros_like(G)
    for b in tqdm(range(B), desc="Batch Processing"):
        for c in range(C):
            G_inf_flat = G_infinity[b, c].view(-1)  # Shape: (Hk*Wk)
            mask_flat = masks.view(N, -1)  # Shape: (N, Hk*Wk)
            values = mask_flat.float().matmul(G_inf_flat.unsqueeze(1)).squeeze(1)  # Shape: (N,)
            counts = mask_flat.sum(dim=1).float()  # Shape: (N,)
            counts[counts == 0] = 1  # Avoid division by zero
            pixel_values = values / torch.sqrt(counts)
            warped_G[b, c] = pixel_values.view(H, W)

    return warped_G

# Function to generate the initial noise frame
def generate_initial_noise_frame(channels, height, width, device = None):
    """
    Generates the initial noise frame.
    """
    return torch.randn((1, channels, height, width), device=device)

# Function to initialize correlated noise frames
def initialize_correlated_noise_frames(initial_frame, k, s, num_frames=128, dx=1, dy=1):
    frames = []
    G = initial_frame
    B, C, H, W = G.shape

    for frame_idx in tqdm(range(num_frames), desc="Frame Generation"):
        def T_func(vertices):
            transformed_vertices = vertices.clone()
            transformed_vertices[:, 0] += dx * frame_idx  % W
            transformed_vertices[:, 1] += dy * frame_idx  % H
            return transformed_vertices
            # return inverted_explosion_transportation(vertices, frame_idx, downward_strength, diffusion_scale, width, height)

        warped_G = distribution_preserving_noise_warping_full_frame_parallel(G, T_func, k, s)
        frames.append(warped_G)

    frames = torch.stack(frames, dim=0)  # Shape: (num_frames, B, C, H, W)
    return frames

###############################


def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0,\
                        cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    fs = cond["fs"]
    del cond["fs"]
    if noise_shape[-1] == 32:
        timestep_spacing = "uniform"
        guidance_rescale = 0.0
    else:
        timestep_spacing = "uniform_trailing"
        guidance_rescale = 0.7
    ## construct unconditional guidance
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            #prompts = N * T * [""]  ## if is_imgbatch=True
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)
                
        ## process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0],3,224,224).to(model.device)
            ## img: b c h w >> b l c
            uc_img = model.embedder(uc_img)
            uc_img = model.image_proj_model(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)
        
        if isinstance(cond, dict):
            uc = {key:cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb
    else:
        uc = None
    
    print(noise_shape)
    initial_frame = generate_initial_noise_frame(noise_shape[1], noise_shape[3], noise_shape[4], device = model.device)

    # Define your T_func here
    # ...

    # Upsampling factor and polygon subdivision steps
    k = 4
    s = 2

    # Generate the sequence of noise frames
    x_T = initialize_correlated_noise_frames(
        initial_frame, k, s, num_frames=noise_shape[2], dx=1, dy=0
    )
    x_T = rearrange(x_T, 'f b c h w -> b c f h w')
    print(x_T.shape)
    # x_T = None
    batch_variants = []

    for _ in range(n_samples):
        if ddim_sampler is not None:
            kwargs.update({"clean_cond": True})
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=noise_shape[0],
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=cfg_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            temporal_length=noise_shape[2],
                                            conditional_guidance_scale_temporal=temporal_cfg_scale,
                                            x_T=x_T,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


def get_filelist(data_dir, ext='*'):
    file_list = glob.glob(os.path.join(data_dir, '*.%s'%ext))
    file_list.sort()
    return file_list

def get_dirlist(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            if (os.path.isdir(m)):
                list.append(m)
    list.sort()
    return list


def load_model_checkpoint(model, ckpt):
    def load_checkpoint(model, ckpt, full_strict):
        state_dict = torch.load(ckpt, map_location="cpu")
        if "state_dict" in list(state_dict.keys()):
            state_dict = state_dict["state_dict"]
            try:
                model.load_state_dict(state_dict, strict=full_strict)
            except:
                ## rename the keys for 256x256 model
                new_pl_sd = OrderedDict()
                for k,v in state_dict.items():
                    new_pl_sd[k] = v

                for k in list(new_pl_sd.keys()):
                    if "framestride_embed" in k:
                        new_key = k.replace("framestride_embed", "fps_embedding")
                        new_pl_sd[new_key] = new_pl_sd[k]
                        del new_pl_sd[k]
                model.load_state_dict(new_pl_sd, strict=full_strict)
        else:
            ## deepspeed
            new_pl_sd = OrderedDict()
            for key in state_dict['module'].keys():
                new_pl_sd[key[16:]]=state_dict['module'][key]
            model.load_state_dict(new_pl_sd, strict=full_strict)

        return model
    load_checkpoint(model, ckpt, full_strict=True)
    print('>>> model checkpoint loaded.')
    return model


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_video_batch(filepath_list, frame_stride, video_size=(256,256), video_frames=16):
    '''
    Notice about some special cases:
    1. video_frames=-1 means to take all the frames (with fs=1)
    2. when the total video frames is less than required, padding strategy will be used (repeated last frame)
    '''
    fps_list = []
    batch_tensor = []
    assert frame_stride > 0, "valid frame stride should be a positive interge!"
    for filepath in filepath_list:
        padding_num = 0
        vidreader = VideoReader(filepath, ctx=cpu(0), width=video_size[1], height=video_size[0])
        fps = vidreader.get_avg_fps()
        total_frames = len(vidreader)
        max_valid_frames = (total_frames-1) // frame_stride + 1
        if video_frames < 0:
            ## all frames are collected: fs=1 is a must
            required_frames = total_frames
            frame_stride = 1
        else:
            required_frames = video_frames
        query_frames = min(required_frames, max_valid_frames)
        frame_indices = [frame_stride*i for i in range(query_frames)]

        ## [t,h,w,c] -> [c,t,h,w]
        frames = vidreader.get_batch(frame_indices)
        frame_tensor = torch.tensor(frames.asnumpy()).permute(3, 0, 1, 2).float()
        frame_tensor = (frame_tensor / 255. - 0.5) * 2
        if max_valid_frames < required_frames:
            padding_num = required_frames - max_valid_frames
            frame_tensor = torch.cat([frame_tensor, *([frame_tensor[:,-1:,:,:]]*padding_num)], dim=1)
            print(f'{os.path.split(filepath)[1]} is not long enough: {padding_num} frames padded.')
        batch_tensor.append(frame_tensor)
        sample_fps = int(fps/frame_stride)
        fps_list.append(sample_fps)
    
    return torch.stack(batch_tensor, dim=0)

from PIL import Image
def load_image_batch(filepath_list, image_size=(256,256)):
    batch_tensor = []
    for filepath in filepath_list:
        _, filename = os.path.split(filepath)
        _, ext = os.path.splitext(filename)
        if ext == '.mp4':
            vidreader = VideoReader(filepath, ctx=cpu(0), width=image_size[1], height=image_size[0])
            frame = vidreader.get_batch([0])
            img_tensor = torch.tensor(frame.asnumpy()).squeeze(0).permute(2, 0, 1).float()
        elif ext == '.png' or ext == '.jpg':
            img = Image.open(filepath).convert("RGB")
            rgb_img = np.array(img, np.float32)
            #bgr_img = cv2.imread(filepath, cv2.IMREAD_COLOR)
            #bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_img = cv2.resize(rgb_img, (image_size[1],image_size[0]), interpolation=cv2.INTER_LINEAR)
            img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).float()
        else:
            print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
            raise NotImplementedError
        img_tensor = (img_tensor / 255. - 0.5) * 2
        batch_tensor.append(img_tensor)
    return torch.stack(batch_tensor, dim=0)


def save_videos(batch_tensors, savedir, filenames, fps=10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z