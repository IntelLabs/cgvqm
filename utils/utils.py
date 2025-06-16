import torch
import torchvision.transforms.functional as F
from torchvision.io.video import read_video, write_video
from torch.nn.functional import interpolate
from fractions import Fraction

def preprocess(vid):
    need_squeeze = False
    if vid.ndim < 5:
        vid = vid.unsqueeze(dim=0)
        need_squeeze = True

    N, T, C, H, W = vid.shape
    vid = vid.view(-1, C, H, W)
    vid = F.convert_image_dtype(vid, torch.float)
    vid = F.normalize(vid, mean=(0.43216, 0.394666, 0.37645), std=(0.22803, 0.22145, 0.216989))
    vid = vid.view(N, T, C, H, W)
    vid = vid.permute(0, 2, 1, 3, 4)  # (N, T, C, H, W) => (N, C, T, H, W)

    if need_squeeze:
        vid = vid.squeeze(dim=0)
    return vid

def load_resize_vids(test_vid_path, ref_vid_path):
    D, _, metadata_D = read_video(test_vid_path, output_format="TCHW",pts_unit='sec')
    R, _, metadata_R = read_video(ref_vid_path, output_format="TCHW",pts_unit='sec')
    T,C,H,W = R.shape

    if D.shape[2:]!= R.shape[2:]:
        D = interpolate(D,size=(R.shape[2],R.shape[3]),mode='bilinear')
    if D.shape[0]!=R.shape[0]:
        D = D.unsqueeze(0).permute(0,2,1,3,4)
        R = R.unsqueeze(0).permute(0,2,1,3,4)
        D = interpolate(D,size=(R.shape[2],R.shape[3],R.shape[4]),mode='nearest')
        D = D.squeeze().permute(1,0,2,3)
        R = R.squeeze().permute(1,0,2,3)

    return D,R, {'shape':(T,C,H,W),'fps':metadata_R['video_fps']}

def visualize_emap(emap, test_vid_path, scaling_factor, out_path):
    #normalize emap
    emap = torch.clamp(emap/scaling_factor,0,1)
    D, _, metadata_D = read_video(test_vid_path, output_format="TCHW",pts_unit='sec')
    fps = Fraction(metadata_D['video_fps'])
    if emap.shape[1:]!=D.shape[2:]:
        emap = interpolate(emap.unsqueeze(0),size=(D.shape[2],D.shape[3]),mode='bilinear').squeeze()
    if emap.shape[0]!=D.shape[0]:
        emap = emap.unsqueeze(0).unsqueeze(0)
        D = D.unsqueeze(0).permute(0,2,1,3,4)
        emap = interpolate(emap,size=(D.shape[2],D.shape[3],D.shape[4]),mode='nearest')
        D = D.squeeze().permute(1,0,2,3)
        emap = emap.squeeze()

    heatmap = visualize_diff_map(emap.unsqueeze(0).permute(1,0,2,3), context_image=D, type="pmap", colormap_type="threshold").permute(1,0,2,3)
    heatmap = (heatmap * 255).type(torch.ByteTensor).permute(1,2,3,0)
    write_video(out_path,heatmap,fps,video_codec='libx264',options={'crf':'20'})

# Visualization code take from : https://github.com/gfxdisp/FovVideoVDP/blob/main/pyfvvdp/visualize_diff_map.py

def visualize_diff_map(diff_map, context_image=None, type="pmap" , colormap_type="supra-threshold"):
    diff_map = torch.clamp(diff_map, 0.0, 1.0)

    if context_image is None:
        tmo_img = torch.ones_like(diff_map) * 0.5
    else:
        tmo_img = vis_tonemap(log_luminance(context_image), 0.6)

    if colormap_type == 'threshold':

        color_map = torch.tensor([
            [0.2, 0.2, 1.0],
            [0.2, 1.0, 1.0],
            [0.2, 1.0, 0.2],
            [1.0, 1.0, 0.2],
            [1.0, 0.2, 0.2],
        ], device=diff_map.device)
        color_map_in = torch.tensor([0.00, 0.25, 0.50, 0.75, 1.00], device=diff_map.device)

    elif colormap_type == 'supra-threshold':

        color_map = torch.tensor([
            [0.2, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.2],
        ], device=diff_map.device)
        color_map_in = torch.tensor([0.0, 0.5, 1.0], device=diff_map.device)

    elif colormap_type == 'monochromatic':
        
        color_map = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], device=diff_map.device)
        
        color_map_in = torch.tensor([0.0, 1.0], device=diff_map.device)
    else:
        print("Unknown colormap: %s" % colormap_type)

    cmap = torch.zeros_like(diff_map)
    if cmap.shape[1] == 1:
        cmap = torch.cat([cmap]*3, 1)

    color_map_l = color_map[:,0:1] * 0.212656 + color_map[:,1:2] * 0.715158 + color_map[:,2:3] * 0.072186
    color_map_ch = color_map / (torch.cat([color_map_l] * 3, 1) + 0.0001)

    cmap[:,0:1,...] = interp1(color_map_in, color_map_ch[:,0], diff_map)
    cmap[:,1:2,...] = interp1(color_map_in, color_map_ch[:,1], diff_map)
    cmap[:,2:3,...] = interp1(color_map_in, color_map_ch[:,2], diff_map)

    cmap = (cmap * torch.cat([tmo_img]*3, dim=1)).clip(0.,1.)

    return cmap

def get_interpolants_v1(x_q, x):
    imax = torch.bucketize(x_q, x)
    imax[imax >= x.shape[0]] = x.shape[0] - 1
    imin = (imax - 1).clamp(0, x.shape[0] - 1)

    ifrc = (x_q - x[imin]) / (x[imax] - x[imin] + 0.000001)
    ifrc[imax == imin] = 0.
    ifrc[ifrc < 0.0] = 0.

    return imin, imax, ifrc

def interp1(x, v, x_q):
    shp = x_q.shape
    x_q = x_q.flatten()

    imin, imax, ifrc = get_interpolants_v1(x_q, x)

    filtered = v[imin] * (1.0-ifrc) + v[imax] * (ifrc) 

    return filtered.reshape(shp)

def log_luminance(x):
    y = luminance_NCHW(x)
    clampval = torch.min(y[y>0.0])
    return torch.log(torch.clamp(y, min=clampval))


def vis_tonemap(b, dr):
    t = 3.0
    
    b_min = torch.min(b)
    b_max = torch.max(b)
    
    if b_max-b_min < dr: # No tone-mapping needed
        tmo_img = (b/(b_max-b_min+1e03)-b_min)*dr + (1-dr)/2
        return tmo_img

    b_scale = torch.linspace( b_min, b_max, 1024, device=b.device)
    b_p = torch.histc( b, 1024, b_min, b_max )
    b_p = b_p / torch.sum(b_p)
    
    sum_b_p = torch.sum(torch.pow(b_p, 1.0/t))

    dy = torch.pow(b_p, 1.0/t) / sum_b_p
    
    v = torch.cumsum(dy, 0)*dr + (1.0-dr)/2.0
    
    tmo_img = interp1(b_scale, v, b)

    return tmo_img

def luminance_NCHW(x):
    if x.shape[1] == 3: # NC***
        y = (
            x[:,0:1,...] * 0.212656 + 
            x[:,1:2,...] * 0.715158 + 
            x[:,2:3,...] * 0.072186)
    else:
        y = x

    return y