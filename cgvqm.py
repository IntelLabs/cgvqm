from utils import resnet18
from utils.utils import preprocess, load_resize_vids, visualize_emap
from enum import Enum

import torch
import pickle # nosec
import torch.nn as nn
from torch import Tensor
import torchvision.transforms.functional as F
import numpy as np
import os

class CGVQM_TYPE(Enum):
    CGVQM_2 = 1
    CGVQM_5 = 2

class CGVQM(resnet18.VideoResNet):

    def init_weights(self, weights_file=None, num_layers=6):
            device = self.get_device()
            self.num_layers = num_layers
            self.chns = [3,64,64,128,256,512][:num_layers]
            self.register_parameter("feature_weights", nn.Parameter(torch.ones(1,sum(self.chns),1,1,1).to(device)))
            self.register_parameter("alpha", nn.Parameter(torch.tensor(1.).to(device)))        
            if weights_file!=None:
                with open(weights_file, 'rb') as fp:
                    wo,ao = pickle.load(fp) # nosec
                    assert(sum(self.chns)==wo.shape[1])
                    self.feature_weights.data = wo.to(device)
                    self.alpha.data = ao.to(device)

    def get_device(self):
        return next(self.parameters()).device

    def feature_diff(self, x: Tensor, y: Tensor, cache_path=None):

        def fdiff(a,b,w, normalize_channel=True):
            #unit normalize in channel dimension
            eps=1e-10
            if normalize_channel:
                norm_a = a.pow(2).sum(dim=1,keepdim=True).sqrt()
                a = a/(norm_a+eps)
                norm_b = b.pow(2).sum(dim=1,keepdim=True).sqrt()
                b = b/(norm_b+eps)
            diff = (w*((a - b)).pow(2).mean([2,3,4],keepdim=True)).sum()
            emap = (w*((a - b)).pow(2)).sum(1,keepdim=True)
            emap = torch.nn.functional.interpolate(emap,size=tuple(x.shape[2:]),mode='trilinear')
            cache = ((a - b)).pow(2).mean([2,3,4],keepdim=True) if cache_path!=None else None

            return diff, emap, cache
        
        feature_weights = torch.split(self.feature_weights.abs(), self.chns, dim=1)
        
        cache = []
        diff = 0
        emap = torch.zeros(x.shape[0],1,x.shape[2],x.shape[3],x.shape[4]).to(self.get_device())

        d,em,cache_n = fdiff(x,y,feature_weights[0],False)
        diff += d
        emap += em
        
        if cache_path!=None: cache.append(cache_n.cpu())

        if self.num_layers==1: return self.alpha*diff, self.alpha*(emap)

        hx, hy = self.stem(x), self.stem(y)
        d,em,cache_n =  fdiff(hx,hy,feature_weights[1])
        diff += d
        emap += em

        if cache_path!=None: cache.append(cache_n.cpu())
        if self.num_layers==2: return self.alpha*diff, self.alpha*(emap)

        hx, hy = self.layer1(hx), self.layer1(hy)
        d,em,cache_n =  fdiff(hx,hy,feature_weights[2])
        diff += d
        emap += em
        if cache_path!=None: cache.append(cache_n.cpu())
        if self.num_layers==3: return self.alpha*diff, self.alpha*(emap)

        hx, hy = self.layer2(hx), self.layer2(hy)
        d,em,cache_n =  fdiff(hx,hy,feature_weights[3])
        diff += d
        emap += em
        if cache_path!=None: cache.append(cache_n.cpu())
        if self.num_layers==4: return self.alpha*diff, self.alpha*(emap)

        hx, hy = self.layer3(hx), self.layer3(hy)
        d,em,cache_n =  fdiff(hx,hy,feature_weights[4])
        diff += d
        emap += em
        if cache_path!=None: cache.append(cache_n.cpu())
        if self.num_layers==5: return self.alpha*diff, self.alpha*(emap)

        hx, hy = self.layer4(hx), self.layer4(hy)
        d,em,cache_n =  fdiff(hx,hy,feature_weights[5])
        diff += d
        emap += em
        if cache_path!=None: cache.append(cache_n.cpu())

        if cache_path!=None: 
            with open(cache_path, 'wb') as fp:
                pickle.dump(cache, fp)

        return self.alpha*diff, self.alpha*(emap)


def run_cgvqm(test_vid_path, ref_vid_path, cgvqm_type = CGVQM_TYPE.CGVQM_2, device='cpu', patch_pool='max', patch_scale=4):

    # Load model
    model = resnet18.r3d_18(weights=resnet18.R3D_18_Weights.DEFAULT).to(device)
    model.__class__ = CGVQM
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if cgvqm_type==CGVQM_TYPE.CGVQM_2:
        model.init_weights(os.path.join(dir_path,'weights','cgvqm-2.pickle'),num_layers=3)
    elif cgvqm_type==CGVQM_TYPE.CGVQM_5:
        model.init_weights(os.path.join(dir_path,'weights','cgvqm-5.pickle'),num_layers=6)
    else:
        raise Exception("ERROR: unkown model type")
    model.eval()

    # Load, resize, and normalize videos
    D,R,metadata = load_resize_vids(test_vid_path, ref_vid_path)
    D = preprocess(D).unsqueeze(0)
    R = preprocess(R).unsqueeze(0)
    
    # Divide video into patches and calculate quality of each patch
    if D.shape[3]%patch_scale!=0 or D.shape[4]%patch_scale!=0:
        print(f'WARNING: Spatial resolution not divisible by {patch_scale}. Error map resolution might not match input videos')
    ps = [int(D.shape[3]/patch_scale),int(D.shape[4]/patch_scale)]

    clip_size = int(min(metadata['fps'],30)) # temporal duration of each patch
    # Pad videos in space-time to be divisible by patch size
    D = torch.nn.functional.pad(D, (0, (ps[1] - D.shape[4] % ps[1]) % ps[1], 0, (ps[0] - D.shape[3] % ps[0]) % ps[0], 0, (clip_size - D.shape[2] % clip_size)  % clip_size), mode='replicate')
    R = torch.nn.functional.pad(R, (0, (ps[1] - R.shape[4] % ps[1]) % ps[1], 0, (ps[0] - R.shape[3] % ps[0]) % ps[0], 0, (clip_size - R.shape[2] % clip_size)  % clip_size), mode='replicate')
    
    emap = torch.zeros([R.shape[2],R.shape[3],R.shape[4]])
    patch_errors = []
    count = 0
    for i in range(0,D.shape[2],clip_size):
        for h in range(0,D.shape[3],ps[0]):
            for w in range(0,D.shape[4],ps[1]):
                Cd = D[:,:,i:min(i+clip_size,D.shape[2]),h:h+min(ps[0],D.shape[3]),w:w+min(ps[1],D.shape[4])].to(device)
                Cr = R[:,:,i:min(i+clip_size,D.shape[2]),h:h+min(ps[0],D.shape[3]),w:w+min(ps[1],D.shape[4])].to(device)
                with torch.no_grad():
                    q,em = model.feature_diff(Cd,Cr)
                    emap[i:min(i+clip_size,D.shape[2]),h:h+min(ps[0],D.shape[3]),w:w+min(ps[1],D.shape[4])] = em.squeeze()
                    patch_errors.append(q)
                count += 1
    
    emap = emap[:metadata['shape'][0],:metadata['shape'][2],:metadata['shape'][3]]

    if patch_pool=='max':
        q = 100 - max(patch_errors)
    elif  patch_pool=='mean':
        q = 100 - torch.mean(torch.stack(patch_errors))
    else:
        raise Exception("ERROR: unkown patch pooling method")

    return q, emap


def demo_cgvqm():

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Configuration
    dist_vid_path = os.path.join(dir_path,'media/Dock_dist.mp4') # Path to distorted video
    ref_vid_path = os.path.join(dir_path,'media/Dock_ref.mp4') # Path to reference video
    errmap_path = os.path.join(dir_path,'media/Dock_dist_emap.mp4') # Path to save CGVQM predicted error map
    cgvqm_type = CGVQM_TYPE.CGVQM_2 # select between CGVQM_TYPE.CGVQM_2 or CGVQM_TYPE.CGVQM_5. (CGVQM_2 is faster and generates more granular error maps than CGVQM_5. However, it shows weaker correlation with human ratings)
    device = 'cuda'
    patch_scale = 4 # FxHxW resolution video will be divided into smaller patches of resolution min(fps,30) x H/patch_scale x W/patch_scale. Increase this value if low on available GPU memory
    patch_pool='mean' # How to pool quality values from across all patches (choose from {'max', 'mean'})

    # Run CGVQM and visualize results
    q, emap = run_cgvqm(dist_vid_path, ref_vid_path, cgvqm_type = cgvqm_type, device=device, patch_pool=patch_pool, patch_scale=patch_scale)
    qlabels = ['very annoying', 'annoying', 'slightly annoying', 'perceptible but not annoying', 'imperceptible']
    print(f'Quality of Dock_dist is {q.item():.2f}/100 ({qlabels[int(np.round(q.item()/25))]}). Quality map written to media/Dock_dist_emap.mp4')
    visualize_emap(emap, dist_vid_path, 100, errmap_path)


if __name__ == "__main__":
    demo_cgvqm()