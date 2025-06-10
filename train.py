import os
import pandas as pd
import gc
from pathlib import Path
import numpy as np
import glob
import torch
import pickle # nosec

from utils import resnet18
from utils.utils import preprocess, load_resize_vids
from cgvqm import CGVQM

# ==========================================
# Helper: Prepare training cache
# ==========================================
def prepare_training_cache(data, cache_dir, device='cuda'):
    os.makedirs(cache_dir, exist_ok=True)
    data.to_csv(os.path.join(cache_dir, 'index.csv'))

    model = resnet18.r3d_18(weights=resnet18.R3D_18_Weights.DEFAULT).to(device)
    model.__class__ = CGVQM
    model.init_weights()
    model.eval()

    def prepare_cache_for_video(dist_path, ref_path, cache_path, device):
        # Load, resize, and normalize videos
        D, R, metadata = load_resize_vids(dist_path, ref_path)
        D = preprocess(D).unsqueeze(0)
        R = preprocess(R).unsqueeze(0)

        os.makedirs(cache_path, exist_ok=True)
        for f in Path(cache_path).glob("*"):
            if f.is_file():
                f.unlink()

        # Divide into 1s clips of 512x512 resolution
        patch_size = 512
        clip_size = int(min(metadata['fps'], 30))

        D = D[:, :, :clip_size * (D.shape[2] // clip_size),
              :patch_size * (D.shape[3] // patch_size),
              :patch_size * (D.shape[4] // patch_size)]
        R = R[:, :, :clip_size * (R.shape[2] // clip_size),
              :patch_size * (R.shape[3] // patch_size),
              :patch_size * (R.shape[4] // patch_size)]

        count = 0
        for i in range(0, D.shape[2], clip_size):
            for h in range(0, D.shape[3], patch_size):
                for w in range(0, D.shape[4], patch_size):
                    Cd = D[:, :, i:i+clip_size, h:h+patch_size, w:w+patch_size].to(device)
                    Cr = R[:, :, i:i+clip_size, h:h+patch_size, w:w+patch_size].to(device)
                    with torch.no_grad():
                        q, _ = model.feature_diff(Cd, Cr, cache_path=os.path.join(cache_path, f'{count}.p3d'))
                    count += 1

        del Cd, Cr
        gc.collect()
        torch.cuda.empty_cache()

    def process_batch(start, end, device_id):
        for i in range(start, end):
            dist_path = data.dist_vid_path[i].replace('\\', '/')
            ref_path = data.ref_vid_path[i].replace('\\', '/')
            cache_path = os.path.join(cache_dir, str(i))
            print(f'INFO: processing video id {i}/{end-start}')
            prepare_cache_for_video(dist_path, ref_path, cache_path, device_id)

    process_batch(0, len(data), device)

# ==========================================
# Multi-dataset training
# ==========================================
def train_resnet_mdt(cache_dir_list, num_layers=6, DEVICE='cuda'):
    GT, data, training_data, vidNdxs = [], [], [], []

    for cache_dir in cache_dir_list:
        training_data_v = pd.read_csv(os.path.join(cache_dir, 'index.csv'))

        # You can replace this with a subset CSV if needed
        training_subset = training_data_v.copy()
        vidNdx = [str(training_data_v.loc[training_data_v['dist_vid_path'] == training_subset.dist_vid_path[i]].index[0])
                  for i in range(len(training_subset))]

        fdiffs, GT_v = [], []
        for vid_id in vidNdx:
            fdiffs_patches = []
            for filepath in glob.iglob(os.path.join(cache_dir, vid_id, '*.p3d')):
                with open(filepath, 'rb') as fp:
                    fdiffs_patches.append(pickle.load(fp)) # nosec
            fdiffs.append(fdiffs_patches)
            GT_v.append(training_data_v.dmos[int(vid_id)])

        data_v = []
        for v in range(len(fdiffs)):  # For every video
            patch_data = []
            for j in range(min(len(fdiffs[v][0]), num_layers)):  # For every layer
                layer_data = torch.cat([fdiffs[v][i][j] for i in range(len(fdiffs[v]))], dim=0).to(DEVICE)
                patch_data.append(layer_data)
            data_v.append(patch_data)

        GT.append(torch.abs(torch.tensor(GT_v)).to(DEVICE))
        data.append(data_v)
        training_data.append(training_data_v)
        vidNdxs.append(vidNdx)

    chns = [3, 64, 64, 128, 256, 512][:num_layers]
    w = (torch.rand(1, sum(chns), 1, 1, 1)).to(DEVICE)
    w = w / w.sum()

    def loss(w, a):
        w = w.abs()
        w_split = torch.split(w, chns, dim=1)
        total_loss = torch.tensor(0., device=DEVICE)

        for did in range(len(GT)):
            vqd = torch.zeros(GT[did].shape, device=DEVICE)
            for v in range(len(data[did])):  # For every video
                val = torch.zeros(data[did][v][0].shape[0], device=DEVICE)
                for i in range(min(len(data[did][v]), num_layers)):  # For every layer
                    val += (w_split[i] * data[did][v][i]).sum(dim=(1, 2, 3, 4))
                vqd[v] = (a * val).max()

            vx = vqd - vqd.mean()
            vy = GT[did] - GT[did].mean()
            plcc_loss = 1 - torch.abs(torch.sum(vx * vy) * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2)))
            total_loss += plcc_loss

        return total_loss

    # Optimization setup
    wo = torch.tensor(w, requires_grad=True)
    ao = torch.tensor(1000., device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([{'params': wo, 'lr': 1e-6}, {'params': ao, 'lr': 1.0}], lr=1e-6)

    steps = 100000
    best_loss = float('inf')

    torch.autograd.set_detect_anomaly(True)

    for i in range(steps):
        optimizer.zero_grad()
        current_loss = loss(wo, ao)

        if current_loss.item() < best_loss:
            best_loss = current_loss.item()
            with open(os.path.join(cache_dir_list[0], f'weights_mdt_plcc_nl{num_layers}.pickle'), 'wb') as fp:
                pickle.dump([wo.detach().cpu(), ao.detach().cpu()], fp) # nosec

        current_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'At step {i+1:5d}, training loss = {current_loss.item():.4f}, '
                  f'a = {ao.item():.4f}, w.min = {wo.min():.4f}, w.max = {wo.max():.4f}, w.mean = {wo.mean():.4f}, NL = {num_layers}')
            np.savetxt(os.path.join(cache_dir_list[0], 'wo.txt'), wo[0, :, 0, 0, 0].cpu().detach().numpy())

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    datasets = ['path/to/cgvqd']
    cache_dir_list = []

    for dataset in datasets:
        data = pd.read_csv(os.path.join(dataset, 'dmos.csv'))
        data['dist_vid_path'] = dataset + '/vids/' + data['dist_vid_path']
        data['ref_vid_path'] = dataset + '/vids/' + data['ref_vid_path']
        data['dmos'] = 100 - np.clip(data['dmos'], 0, 100)

        cache_dir = os.path.join(dataset, 'cgvqm_train_cache')
        prepare_training_cache(data, cache_dir, device='cuda')
        cache_dir_list.append(cache_dir)

    train_resnet_mdt(cache_dir_list, num_layers=6, DEVICE='cuda')
