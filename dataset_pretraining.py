import numpy as np
import random
from scipy.ndimage import gaussian_filter, binary_erosion, affine_transform
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates, binary_erosion, distance_transform_edt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



def apply_window(img_np, window_center, window_width):
    """
    窗宽窗位归一化，将 HU 值映射到 [0, 1]
    window_center: 窗位 (WC)
    window_width:  窗宽 (WW)
    """
    low  = window_center - window_width / 2
    high = window_center + window_width / 2
    img_clipped = np.clip(img_np, low, high)
    img_normed  = (img_clipped - low) / (high - low)
    return img_normed

def get_multichannel_cta_from_hu(cta_hu_array):
    """
    将 HU 值的 CTA (D, H, W) 转换为 4 通道归一化数据 (4, D, H, W)。
    """
    # 定义 4 个窗口: (WL, WW)
    windows = [
        (-100, 140),  # Ch0: Fat
        (50, 400),    # Ch1: Soft Plaque
        (350, 700),   # Ch2: Angiography
        (500, 2000)   # Ch3: Calcification
    ]
    
    channels = []
    # 转换为 float32 进行计算
    cta_hu_array = cta_hu_array.astype(np.float32)
    
    for wl, ww in windows:
        min_val = wl - ww / 2
        max_val = wl + ww / 2
        
        # Clip & Normalize
        roi = np.clip(cta_hu_array, min_val, max_val)
        roi = (roi - min_val) / (max_val - min_val)
        channels.append(roi)
    
    return np.stack(channels, axis=0) # (4, D, H, W)

class JointSSLTransform3D:
    """
    支持同时对 Image (4, D, H, W) 和 Mask (1, D, H, W) 进行几何变换。
    """
    def __init__(self, angle_range=(-15, 15), zoom_range=(0.9, 1.1), shift_range=(-5, 5)):
        self.angle_range = angle_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range

    def apply_affine(self, img, mask):
        # img: (C, D, H, W), mask: (1, D, H, W)
        spatial_shape = img.shape[1:] 
        center = (np.array(spatial_shape) - 1) / 2.0 

        # 1. 生成参数
        if random.random() > 0.5:
            shifts = np.array([random.uniform(*self.shift_range) for _ in range(3)])
            offset_shift = -shifts
        else:
            offset_shift = np.zeros(3)

        R_inv = np.eye(3)
        # 简化旋转逻辑
        if random.random() > 0.5: # Random rotation
            # 这里简单混合三个轴的旋转
            for axis in [0, 1, 2]:
                if random.random() > 0.5:
                    angle = -np.deg2rad(random.uniform(*self.angle_range))
                    c, s = np.cos(angle), np.sin(angle)
                    if axis == 0: R_local = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    elif axis == 1: R_local = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                    else: R_local = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                    R_inv = R_inv @ R_local

        if random.random() > 0.5:
            zoom_factor = random.uniform(*self.zoom_range)
            S_inv = np.diag([zoom_factor]*3)
        else:
            S_inv = np.eye(3)
            
        M = S_inv @ R_inv 
        s = center - M @ center + offset_shift
        
        # 2. 应用变换 (需要对 Image 的每个 Channel 循环应用)
        # Image
        img_transformed = np.zeros_like(img)
        for c in range(img.shape[0]):
            img_transformed[c] = affine_transform(
                img[c], M, offset=s, order=1, mode='constant', cval=0.0, output_shape=spatial_shape
            )
        
        # Mask (只变换第0通道)
        mask_transformed = np.zeros_like(mask)
        mask_transformed[0] = affine_transform(
            mask[0], M, offset=s, order=0, mode='constant', cval=0, output_shape=spatial_shape
        )
        
        return img_transformed, mask_transformed

    def random_flip(self, img, mask):
        # 遍历空间轴 1, 2, 3 (对应 D, H, W)
        for spatial_axis in [0, 1, 2]:
            if random.random() > 0.5:
                img = np.flip(img, axis=spatial_axis + 1).copy()
                mask = np.flip(mask, axis=spatial_axis + 1).copy() # Mask 也有 Channel dim
        return img, mask

    

    def __call__(self, img_tensor, mask_tensor):
        # 输入必须是 Numpy: (C, D, H, W)
        img_np = img_tensor.numpy()
        mask_np = mask_tensor.numpy()
        
        # img_aug, mask_aug = self.apply_affine(img_np, mask_np)
        img_aug, mask_aug = self.random_flip(img_np, mask_np)

        # if random.random() > 0:
        #     level = random.uniform(5.0, 20.0)
        #     img_aug = self.add_poisson_noise(img_aug, lam_scale=1.0)
        
        return torch.from_numpy(img_aug), torch.from_numpy(mask_aug)


# class LesionSynthesizerHU:
#     def __init__(self, 
#                  # 参数现在是真实的 HU 值
#                  calc_hu_range=(800, 1500),   # 钙化通常 > 400，甚至 > 1000
#                  soft_hu_range=(30, 90),      # 软斑块/脂质斑块通常 30-70 HU
#                  blob_sigma=(1.5, 4.0)):      
        
#         self.calc_hu_range = calc_hu_range
#         self.soft_hu_range = soft_hu_range
#         self.sigma_range = blob_sigma

#     def _generate_random_blob(self, shape, center_mask=None, erode_iterations=0):
#         """生成 Blob 形状 (0~1)"""
#         blob = np.zeros(shape, dtype=np.float32)
#         valid_mask = center_mask
        
#         if center_mask is not None and erode_iterations > 0:
#             eroded = binary_erosion(center_mask, iterations=erode_iterations)
#             if np.sum(eroded) > 0: valid_mask = eroded
        
#         if valid_mask is not None and np.sum(valid_mask) > 0:
#             coords = np.argwhere(valid_mask > 0)
#             center = coords[random.randint(0, len(coords)-1)]
#         else:
#             center = np.array(shape) // 2
        
#         blob[center[0], center[1], center[2]] = 1.0
        
#         # 随机椭球 Sigma
#         sigma_z = random.uniform(*self.sigma_range)
#         sigma_y = random.uniform(*self.sigma_range)
#         sigma_x = random.uniform(*self.sigma_range)
#         blob = gaussian_filter(blob, sigma=(sigma_z, sigma_y, sigma_x))
        
#         if blob.max() > 0:
#             blob = blob / blob.max()
#         return blob

#     def synthesize_calcified(self, img_hu, vessel_mask):
#         """
#         合成钙化：将 HU 值拉高到 800+
#         """
#         # 1. 生成 Blob (不腐蚀，钙化可在边缘)
#         blob = self._generate_random_blob(img_hu.shape, center_mask=vessel_mask, erode_iterations=0)
        
#         # 2. 目标 HU 值
#         target_hu = random.uniform(*self.calc_hu_range)
        
#         # 3. 混合 (简单的加权覆盖)
#         # 钙化密度极高，几乎完全覆盖原有组织
#         # 阈值化 blob 使得核心区域更实
#         core_blob = np.copy(blob)
#         core_blob[core_blob < 0.3] = 0 # 边缘锐利化
        
#         # 混合：img = img * (1-blob) + target * blob
#         # 这里的 blob 充当 alpha channel
#         blend_factor = core_blob 
#         img_aug = img_hu * (1.0 - blend_factor) + target_hu * blend_factor
        
#         lesion_mask = (core_blob > 0.3).astype(np.float32)
#         return img_aug, [1.0,0.0], lesion_mask

#     def synthesize_soft(self, img_hu, vessel_mask):
#         """
#         合成软斑块：将 HU 值拉低/拉高到 30-90 之间
#         """
#         # 1. 生成 Blob (腐蚀 Mask，软斑块通常在管腔内贴壁)
#         blob = self._generate_random_blob(img_hu.shape, center_mask=vessel_mask, erode_iterations=1)
        
#         # 2. 目标 HU 值
#         target_hu = random.uniform(*self.soft_hu_range)
        
#         # 3. 混合
#         # 软斑块取代了原本的高密度造影剂 (CTA血管通常 300-500 HU)
#         # 所以这会表现为一个“低密度充盈缺损”
        
#         # 限制在血管 Mask 内
#         effective_lesion = blob * (vessel_mask > 0)
#         if effective_lesion.max() > 0:
#             effective_lesion = effective_lesion / effective_lesion.max()
            
#         img_aug = img_hu * (1.0 - effective_lesion) + target_hu * effective_lesion
        
#         lesion_mask = (effective_lesion > 0.2).astype(np.float32)
#         return img_aug, [0.0,1.0], lesion_mask

#     def __call__(self, img_hu, vessel_mask):
#         """
#         Input: img_hu (D, H, W) float32 or int16
#         Output: img_aug_hu (D, H, W), label (float), lesion_mask (D, H, W)
#         """
#         # 确保是 float 用于计算
#         img_hu = img_hu.astype(np.float32)
        

#         if random.random() < 0.3:
#             return self.synthesize_calcified(img_hu, vessel_mask)
#         else:
#             return self.synthesize_soft(img_hu, vessel_mask)



class LesionSynthesizerLite:
    def __init__(self, 
                 calc_hu_range=(800, 1500), 
                 soft_hu_range=(30, 90), 
                 blob_sigma=(1.0, 3.5)): 
        
        self.calc_hu_range = calc_hu_range
        self.soft_hu_range = soft_hu_range
        self.sigma_range = blob_sigma

    def _get_random_center(self, mask, erode_iter=0):
        """快速获取Mask内的随机中心点"""
        if mask is None or np.sum(mask) == 0:
            return None
        
        valid_mask = mask
        if erode_iter > 0:
            eroded = binary_erosion(mask, iterations=erode_iter)
            if np.sum(eroded) > 0:
                valid_mask = eroded
        
        coords = np.argwhere(valid_mask > 0)
        return coords[random.randint(0, len(coords)-1)]

    def _generate_composite_blob(self, shape, center, num_blobs=2):

        canvas = np.zeros(shape, dtype=np.float32)
        
        # 主球体
        canvas[center[0], center[1], center[2]] = 1.0
        
        # 随机添加 1-2 个副球体（在主球体附近微小偏移）
        # 模拟不规则形状
        for _ in range(num_blobs - 1):
            offset_z = random.randint(-2, 2)
            offset_y = random.randint(-3, 3)
            offset_x = random.randint(-3, 3)
            
            nz = np.clip(center[0] + offset_z, 0, shape[0]-1)
            ny = np.clip(center[1] + offset_y, 0, shape[1]-1)
            nx = np.clip(center[2] + offset_x, 0, shape[2]-1)
            
            # 赋予副球体不同的强度，增加随机性
            canvas[nz, ny, nx] = random.uniform(0.5, 1.0)

        # 统一做一次高斯模糊，把这些散点融合成一个不规则的块
        # 随机 Sigma 制造椭球感
        sigma_z = random.uniform(*self.sigma_range)
        sigma_y = random.uniform(*self.sigma_range)
        sigma_x = random.uniform(*self.sigma_range)
        
        blob = gaussian_filter(canvas, sigma=(sigma_z, sigma_y, sigma_x))
        
        # 归一化
        if blob.max() > 0:
            blob = blob / blob.max()
            
        return blob

    def synthesize_calcified(self, img_hu, vessel_mask):
        # 1. 找中心（钙化不需要腐蚀，可以在边缘）
        center = self._get_random_center(vessel_mask, erode_iter=0)
        if center is None: return img_hu, 0, np.zeros_like(img_hu)

        # 2. 生成组合 Blob (模拟钙化的棱角，用较少的球叠加，比如 2 个)
        blob = self._generate_composite_blob(img_hu.shape, center, num_blobs=random.randint(1, 3))
        
        # 3. 钙化特性：高亮、锐利
        target_hu = random.uniform(*self.calc_hu_range)
        
        # 简单的阈值处理让核心变硬
        core_blob = np.copy(blob)
        # core_blob[core_blob < 0.3] = 0 
        core_blob = 1 / (1 + np.exp(-10 * (core_blob - 0.3)))
        core_blob = (core_blob - core_blob.min()) / (core_blob.max() - core_blob.min())
        
        # 混合
        img_aug = img_hu * (1.0 - core_blob) + target_hu * core_blob
        lesion_mask = (core_blob > 0.2).astype(np.float32)
        
        return img_aug, 1, lesion_mask

    def synthesize_soft(self, img_hu, vessel_mask):
        # 1. 找中心（软斑块必须在血管内，腐蚀 1-2 次确保不飞出血管）
        center = self._get_random_center(vessel_mask, erode_iter=1)
        if center is None: return img_hu, 0, np.zeros_like(img_hu)

        # 2. 生成组合 Blob (软斑块通常比较长，可以多叠加几个球)
        blob = self._generate_composite_blob(img_hu.shape, center, num_blobs=random.randint(1, 3))
        
        # 3. 软斑块特性：低密度、受血管壁限制
        target_hu = random.uniform(*self.soft_hu_range)
        
        # 限制在血管掩膜内 (Clip)
        # effective_lesion = blob * (vessel_mask > 0)
        soft_mask = gaussian_filter(vessel_mask.astype(np.float32), sigma=1.0)
        effective_lesion = blob * soft_mask
        if effective_lesion.max() > 0:
            effective_lesion /= effective_lesion.max()

        img_aug = img_hu * (1.0 - effective_lesion) + target_hu * effective_lesion
        lesion_mask = (effective_lesion > 0.2).astype(np.float32)
        
        return img_aug, 2, lesion_mask # 假设 2 是软斑块 label

    def __call__(self, img_hu, vessel_mask):
        # 强制转为 float32 节省内存 (避免 float64)
        img_hu = img_hu.astype(np.float32)
        
        if random.random() < 0.5:
            return self.synthesize_calcified(img_hu, vessel_mask)
        else:
            return self.synthesize_soft(img_hu, vessel_mask)




def save_validation_snapshot(images, targets, save_dir, batch_idx):
    """
    从当前 Batch 中取样，保存 Input vs GT vs Prediction 的对比图。
    """
    print(f"[Debug] Saving validation snapshot for batch {batch_idx} to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 转换数据 (B, 1, D, H, W) -> numpy
    # detach(): 切断梯度, cpu(): 移至内存
    # imgs_np = images.squeeze().numpy() # (B, C, D, H, W)
    # tgts_np = targets.squeeze().numpy() # (B,targets 1, D, H, W)
    

    
    # 取出单个 3D volume (去掉 channel 维度)
    # shape: (D, H, W)
    img = images.squeeze().numpy()
    tgt = targets.squeeze().numpy()

    # 3. 确定切片位置
    # 如果有病灶，切在病灶中心；如果没有，切在图像几何中心

    img = apply_window(img, window_center=500, window_width=2000)

    d, h, w = img.shape

    coords = np.argwhere(tgt > 0)
    center = coords.mean(axis=0).astype(int)
    cz, cy, cx = center[0], center[1], center[2]


    # 防止索引越界
    cz, cy, cx = np.clip([cz, cy, cx], 0, [d-1, h-1, w-1])

    # 4. 绘图 (3行3列)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Batch {batch_idx} | \n(Slice at {cz}, {cy}, {cx})", fontsize=16)
    
    # 行定义
    rows_data = [
        ("Input Image", img, None),
        ("Ground Truth", tgt, "Greens"),  # 绿色显示 GT
    ]

    for row_idx, (title, vol, cmap_overlay) in enumerate(rows_data):
        # 设置显示范围：Mask 是 0-1，Image 假设归一化过也是 0-1
        vmin, vmax = 0, 1
        
        # 如果是 Mask，用特定颜色；如果是原图，用灰度
        cmap = 'gray' if cmap_overlay is None else cmap_overlay

        # --- Axial (Z) ---
        axes[row_idx, 0].imshow(vol[cz, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(title, fontsize=14, fontweight='bold')
        if row_idx == 0: axes[row_idx, 0].set_title("Axial")

        # --- Sagittal (Y) ---
        axes[row_idx, 1].imshow(np.rot90(vol[:, cy, :]), cmap=cmap, vmin=vmin, vmax=vmax)
        if row_idx == 0: axes[row_idx, 1].set_title("Sagittal")

        # --- Coronal (X) ---
        axes[row_idx, 2].imshow(np.rot90(vol[:, :, cx]), cmap=cmap, vmin=vmin, vmax=vmax)
        if row_idx == 0: axes[row_idx, 2].set_title("Coronal")

    # 保存
    save_path = os.path.join(save_dir, f"batch_{batch_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class CT3D_Anomaly_Dataset(Dataset):
    def __init__(self, 
                 excel_file, 
                 npz_root, 
                 patch_shape=(64, 64, 64), 
                 min_mask_voxels=50):
        
        self.df = pd.read_excel(excel_file)
        self.npz_root = npz_root
        self.patch_shape = np.array(patch_shape)
        self.min_mask_voxels = min_mask_voxels
        
        # 初始化 HU 空间合成器
        self.lesion_synthesizer = LesionSynthesizerLite(
            calc_hu_range=(800, 1500),  # 强钙化 HU
            soft_hu_range=(30, 90),     # 软斑块 HU
            blob_sigma=(1.0, 2.0)
        )
        
        # 几何增强 (作用于 Tensor)
        self.augmentor = JointSSLTransform3D(
            angle_range=(-15, 15),
            zoom_range=(0.9, 1.1)
        )

    def __len__(self):
        return len(self.df)

    def load_npz_data(self, name):
        # 注意：这里我们主要需要 'CTA_HU' 和 'CA'
        npz_filename = f"CTA_{name}.npz"
        npz_path = os.path.join(self.npz_root, name, npz_filename)
        # print(f"[Debug] Loading NPZ path: {npz_path}")  # 调试打印路径
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")
        try:
            data = np.load(npz_path)
            # 返回: Raw HU, Mask
            # 假设之前保存时 key 是 'CTA_HU' 和 'CA'
            return data['CTA_HU'], data['CA'] 
        except Exception as e:
            raise RuntimeError(f"Error loading NPZ {name}: {e}")

    def extract_patch_centered(self, volume, center, shape):
        # 简单的 Padding + Crop (为了完整性，这里写一个简单版)
        d, h, w = volume.shape
        cd, ch, cw = shape
        
        start = center - shape // 2
        end = start + shape
        
        pad_before = np.maximum(-start, 0)
        pad_after = np.maximum(end - np.array(volume.shape), 0)
        padding = tuple((b, a) for b, a in zip(pad_before, pad_after))
        
        # 使用 volume 最小值 padding
        val = volume.min()
        volume_pad = np.pad(volume, padding, mode='constant', constant_values=val)
        
        start += pad_before
        end += pad_before
        
        return volume_pad[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    def add_poisson_noise(self,ct_image, I0=1e5, L=200.0, sigma_e=2.0):
        """
        更真实的CCTA泊松噪声模拟（图像域近似）

        I0: 入射光子数
        L: 等效路径长度 (mm) —— 用200-350更真实
        sigma_e: 电子噪声（HU域高斯）
        """
        mu_water = 0.02

        # HU → μ
        mu = (ct_image + 1000.0) / 1000.0 * mu_water
        mu = np.clip(mu, 0, None)

        # 更真实 line integral
        line_integral = mu * L

        # Beer–Lambert
        I_clean = I0 * np.exp(-line_integral)

        # Poisson
        I_noisy = np.random.poisson(I_clean).astype(np.float32)
        I_noisy = np.clip(I_noisy, 1, None)

        # back to HU
        mu_noisy = -np.log(I_noisy / I0) / L
        ct_noisy = (mu_noisy / mu_water * 1000.0) - 1000.0

        # 加电子噪声
        ct_noisy += np.random.normal(0, sigma_e, ct_noisy.shape)

        # 轻微平滑，sigma非常小
        ct_noisy = gaussian_filter(ct_noisy, sigma=0.3)

        return ct_noisy.astype((ct_image.dtype))

    def __getitem__(self, idx):
        # 1. 获取文件名
        try:
            row = self.df.iloc[idx]
            name = str(row['Deidentification Patient Name'])
            
            # --- 调试打印：检查路径是否正确 ---
            # 假设你的 load_npz_data 内部会拼接路径，这里先手动模拟一下，或者在 load_npz_data 里打印
            # print(f"[Debug] Loading index {idx}, Name: {name}") 

            # 这里的 idx 在重试时需要随机更新
            current_idx = idx #if attempt == 0 else random.randint(0, len(self.df) - 1)
            name = str(self.df.iloc[current_idx]['Deidentification Patient Name'])
            
            # --- 优化建议：如果是 npz，尝试使用 mmap_mode ---
            # image_hu, mask = self.load_npz_data(name) 
            # 建议修改 load_npz_data 内部实现，使用 mmap_mode='r'
            
            image_hu, mask = self.load_npz_data(name)

            # level = random.uniform(5.0, 20.0)
            # 2. 确定采样中心
            mask_coords = np.argwhere(mask > 0)
            if len(mask_coords) < self.min_mask_voxels:
                center = np.array(image_hu.shape) // 2
                patch_mask = np.zeros(self.patch_shape)
            else:
                center = mask_coords[random.randint(0, len(mask_coords) - 1)]
            
            # 3. 提取 Patch
            patch_hu = self.extract_patch_centered(image_hu, center, self.patch_shape)
            patch_mask = self.extract_patch_centered(mask, center, self.patch_shape)

            # !!! 关键优化：提取完 patch 后，立刻删除大的全量数据 !!!
            del image_hu
            del mask
            del mask_coords
            
            # 4. 合成病灶
            img_aug_hu, label, lesion_mask = self.lesion_synthesizer(patch_hu, patch_mask)
            img_aug_hu = self.add_poisson_noise(img_aug_hu, I0=5e4, L=200.0, sigma_e=5.0)
            
            
            # 5. 通道转换
            img_multichannel = get_multichannel_cta_from_hu(img_aug_hu)

            # 6. 转 Tensor
            img_tensor = torch.from_numpy(img_multichannel).float()
            lesion_mask_tensor = torch.from_numpy(lesion_mask).unsqueeze(0).float()

            # 7. 增强
            patch_mask_tensor = torch.from_numpy(patch_mask).unsqueeze(0).float()

            img_tensor_aug, lesion_mask_tensor_aug = self.augmentor(img_tensor, lesion_mask_tensor)

            # img_tensor_aug, lesion_mask_tensor_aug = self.augmentor(img_tensor, patch_mask_tensor)

            return img_tensor_aug, torch.tensor([label], dtype=torch.float32), lesion_mask_tensor_aug
            # return img_aug_hu, lesion_mask

        except Exception as e:

            # 如果有其他坏数据导致的错误（比如文件损坏），也可以这样处理

            print(f"[Error] Error loading index {idx}: {e}")

            new_idx = random.randint(0, len(self.df) - 1)

            return self.__getitem__(new_idx)    

if __name__ == "__main__":

    EXCEL_FILE = "/home/zvv0112/MACE_CCTA/data/folds_30MACE2/fold1/test.xlsx"
    NPZ_ROOT = "/gpfs/projects/p32902/datasets/CCTA_MACE_npz4_with_raw"
    
    from torch.utils.data import DataLoader

    PATCH_SHAPE = (128, 128, 128)

    dataset = CT3D_Anomaly_Dataset(
        excel_file=EXCEL_FILE,
        npz_root=NPZ_ROOT,
        patch_shape=PATCH_SHAPE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4
    )

    # 测试前5个batch
    for i, (img, mask) in enumerate(dataloader):
        print(f"Batch {i} - Image shape: {img.shape}, Mask shape: {mask.shape}")
        save_validation_snapshot(img, mask, save_dir="./debug_vis_noncal", batch_idx=i)
        if i >= 100:
            break