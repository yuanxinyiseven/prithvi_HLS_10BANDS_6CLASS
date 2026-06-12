import os
import logging

# 强制 GDAL 环境保持安静，并统一使用 UTF-8
os.environ["GDAL_QUIET"] = "ON"
os.environ["GDAL_FILENAME_IS_UTF8"] = "YES"

# 强行将 rasterio 的日志器级别设为 ERROR，停止输出 DEBUG 信息
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("fiona").setLevel(logging.ERROR)


import glob
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray
import re
from osgeo import gdal


from terratorch.datasets.utils import default_transform, filter_valid_files, to_rgb, validate_bands


class TemporalTransforms:
    """
    支持 4D 时序数据 [C, T, H, W] 的 Transform 类
    """
    
    def __init__(self, do_flip=True, do_rotate=False, normalize=True):
        """
        Args:
            do_flip: 是否随机翻转
            do_rotate: 是否随机旋转（0/90/180/270）
            normalize: 是否归一化到 [0, 1]
        """
        self.do_flip = do_flip
        self.do_rotate = do_rotate
        self.normalize = normalize
    
    def __call__(self, data: dict) -> dict:
        """
        Args:
            data: {'image': [C, T, H, W], 'mask': [H, W]}
        
        Returns:
            data: 同上，但已应用 transform
        """
        image = data['image']  # [C, T, H, W]
        mask = data['mask']    # [H, W]
        
        # 水平翻转
        if self.do_flip and np.random.rand() > 0.5:
            image = torch.flip(image, dims=(-1,))  # 沿 W 维度翻转
            mask = torch.flip(mask, dims=(-1,))
        
        # 竖直翻转
        if self.do_flip and np.random.rand() > 0.5:
            image = torch.flip(image, dims=(-2,))  # 沿 H 维度翻转
            mask = torch.flip(mask, dims=(-2,))
        
        # 随机旋转 (0/90/180/270)
        if self.do_rotate and np.random.rand() > 0.5:
            k = np.random.randint(1, 4)  # 1=90°, 2=180°, 3=270°
            image = torch.rot90(image, k=k, dims=(-2, -1))
            mask = torch.rot90(mask, k=k, dims=(-2, -1))
        
        # 归一化到 [0, 1]
        if self.normalize:
            # 假设输入范围是 [0, 10000] 或类似
            # 可根据实际数据范围调整
            image = image / image.max() if image.max() > 0 else image
        
        data['image'] = image
        data['mask'] = mask
        return data


class MultiTemporalCropClassification(NonGeoDataset):
    """NonGeo dataset implementation for [multi-temporal crop classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)."""
    """
    针对 HLS 10波段、12时相数据定制的 Dataset
    
    Prithvi 模型期望的输入格式: [B, C, T, H, W]
    
    数据流程说明:
    1. Dataset 返回单个样本:
       - image: [6, 12, 224, 224]  <- [C, T, H, W]
       - mask: [224, 224]           <- [H, W]
    
    2. DataLoader Batch 处理:
       - images: [B, 6, 12, 224, 224]  <- [B, C, T, H, W]
       - masks: [B, 224, 224]          <- [B, H, W]
    
    这个格式可以直接输入到 Prithvi 模型!
    
    Transform 说明:
    - 使用自定义 TemporalTransforms，支持 4D 时序数据
    - 包含：水平/竖直翻转、旋转、归一化
    - 替代了不支持 4D 的 Albumentations
    """
    # 显式指出前 6 个波段名（对应 Prithvi 所需的 B, G, R, NIR, SWIR1, SWIR2）
    all_band_names = (
        "Band_1",
        "Band_2",
        "Band_3",
        "Band_4",
        "Band_5",
        "Band_6",
    )

    class_names = (
        "forest",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Nonforest",
    )

    num_classes = 6
    time_steps = 12
    num_bands = 6

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = all_band_names,
        transform: bool = True,  # 简化为 bool 类型
        no_data_replace: float | None = 0.0,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = False,
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            split (str): one of 'train' or 'val'.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (bool): Whether to apply augmentation. Defaults to True.
                Flip and rotate will be applied if True.
            no_data_replace (float | None): Replace nan values in input images with this value.
            no_label_replace (int | None): Replace nan values in label with this value.
            expand_temporal_dimension (bool): Keep temporal dimension expanded (not used currently).
            reduce_zero_label (bool): Subtract 1 from all labels.
        """
        super().__init__()
        
        base_root = Path(data_root)
        if split == "train":
            self.data_root = base_root / "train"
        elif split == "val":
            self.data_root = base_root / "val"
        elif split == "test":
            self.data_root = base_root / "test"
        else:
            raise ValueError(f"未知的 split 类型: {split}. 请选择 'train', 'val' 或 'test'.")
        self.split = split

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension

        # 创建 Transform
        # 只在 train split 时应用数据增强
        self.transform = None
        if transform and split == "train":
            self.transform = TemporalTransforms(
                do_flip=True,
                do_rotate=True,
                normalize=False  # 不做归一化，保持原始数据范围
            )

        # ─── 解析文件路径与对齐时序影像 ───
        self.samples_list = []
        label_root = self.data_root / "label"
        image_root = self.data_root / "Image"

        # 遍历 HLJ2016 - HLJ2023 文件夹
        if label_root.exists():
            print(f"-> 正在扫描 label 根目录: {label_root}")
            for year_dir in sorted(label_root.glob("HLJ*")):
                if not year_dir.is_dir():
                    continue
                year_match = re.search(r"HLJ(\d{4})", year_dir.name)
                if not year_match:
                    continue
                year = year_match.group(1)

                print(f"   发现年份文件夹: {year_dir.name} -> 提取年份: {year}")

                label_files = sorted(year_dir.glob("label_*_*_224.tif"))
                if not label_files:
                    print(f"   ⚠️ 警告: 在 {year_dir.name} 下没有找到形如 label_*_*_224.tif 的文件！")

                for label_file in sorted(year_dir.glob("label_*_*_224.tif")):
                    filename = label_file.name
                    parts = filename.split("_")
                    sample_id = parts[2]
                    img_size = parts[3].split(".")[0]

                    img_dir_name = f"HLJ_HLS_PATCHES_{year}_clip_class_1_HighQuality"
                    target_img_dir = image_root / img_dir_name

                    if not target_img_dir.exists():
                        print(f"   ❌ 找不到影像文件夹: {img_dir_name} (完整路径: {target_img_dir})")
                        continue

                    center_year = int(year)
                    years_to_find = [center_year - 1, center_year, center_year + 1]
                    quarters = ["Q1", "Q2", "Q3", "Q4"]
                    
                    temporal_image_paths = []
                    is_valid_sample = True

                    for y in years_to_find:
                        for q in quarters:
                            img_name = f"HLJ_{y}_{sample_id}_{q}_{img_size}.tif"
                            img_path = target_img_dir / img_name
                            if img_path.exists():
                                temporal_image_paths.append(img_path)
                            else:
                                print(f"   ❌ 文件夹 {img_dir_name} 内缺少单张影像: {img_name}")
                                is_valid_sample = False
                                break
                        if not is_valid_sample:
                            break

                    if is_valid_sample and len(temporal_image_paths) == 12:
                        self.samples_list.append({
                            "label_path": label_file,
                            "image_paths": temporal_image_paths
                        })
        else:
            print(f"❌ 错误: label 根目录根本不存在: {label_root}")
        
        print(f"统计: [{split.upper()}] 阶段成功对齐并加载了 {len(self.samples_list)} 个 12时相样本。\n")
        if len(self.samples_list) == 0:
            print(f"警告: 在 {self.data_root} 未找到任何匹配的 12时相对齐样本！请检查路径规则。")

    def __len__(self) -> int:
        return len(self.samples_list)

    @staticmethod
    def _read_tif_with_gdal(file_path: str) -> np.ndarray:
        """使用 GDAL 直接读取 TIF 文件，避免 rasterio 的路径编码问题"""
        gdal.UseExceptions()
        gdal.PushErrorHandler('CPLQuietErrorHandler')
        
        try:
            ds = gdal.Open(file_path)
            if ds is None:
                raise RuntimeError(f"Cannot open file with GDAL: {file_path}")
            
            band_count = ds.RasterCount
            
            # 读取所有波段
            data_list = []
            for band_idx in range(1, band_count + 1):
                band = ds.GetRasterBand(band_idx)
                band_data = band.ReadAsArray()
                if band_data is None:
                    raise RuntimeError(f"Failed to read band {band_idx} from {file_path}")
                data_list.append(band_data)
            
            # 堆叠波段
            result = np.stack(data_list, axis=0).astype(np.float32)
            ds = None
            
            return result
        finally:
            gdal.PopErrorHandler()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_info = self.samples_list[index]
        
        label_path = str(sample_info["label_path"])
        
        try:
            # 读取标签
            mask_data = self._read_tif_with_gdal(label_path)
            mask = mask_data[0].astype(np.int64)
        except Exception as e:
            print(f"Error reading label {label_path}: {e}")
            raise

        # 读取 12 幅影像
        time_series_list = []
        for img_path in sample_info["image_paths"]:
            try:
                img_data = self._read_tif_with_gdal(str(img_path))
                # 只切取前 6 个波段
                img_data = img_data[:6, :, :]
                time_series_list.append(img_data)
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                raise

        # 堆叠 12 个时相
        image = np.stack(time_series_list, axis=0)  # [12, 6, H, W] = [T, C, H, W]
        
        # 转换为 Prithvi 期望的格式 [C, T, H, W]
        image = np.transpose(image, (1, 0, 2, 3)).astype(np.float32)

        # 转换为 Tensor
        image = torch.from_numpy(image).float()  # [6, 12, 224, 224]
        mask = torch.from_numpy(mask).long()     # [224, 224]

        output = {
            "image": image,
            "mask": mask
        }

        # 应用 Transform（支持 4D 时序数据）
        if self.transform:
            output = self.transform(output)

        return output

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        # 处理不同的输入格式
        images = sample["image"]
        mask = sample["mask"]
        
        # 如果是 Tensor，转换为 numpy
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # 期望格式：[C, T, H, W] = [6, 12, 224, 224]
        # 取第一个时间步的前 3 个波段作为 RGB
        if images.ndim == 4 and images.shape[0] == self.num_bands and images.shape[1] == self.time_steps:
            # [C, T, H, W] 格式
            img_rgb = images[:3, 0, :, :]  # [3, H, W] - 前 3 个波段，第一个时间步
            img_rgb = np.transpose(img_rgb, (1, 2, 0))  # [H, W, 3]
        else:
            raise ValueError(f"Expected image shape [C, T, H, W] = [6, 12, H, W], but got {images.shape}")
        
        # 归一化到 [0, 255]
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
        
        # 绘制
        num_images = 3
        if "prediction" in sample:
            num_images += 1
        
        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")
        
        # RGB 图
        ax[0].axis("off")
        ax[0].title.set_text("RGB Image (T=0)")
        ax[0].imshow(img_rgb)
        
        # Ground Truth Mask
        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        ax[1].axis("off")
        ax[1].title.set_text("Ground Truth Mask")
        ax[1].imshow(mask, cmap="jet", norm=norm)
        
        # 图例
        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), self.class_names[i]] for i in range(self.num_classes)]
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[2].axis("off")
        ax[2].legend(handles, labels, loc="center", fontsize=10)
        
        # 预测结果（如果有）
        if "prediction" in sample:
            prediction = sample["prediction"]
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.cpu().numpy()
            ax[3].axis("off")
            ax[3].title.set_text("Predicted Mask")
            ax[3].imshow(prediction, cmap="jet", norm=norm)
        
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
