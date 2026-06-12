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

import albumentations as A
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import torch
from einops import rearrange
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from torch import Tensor
from torchgeo.datasets import NonGeoDataset
from xarray import DataArray
import re


from terratorch.datasets.utils import default_transform, filter_valid_files, to_rgb, validate_bands


class MultiTemporalCropClassification(NonGeoDataset):
    """NonGeo dataset implementation for [multi-temporal crop classification](https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification)."""
    """
    针对 HLS 10波段、12时相数据定制的 Dataset
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

    # rgb_bands = ("RED", "GREEN", "BLUE")

    # BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}

    num_classes = 6
    time_steps = 12
    # splits = {"train": "training", "val": "validation"}  # Only train and val splits available
    # col_name = "chip_id"
    # date_columns = ["first_img_date", "middle_img_date", "last_img_date"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: Sequence[str] = all_band_names,
        transform: A.Compose | None = None,
        no_data_replace: float | None = 0.0,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = False, # 【改】默认为 False，防止类别0被减一变成-1
        # use_metadata: bool = False,
        # metadata_file_name: str = "chips_df.csv",
    ) -> None:
        """Constructor

        Args:
            data_root (str): Path to the data root directory.
            split (str): one of 'train' or 'val'.
            bands (list[str]): Bands that should be output by the dataset. Defaults to all bands.
            transform (A.Compose | None): Albumentations transform to be applied.
                Should end with ToTensorV2(). If used through the corresponding data module,
                should not include normalization. Defaults to None, which applies ToTensorV2().
            no_data_replace (float | None): Replace nan values in input images with this value.
                If None, does no replacement. Defaults to None.
            no_label_replace (int | None): Replace nan values in label with this value.
                If none, does no replacement. Defaults to None.
            expand_temporal_dimension (bool): Go from shape (time*channels, h, w) to (channels, time, h, w).
                Defaults to True.
            reduce_zero_label (bool): Subtract 1 from all labels. Useful when labels start from 1 instead of the
                expected 0. Defaults to True.
            use_metadata (bool): whether to return metadata info (time and location).
        """
        super().__init__()
        # if split not in self.splits:
        #     msg = f"Incorrect split '{split}', please choose one of {self.splits}."
        #     raise ValueError(msg)
        # split_name = self.splits[split]
        base_root = Path(data_root) # 传入的应该是类似 E:\HLJ_data_10bands_6class_64\train
        if split == "train":
            self.data_root = base_root / "train"
        elif split == "val":
            self.data_root = base_root / "val"
        elif split == "test":
            self.data_root = base_root / "test"
        else:
            raise ValueError(f"未知的 split 类型: {split}. 请选择 'train', 'val' 或 'test'.")
        self.split = split
        

        # validate_bands(bands, self.all_band_names)
        # self.bands = bands
        # self.band_indices = np.asarray([self.all_band_names.index(b) for b in bands])
        # self.data_root = Path(data_root)

        # data_dir = self.data_root / f"{split_name}_chips"
        # self.image_files = sorted(glob.glob(os.path.join(data_dir, "*_merged.tif")))
        # self.segmentation_mask_files = sorted(glob.glob(os.path.join(data_dir, "*.mask.tif")))
        # split_file = self.data_root / f"{split_name}_data.txt"

        # with open(split_file) as f:
        #     split = f.readlines()
        # valid_files = {rf"{substring.strip()}" for substring in split}
        # valid_files = {
        #     os.path.basename(f).split(".")[0] for f in self.image_files
        # }
        # self.image_files = filter_valid_files(
        #     self.image_files,
        #     valid_files=valid_files,
        #     ignore_extensions=True,
        #     allow_substring=True,
        # )
        # self.segmentation_mask_files = filter_valid_files(
        #     self.segmentation_mask_files,
        #     valid_files=valid_files,
        #     ignore_extensions=True,
        #     allow_substring=True,
        # )
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.reduce_zero_label = reduce_zero_label
        self.expand_temporal_dimension = expand_temporal_dimension
        # self.use_metadata = use_metadata
        # self.metadata = None
        # self.metadata_file_name = metadata_file_name
        # if self.use_metadata:
        #     metadata_file = self.data_root / self.metadata_file_name
        #     self.metadata = pd.read_csv(metadata_file)
        #     self._build_image_metadata_mapping()

        # If no transform is given, apply only to transform to torch tensor
        self.transform = transform if transform else default_transform

        # ─── 【增】解析文件路径与对齐时序影像 ───
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
                year = year_match.group(1) # 提取年份字符串，如 "2016"

                # 【调试打印 2】看看找到了哪些年份文件夹
                print(f"   发现年份文件夹: {year_dir.name} -> 提取年份: {year}")

                # 寻找该年份下的所有标签文件
                label_files = sorted(year_dir.glob("label_*_*_224.tif"))
                if not label_files:
                    print(f"   ⚠️ 警告: 在 {year_dir.name} 下没有找到形如 label_*_*_224.tif 的文件！")

                # 寻找该年份下的所有标签文件
                for label_file in sorted(year_dir.glob("label_*_*_224.tif")):
                    # 解析文件名，例如 label_2016_1_224.tif
                    filename = label_file.name
                    parts = filename.split("_")
                    sample_id = parts[2] # 提取样本 ID，如 "1"
                    img_size = parts[3].split(".")[0] # "224"

                    # 构建对应的影像文件夹路径
                    # 规则：HLJ_HLS_PATCHES_2016_clip class_1_HighQuality
                    img_dir_name = f"HLJ_HLS_PATCHES_{year}_clip_class_1_HighQuality"
                    target_img_dir = image_root / img_dir_name

                    if not target_img_dir.exists():
                        # 【调试打印 3】如果是影像文件夹路径没对上，打印出来
                        print(f"   ❌ 找不到影像文件夹: {img_dir_name} (完整路径: {target_img_dir})")
                        continue

                    # 寻找 3 年（前一年、当年、后一年），每年的 Q1-Q4，总共 12 幅影像
                    # 例如当年是 2016，则找 2015, 2016, 2017
                    center_year = int(year)
                    years_to_find = [center_year - 1, center_year, center_year + 1]
                    quarters = ["Q1", "Q2", "Q3", "Q4"]
                    
                    temporal_image_paths = []
                    is_valid_sample = True

                    for y in years_to_find:
                        for q in quarters:
                            # 构建标准文件名：HLJ_2015_1_Q1_224.tif
                            img_name = f"HLJ_{y}_{sample_id}_{q}_{img_size}.tif"
                            img_path = target_img_dir / img_name
                            if img_path.exists():
                                temporal_image_paths.append(img_path)
                            else:
                                # 【调试打印 4】如果是里面的某一个季度 TIF 缺失了，打印出来
                                print(f"   ❌ 文件夹 {img_dir_name} 内缺少单张影像: {img_name}")
                                is_valid_sample = False
                                break
                        if not is_valid_sample:
                            break

                    # 只有当 12 幅影像全部集齐时，才加入训练样本集
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

    # def _build_image_metadata_mapping(self):
    #     """Build a mapping from image filenames to metadata indices."""
    #     self.image_to_metadata_index = dict()

    #     for idx, image_file in enumerate(self.image_files):
    #         image_filename = Path(image_file).name
    #         image_id = image_filename.replace("_merged.tif", "").replace(".tif", "")
    #         metadata_indices = self.metadata.index[self.metadata[self.col_name] == image_id].tolist()
    #         self.image_to_metadata_index[idx] = metadata_indices[0]
    
    def __len__(self) -> int:
        return len(self.samples_list)

    # def _get_date(self, row: pd.Series) -> torch.Tensor:
    #     """Extract and format temporal coordinates (T, date) from metadata."""
    #     temporal_coords = []
    #     for col in self.date_columns:
    #         date_str = row[col]
    #         date = pd.to_datetime(date_str)
    #         temporal_coords.append([date.year, date.dayofyear - 1])

        return torch.tensor(temporal_coords, dtype=torch.float32)

    # def _get_coords(self, image: DataArray) -> torch.Tensor:
    #     px = image.x.shape[0] // 2
    #     py = image.y.shape[0] // 2

    #     # get center point to reproject to lat/lon
    #     point = image.isel(band=0, x=slice(px, px + 1), y=slice(py, py + 1))
    #     point = point.rio.reproject("epsg:4326")

    #     lat_lon = np.asarray([point.y[0], point.x[0]])

    #     return torch.tensor(lat_lon, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_info = self.index[index]
    
        # === 关键修复：禁用所有日志以避免编码问题 ===
        import logging
        import os
        from pathlib import Path
    
        # 完全禁用 rasterio 和 GDAL 日志
        logging.getLogger("rasterio").setLevel(logging.CRITICAL)
        logging.getLogger("fiona").setLevel(logging.CRITICAL)
    
        # 禁用 GDAL 警告和错误输出到日志
        os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'YES'
        os.environ['CPL_DEBUG'] = 'OFF'
        os.environ['GDAL_CACHEMAX'] = '0'
    
        label_path = str(sample_info["label_path"])
    
        try:
            # 使用最小化的 rasterio 环境
            import rasterio
            import rioxarray
        
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN='YES',
                GDAL_CACHEMAX=0,
                GDAL_NUM_THREADS='1',
                CPL_DEBUG=False
            ):
                # 禁用该次调用的日志
                old_logger = logging.getLogger('rasterio._env')
                old_level = old_logger.level
                old_logger.setLevel(logging.CRITICAL)
                old_logger.disabled = True
            
                try:
                    mask_data = rioxarray.open_rasterio(label_path, masked=True)
                finally:
                    old_logger.disabled = False
                    old_logger.setLevel(old_level)
                
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            # 降级处理：使用 GDAL 的 Python 绑定直接读取
            from osgeo import gdal
            gdal.UseExceptions()
            ds = gdal.Open(label_path)
            if ds is None:
                raise RuntimeError(f"Cannot open file: {label_path}")
            band = ds.GetRasterBand(1)
            mask_data = band.ReadAsArray()
            ds = None
            raise
    
        if self.no_label_replace is not None:
            mask_data = mask_data.fillna(self.no_label_replace)
        # image = self._load_file(self.image_files[index], nan_replace=self.no_data_replace)

        # location_coords, temporal_coords = None, None
        # if self.use_metadata:
        #     location_coords = self._get_coords(image)
        #     metadata_idx = self.image_to_metadata_index.get(index, None)
        #     if metadata_idx is not None:
        #         row = self.metadata.iloc[metadata_idx]
        #         temporal_coords = self._get_date(row)

        # # to channels last
        # image = image.to_numpy()
        # if self.expand_temporal_dimension:
        #     image = rearrange(image, "(time channels) h w -> channels time h w", channels=len(self.bands))
        # image = np.moveaxis(image, 0, -1)

        # # filter bands
        # image = image[..., self.band_indices]

        
        # 1. 读取标签 Mask (确保路径转换为标准字符串型，防止 Path 对象传进 C++ 乱码)
        import rasterio
        from rasterio.env import Env

        # 用 rasterio 的本地 path 处理来避免编码问题
        label_path = str(sample_info["label_path"])

        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='YES', GDAL_CACHEMAX=0):
            mask_data = rioxarray.open_rasterio(label_path, masked=True)
        if self.no_label_replace is not None:
            mask_data = mask_data.fillna(self.no_label_replace)
        mask = mask_data.to_numpy()[0].astype(np.int64) # 取出第一通道，形状变成 [H, W]

        # 2. 循环读取 12 幅影像
        time_series_list = []
        for img_path in sample_info["image_paths"]:
            img_da = rioxarray.open_rasterio(str(img_path), masked=True)
            if self.no_data_replace is not None:
                img_da = img_da.fillna(self.no_data_replace)
            
            img_np = img_da.to_numpy() # 形状: [10, H, W]
            # 【核心修改】只切取前 6 个波段，满足 Prithvi 模型要求
            img_np = img_np[:6, :, :] # 形状变成 [6, H, W]
            time_series_list.append(img_np)

        # 堆叠 12 个时相： 形状从 list 变为 [12, 6, H, W]
        image = np.stack(time_series_list, axis=0)

        # 3. 准备进行 Albumentations 变换
        # Albumentations 要求输入形状为 [H, W, Channels]
        # 先利用 einops 平铺时序通道：[12, 6, H, W] -> [72, H, W]
        image = rearrange(image, "time channels h w -> (time channels) h w")
        # 换轴为 channels_last: [H, W, 72]
        image = np.moveaxis(image, 0, -1).astype(np.float32)

        # # ─── 寻找原代码中获取到 img 的位置 ───
        # # 假设此时 img 的形状是 (120, 224, 224)，即 12时相 * 10波段
        # # 或者是 (12, 10, 224, 224)

        # # 强制重塑为 [时间, 波段, H, W]
        # H, W = image.shape[-2], image.shape[-1]
        # image = image.reshape(12, 10, H, W)     

        # # 【核心修改】挑选出符合 Prithvi 要求的 6 个波段索引
        # # 这里的 [0, 1, 2, 3, 4, 5] 需要替换为你数据中 B, G, R, NIR, SWIR1, SWIR2 真实的索引位置
        # selected_indices = [0, 1, 2, 3, 4, 5] 
        # image = image[:, selected_indices, :, :]  # 裁剪后变为 [12, 6, 224, 224]

        # # 如果后续原代码期望的形状是平铺的 (72, 224, 224)
        # image = image.reshape(-1, H, W) 

        output = {
            "image": image,
            "mask": mask
        }

        # if self.reduce_zero_label:
        #     output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        # if self.use_metadata:
        #     output["location_coords"] = location_coords
        #     output["temporal_coords"] = temporal_coords

        return output

    def _load_file(self, path: Path, nan_replace: int | float | None = None) -> DataArray:
        data = rioxarray.open_rasterio(path, masked=True)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        num_images = self.time_steps + 2

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        images = sample["image"]
        images = images[rgb_indices, ...]  # Shape: (T, 3, H, W)

        processed_images = []
        for t in range(self.time_steps):
            img = images[:, t]
            img = img.numpy()
            img = to_rgb(img, rgb_indices, gamma=0.9)
            processed_images.append(img)

        mask = sample["mask"].numpy()
        if "prediction" in sample:
            num_images += 1
        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        for i, img in enumerate(processed_images):
            ax[i + 1].axis("off")
            ax[i + 1].title.set_text(f"T{i}")
            ax[i + 1].imshow(img)

        ax[self.time_steps + 1].axis("off")
        ax[self.time_steps + 1].title.set_text("Ground Truth Mask")
        ax[self.time_steps + 1].imshow(mask, cmap="jet", norm=norm)

        if "prediction" in sample:
            prediction = sample["prediction"]
            ax[self.time_steps + 2].axis("off")
            ax[self.time_steps + 2].title.set_text("Predicted Mask")
            ax[self.time_steps + 2].imshow(prediction, cmap="jet", norm=norm)

        cmap = plt.get_cmap("jet")
        legend_data = [[i, cmap(norm(i)), self.class_names[i]] for i in range(self.num_classes)]
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
