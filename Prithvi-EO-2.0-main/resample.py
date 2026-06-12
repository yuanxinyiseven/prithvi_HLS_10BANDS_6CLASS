import os
from pathlib import Path
import rasterio
from rasterio.enums import Resampling

def resample_tiff(input_path, output_path, target_shape=(224, 224), is_label=False):
    """
    对单张 TIFF 影像进行重采样
    :param input_path: 输入文件路径
    :param output_path: 输出文件路径
    :param target_shape: 目标尺寸 (height, width)
    :param is_label: 是否为标签数据（标签用最邻近插值，影像用双线性插值）
    """
    with rasterio.open(input_path) as src:
        # 计算新的仿射变换参数 (Affine Transformation)
        transform = src.transform * src.transform.scale(
            (src.width / target_shape[1]),
            (src.height / target_shape[0])
        )
        
        # 复制原影像的元数据并更新尺寸和变换矩阵
        profile = src.profile.copy()
        profile.update({
            'height': target_shape[0],
            'width': target_shape[1],
            'transform': transform
        })
        
        # 根据数据类型选择插值方式（对应 ArcMap/QGIS 的 Resampling Type）
        # 标签必须用 NEAREST，防止分类值变成小数；影像建议用 BILINEAR 或 CUBIC
        resample_method = Resampling.nearest if is_label else Resampling.bilinear
        
        # 读取并进行重采样
        data = src.read(
            out_shape=(src.count, target_shape[0], target_shape[1]),
            resampling=resample_method
        )
        
        # 写入新文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

def batch_resample_workspace(root_dir, target_size=224):
    """
    遍历工作空间，按原结构批量重采样
    :param root_dir: E盘下的根目录或需要处理的顶层目录
    """
    root_path = Path(root_dir)
    target_shape = (target_size, target_size)
    
    # 1. 遍历E盘下的第一层子文件夹（各个独立的任务/数据集文件夹）
    for sub_dir in root_path.iterdir():
        if not sub_dir.is_dir():
            continue
            
        # 跳过已经是生成的 _224 文件夹，防止重复处理
        if sub_dir.name.endswith(f'_{target_size}'):
            continue
            
        print(f"正在处理项目文件夹: {sub_dir.name}")
        
        # 创建对应的输出顶层文件夹，例如: "E:/Project" -> "E:/Project_224"
        output_sub_dir = sub_dir.parent / f"{sub_dir.name}_{target_size}"
        
        # 2. 检索当前文件夹下所有的 TIFF 文件
        tiff_files = list(sub_dir.rglob("*.tif")) + list(sub_dir.rglob("*.tiff"))
        
        for tiff_path in tiff_files:
            # 获取相对路径，用于在目标文件夹中重建结构
            relative_path = tiff_path.relative_to(sub_dir)
            
            # 判断当前文件是否属于 label 文件夹（通过路径片段判断）
            is_label = "label" in [part.lower() for part in relative_path.parts]
            
            # 构造新的文件名，如 "img.tif" -> "img_224.tif"
            new_filename = f"{tiff_path.stem}_{target_size}{tiff_path.suffix}"
            
            # 拼接最终的输出绝对路径
            output_file_path = output_sub_dir / relative_path.parent / new_filename
            
            try:
                resample_tiff(
                    input_path=str(tiff_path),
                    output_path=str(output_file_path),
                    target_shape=target_shape,
                    is_label=is_label
                )
                print(f"成功: {relative_path} -> {output_file_path.name}")
            except Exception as e:
                print(f"失败: {tiff_path}，错误原因: {e}")

if __name__ == "__main__":
    # 请将此处修改为你在E盘下的实际工作根目录
    # 如果要直接处理整个E盘，可以写 "E:/"，但建议先指定到具体的子总目录进行测试
    workspace_directory = r"e:\LN_data_10bands_6class_64"
    
    if os.path.exists(workspace_directory):
        print("开始批量重采样...")
        batch_resample_workspace(workspace_directory, target_size=224)
        print("所有重采样任务已完成！")
    else:
        print(f"路径不存在，请检查: {workspace_directory}")