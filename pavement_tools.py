import h5py
import numpy as np
import pandas as pd
import cv2


def stitch_road_from_h5(h5_filepath, num_segments=None, overlap_rows=5, verbose=True):
    """
    从 HDF5 数据库中读取路面高程矩阵并进行无缝边缘拼接。

    参数:
        h5_filepath (str): .h5 数据库文件的相对或绝对路径。
        num_segments (int, optional): 指定要拼接的块数。如果为 None，则拼接数据库中所有的块。
        overlap_rows (int): 计算高差时，用于边缘对齐的行数（默认使用交界处的 5 行）。
        verbose (bool): 是否打印处理过程的提示信息。

    返回:
        np.ndarray: 拼接完成的完整 2D 路面高程矩阵。如果出错则返回 None。
    """
    try:
        with h5py.File(h5_filepath, 'r') as h5f:
            if 'road_segments' not in h5f:
                raise KeyError("HDF5 文件中未找到 'road_segments' 数据组！")

            road_group = h5f['road_segments']
            # 自动获取所有数据块名称并排序 (如 '0001', '0002')
            segment_names = sorted(list(road_group.keys()))

            if not segment_names:
                raise ValueError("HDF5 数据库中没有数据块！")

            if num_segments is not None:
                segment_names = segment_names[:num_segments]

            if verbose:
                print(f"⏳ 准备拼接 {len(segment_names)} 块路面数据...")

            # 1. 动态获取单块数据的裁剪后维度，初始化空白矩阵
            first_data = road_group[segment_names[0]][:]
            data_cropped_shape = first_data[1:-1, 1:-1].shape
            zz = np.zeros(data_cropped_shape)

            # 2. 循环读取并执行边缘拼接逻辑
            for name in segment_names:
                # 懒加载：仅在循环到当前块时，才将其从硬盘读入内存
                data = road_group[name][:]

                # 边缘对齐高差计算（避开左右可能存在的不稳定边缘，仅取中间段 4:-5）
                mean_zz_edge = np.mean(zz[-overlap_rows:, 4:-5])
                mean_data_edge = np.mean(data[:overlap_rows, 4:-5])
                gaocha = mean_zz_edge - mean_data_edge

                # 矩阵裁剪与高差补偿拼接
                data_cropped = data[1:-1, 1:-1] + gaocha
                zz = np.vstack((zz, data_cropped))

            # 3. 去除初始化时用到的第一块空白 0 矩阵
            zz_final = zz[data_cropped_shape[0]:, :]

            if verbose:
                print(f"✅ 拼接完成！最终生成的高程矩阵维度为: {zz_final.shape}")

            return zz_final

    except Exception as e:
        print(f"❌ 拼接工具执行失败: {e}")
        return None


def export_pavement_data(zz_matrix, output_prefix="pavement"):
    """
    将拼接好的高程矩阵导出为 Streamlit 软件所需的一维 CSV 和 二维 PNG 图像。
    """
    if zz_matrix is None:
        print("❌ 传入的矩阵为空，无法导出！")
        return

    # 1. 导出 1D 轮廓用于 Welch 功率谱分析 (Tab 2)
    middle_col_index = zz_matrix.shape[1] // 2
    profile_1d = zz_matrix[:, middle_col_index]

    csv_filename = f"{output_prefix}_1d_profile.csv"
    pd.DataFrame({'Elevation': profile_1d}).to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"📄 已导出 1D 轮廓数据: {csv_filename}")

    # 2. 导出 2D 灰度图像用于 2D FFT 纹理分析 (Tab 1)
    # 将高程归一化到 0-255 并转为 uint8 格式
    zz_norm = ((zz_matrix - zz_matrix.min()) / (zz_matrix.max() - zz_matrix.min()) * 255).astype(np.uint8)

    png_filename = f"{output_prefix}_2d_texture.png"
    # 使用 imencode 绕过中文路径报错问题
    cv2.imencode('.png', zz_norm)[1].tofile(png_filename)
    print(f"🖼️ 已导出 2D 纹理图像: {png_filename}")


# =========================================
# 本地测试模块 (仅在直接运行此文件时执行)
# =========================================
if __name__ == "__main__":
    test_h5_path = r'data/PavementDatabase.h5'

    # 测试 1: 调用拼接工具 (比如先拼 20 块试试水)
    merged_matrix = stitch_road_from_h5(test_h5_path, num_segments=20)

    # 测试 2: 导出结果
    if merged_matrix is not None:
        export_pavement_data(merged_matrix, output_prefix="test_road")