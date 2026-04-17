import os
import numpy as np
import scipy.io as sio
import h5py


def convert_mat_to_h5(src_folder, h5_filepath):
    """
    将文件夹下的所有 .mat 高程矩阵打包进一个 HDF5 数据库文件
    """
    print(f"📂 正在扫描目录: {src_folder}")

    files = [f for f in os.listdir(src_folder) if f.endswith('.mat')]
    files.sort()  # 确保 0001, 0002 顺序正确

    if not files:
        print("❌ 未找到任何 .mat 文件！")
        return

    print(f"⏳ 开始构建 HDF5 数据库，共 {len(files)} 个文件...")

    # 以写模式 ('w') 创建一个全新的 .h5 文件
    with h5py.File(h5_filepath, 'w') as h5f:
        # 创建一个主组 (Group)，类似于文件夹，用来分类存放数据
        road_group = h5f.create_group("road_segments")

        # 记录基础的元数据 (Metadata)，这会让你们的项目显得非常专业！
        road_group.attrs['description'] = "路面三维高程矩阵集合"
        road_group.attrs['unit'] = "mm"

        success_count = 0

        for f in files:
            mat_path = os.path.join(src_folder, f)
            try:
                # 1. 加载 mat 文件
                mat_data = sio.loadmat(mat_path)

                # 2. 提取矩阵数据
                matrix_data = None
                if 'im' in mat_data:
                    matrix_data = mat_data['im']
                elif 'z' in mat_data:
                    matrix_data = mat_data['z']

                if matrix_data is None:
                    continue

                # 3. 数据集命名 (例如把 '0001.mat' 变成 '0001')
                dataset_name = f.replace('.mat', '')

                # 4. 存入 HDF5 数据库
                # compression="gzip" 开启压缩，能大幅减小文件体积
                road_group.create_dataset(
                    dataset_name,
                    data=matrix_data,
                    compression="gzip",
                    compression_opts=4  # 压缩级别 0-9，4 是速度和体积的极佳平衡
                )
                success_count += 1

            except Exception as e:
                print(f"❌ 转换 {f} 时出错: {e}")

    print(f"🎉 数据库构建完成！成功写入 {success_count}/{len(files)} 个路面片段。")
    print(f"👉 HDF5 数据库已保存在: {h5_filepath}")


if __name__ == "__main__":
    # 你的源文件夹路径
    source_dir = 'data/mat2'

    # 你想生成的 .h5 文件的保存路径及名字
    output_h5 = 'data/PavementDatabase.h5'

    convert_mat_to_h5(source_dir, output_h5)