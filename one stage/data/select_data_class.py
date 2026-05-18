import trimesh
import os
from pathlib import Path
from trimesh.viewer.windowed import SceneViewer


# 自定义查看器，确保按键 100% 捕获
class MyViewer(SceneViewer):
    def __init__(self, *args, **kwargs):
        self.marked_callback = kwargs.pop('marked_callback', None)
        super().__init__(*args, **kwargs)

    def on_key_press(self, symbol, modifiers):
        # 捕获 N 键 (pyglet.window.key.N)
        # 这里的 110 是小写 'n' 的键码，78 是大写 'N'
        from pyglet.window import key
        if symbol == key.N:
            if self.marked_callback:
                self.marked_callback()

        # 调用父类方法，处理原有的缩放、退出等逻辑
        super().on_key_press(symbol, modifiers)


def recursive_visualize_and_mark(root_folder, output_file="Crowded_Malocclusion.txt"):
    root_path = Path(root_folder)
    obj_files = sorted(list(root_path.rglob("*.obj")))

    if not obj_files:
        print(f"未找到文件: {root_folder}")
        return

    marked_files = []

    print(f"共找到 {len(obj_files)} 个文件。")
    print(">>> 操作说明: ")
    print("  - 请确保在英文输入法状态下")
    print("  - [N] 键: 记录当前文件 (会有控制台输出)")
    print("  - [关闭窗口]: 下一个文件")

    for i, file_path in enumerate(obj_files):
        print(f"--- [{i + 1}/{len(obj_files)}] 预览: {file_path.name} ---")

        try:
            mesh = trimesh.load(str(file_path))

            # 1. 预处理：解决缩放跳变。将模型中心移至原点，并统一缩放
            mesh.apply_translation(-mesh.centroid)
            scale = 1.0 / mesh.extents.max()
            mesh.apply_scale(scale)

            # 2. 视觉效果：设置为干净的浅灰色
            mesh.visual.face_colors = [200, 200, 200, 255]

            scene = trimesh.Scene(mesh)

            # 定义记录逻辑
            def do_mark():
                full_path = str(file_path.resolve())
                if full_path not in marked_files:
                    marked_files.append(full_path)
                    print(f" [OK] 已成功记录: {file_path.name}")
                else:
                    print(f" [!] 该文件之前已记录过")

            # 3. 使用自定义 Viewer 启动
            # 这会替换默认的 scene.show()
            MyViewer(scene,
                     resolution=[1024, 768],
                     caption=f"PRESS N TO SAVE | {file_path.name}",
                     marked_callback=do_mark)

        except Exception as e:
            print(f"加载错误 {file_path.name}: {e}")

    # 4. 保存结果
    if marked_files:
        with open(output_file, "w", encoding="utf-8") as f:
            for item in marked_files:
                f.write(item + "\n")
        print(f"\n全部完成！共标记 {len(marked_files)} 个文件。结果见: {output_file}")


def data_select():
    import shutil
    file_names = [
        "processed_Normal Cases.txt",
        "processed_Missing Teeth.txt",
        "processed_Crowded_Malocclusion.txt"
    ]

    # 1. 改为列表 [] 保证顺序
    save_names = ["Normal_testc", "Missing_testc", "Crowded_testc"]

    file_root = "G:/teethMICCAI2022/3DTeethLand_landmarks_test/"
    save_root = "G:/teethMICCAI2022/train_land_onenet_data/"

    # 循环读取 3 个 TXT 文件
    for txt_file, folder_name in zip(file_names, save_names):
        txt_path = os.path.join(file_root, txt_file)

        if not os.path.exists(txt_path):
            print(f"跳过：找不到 TXT 文件 {txt_path}")
            continue

        print(f"--- 正在处理文件: {txt_file} -> 目标文件夹: {folder_name} ---")

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # 2. 增加一层循环，遍历 TXT 里的每一行
                lines = f.readlines()
                for line in lines:
                    content = os.path.basename(line.strip()).replace(".obj", "")  # 3. 去掉换行符
                    if not content:
                        continue  # 跳过空行

                    # 构建源路径和目标路径
                    # 建议使用 os.path.join 避免斜杠问题
                    src_path = os.path.join(save_root, "testc", f"{content}_c.npy")

                    # 确保目标子文件夹存在
                    target_dir = os.path.join(save_root, folder_name)
                    os.makedirs(target_dir, exist_ok=True)

                    dst_path = os.path.join(target_dir, f"{content}_c.npy")

                    # 执行拷贝
                    if os.path.exists(src_path):
                        shutil.copy(src_path, dst_path)
                    else:
                        print(f"警告：找不到源文件 {src_path}")

        except Exception as e:
            print(f"处理 {txt_file} 时出错: {e}")


if __name__ == "__main__":
    # 请确保路径正确
    TARGET_FOLDER = 'H:/teethMICCAI2022/raw_data/test/data_part_777/'
    #recursive_visualize_and_mark(TARGET_FOLDER)

    data_select()