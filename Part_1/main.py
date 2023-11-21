import os
import glob
import argparse
import logging
import random
import shutil
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error
from tqdm import tqdm
import imagehash
from collections import defaultdict
import platform

import viewpoint_selector
from svg_processor import format_svg

if platform.system() == "Windows":
    from OCC.Display.SimpleGui import init_display
    from OCC.Extend.DataExchange import read_step_file, read_stl_file
    from OCC.Core.Quantity import *
    import OCC.Core.V3d

    PROJ_LIST = [
        OCC.Core.V3d.V3d_Zpos,
        OCC.Core.V3d.V3d_Zneg,
        OCC.Core.V3d.V3d_Xpos,
        OCC.Core.V3d.V3d_Xneg,
        OCC.Core.V3d.V3d_Ypos,
        OCC.Core.V3d.V3d_Yneg,
        OCC.Core.V3d.V3d_XposYpos,
        OCC.Core.V3d.V3d_XposZpos,
        OCC.Core.V3d.V3d_YposZpos,
        OCC.Core.V3d.V3d_XnegYneg,
        OCC.Core.V3d.V3d_XnegYpos,
        OCC.Core.V3d.V3d_XnegZneg,
        OCC.Core.V3d.V3d_XnegZpos,
        OCC.Core.V3d.V3d_YnegZneg,
        OCC.Core.V3d.V3d_YnegZpos,
        OCC.Core.V3d.V3d_XposYneg,
        OCC.Core.V3d.V3d_XposZneg,
        OCC.Core.V3d.V3d_YposZneg,
        OCC.Core.V3d.V3d_XposYposZpos,
        OCC.Core.V3d.V3d_XposYnegZpos,
        OCC.Core.V3d.V3d_XposYposZneg,
        OCC.Core.V3d.V3d_XnegYposZpos,
        OCC.Core.V3d.V3d_XposYnegZneg,
        OCC.Core.V3d.V3d_XnegYposZneg,
        OCC.Core.V3d.V3d_XnegYnegZpos,
        OCC.Core.V3d.V3d_XnegYnegZneg,
    ]

if platform.system() == "Linux":
    import torch

logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if platform.system() == "Linux":
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_class_info(path):
    classes = os.path.normpath(path).split(os.path.sep)
    if len(classes) > 1:
        classes.pop(0)
    return classes


def remove_all_temp_folders(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir in dirs:
            if dir.startswith("."):
                dir_to_remove = os.path.join(root, dir)
                print(f"dir_to_remove: {dir_to_remove}")
                try:
                    os.rmdir(dir_to_remove)
                except OSError as e:
                    print(f"remove_all_temp_folders Error: {dir_to_remove} - {e}")


def is_image_folder(directory):
    for item in os.listdir(directory):
        if item.lower().endswith((".png", ".jpg", ".jpeg")):
            return True

        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return False

    return False


def get_file_type_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".step", ".stp", ".stl")):
                return "cad"
            if file.endswith((".png", ".jpg", ".jpeg")):
                return "image"
    return ""


def get_duplicate_data_using_mse(args, files):
    tol = args.sim_tol
    duplicate_data = []

    img_list = [(f, np.array(Image.open(f))) for f in files]
    len_ = len(img_list)
    for i in range(len_):
        filepath, img = img_list[i]
        if filepath in duplicate_data:
            continue
        for j in range(i + 1, len_):
            mse = mean_squared_error(img, img_list[j][1])
            if mse <= tol:
                duplicate_data.append(filepath)
                if args.debug:
                    print("mse tol:", mse, tol)
                    print("file to be removed:", filepath)
                    print("file being compared:", img_list[j][0])
                break

    return duplicate_data


def get_duplicate_data_using_hash(args, files):
    image_hashes = {}
    for file in files:
        try:
            with Image.open(file) as img:
                h = imagehash.average_hash(img)
                image_hashes[file] = h
        except Exception as e:
            print(f"get_duplicate_data_using_hash Error processing {file}: {e}")

    seen_hashes = set()
    duplicate_data = []

    for file, h in image_hashes.items():
        if h in seen_hashes:
            duplicate_data.append(file)
        else:
            seen_hashes.add(h)

    return duplicate_data


def remove_duplicate_data(args, directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return

    if args.remove_duplicate_method == "hash":
        duplicate_data = get_duplicate_data_using_hash(args, files)
    else:
        duplicate_data = get_duplicate_data_using_mse(args, files)

    for file_to_be_removed in duplicate_data:
        os.remove(file_to_be_removed)
        if args.debug:
            print("remove_duplicate_data file_to_be_removed:", file_to_be_removed)


def get_bounding_box(img):
    img = img.convert("L")
    threshold = 128
    img = img.point(lambda p: p < threshold and 255) 
    bbox = img.getbbox()
    return bbox


def extra_processing(image_path, output_path, args):
    assert(args.width == args.height)

    target_size = args.width
    padding_size = args.padding_size

    img = Image.open(image_path)
    bbox = get_bounding_box(img)
    object_width = bbox[2] - bbox[0]
    object_height = bbox[3] - bbox[1]
    cropped = img.crop(bbox)

    if args.extra_processing_mode == 1:
        width_ratio = (target_size - padding_size) / object_width
        height_ratio = (target_size - padding_size) / object_height
    else:
        width_ratio = (target_size - padding_size) / img.width
        height_ratio = (target_size - padding_size) / img.height
    scale_ratio = min(width_ratio, height_ratio)

    new_width = int(object_width * scale_ratio)
    new_height = int(object_height * scale_ratio)
    resized = cropped.resize((new_width, new_height))
    final_img = Image.new("RGB", (target_size, target_size), "white")
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2
    final_img.paste(resized, (x_offset, y_offset))

    if args.out_format == "png":
        final_img.save(output_path, format="PNG")
    else:
        final_img.save(output_path, format="JPEG")


def resize_and_save(input_path, output_path, args):
    assert(args.width == args.height)
    with Image.open(input_path) as img:
        img_resized = img.resize((args.width, args.height))
        if args.out_format == "png":
            img_resized.save(output_path, "PNG")
        else:
            img_resized.save(output_path, "JPEG")


def check_shape_already_done(folder):
    file_list = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.jpeg"))
    return len(file_list) == len(PROJ_LIST)


def count_image_files(folder):
    total_count = 0
    for root, dirs, files in os.walk(folder):
        for filename in files:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in (".png", ".jpg", ".jpeg"):
                total_count += 1
            else:
                raise Exception(f"Error: count_image_files wrong file extension - {os.path.join(root, filename)}.")
    return total_count


def count_cad_files(folder):
    total_count = 0
    for root, dirs, files in os.walk(folder):
        for filename in files:
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension in (".step", ".stp", ".stl"):
                total_count += 1
    return total_count


def statistics_report(directory):
    from my_categories import CATEGORIES, CLASS_ENG2CHN
    file_type = get_file_type_in_directory(directory)
    report1 = defaultdict(int)
    report2 = defaultdict(int)

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if (file_type == "cad" and file_extension not in (".step", ".stp", ".stl")) or \
               (file_type == "image" and file_extension not in (".png", ".jpg", ".jpeg")):
                raise Exception(f"Error: statistics_report file extension not supported - {os.path.join(root, file)}.")
            parts = os.path.normpath(root).split(os.sep)
            class_info_list = parts[1:-1] if file_type == "image" else parts[1:]
            class_name = "/".join(class_info_list)
            report2[class_name] += 1
            report1[CATEGORIES[class_name]] += 1

    print("\n--- Taxonomy:")
    for class_name in report1:
        sub_class_list = []
        for sub_class_name in report2:
            if CATEGORIES[sub_class_name] == class_name:
                sub_class_list.append(sub_class_name)
        print(f"{class_name} ({CLASS_ENG2CHN[class_name]}): {sub_class_list}\n")

    total_num = 0
    print("\n--- Level 2 class info:")
    for class_name in report2:
        total_num += report2[class_name]
        print(f"{class_name}: {report2[class_name]}")

    print("\n--- Level 1 class info:")
    for class_name in report1:
        print(f"{class_name} ({CLASS_ENG2CHN[class_name]}): {report1[class_name]}")

    print(f"\n--- Total number: {total_num}")
    print(f"--- Level 1 class amount: {len(report1)}")
    print(f"--- Level 2 class amount: {len(report2)}")

    return report1, report2


def random_select_files(src_dir, dst_dir, N):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    count = 0
    for cad_class in tqdm(os.listdir(src_dir)):
        files = []
        cad_class_path = os.path.join(src_dir, cad_class)
        for cad_file in os.listdir(cad_class_path):
            cad_file_path = os.path.join(cad_class_path, cad_file)
            file_extension = os.path.splitext(cad_file)[1].lower()
            if file_extension not in (".step", ".stp", ".stl"):
                raise Exception(f"Error: random_select_files file extension not supported - {cad_file_path}.")
            files.append((cad_file_path, cad_class))

        selected_files = random.sample(files, N)

        for cad_file_path, cad_class in selected_files:
            cad_filename = os.path.basename(cad_file_path)
            dst_file_path = os.path.join(dst_dir, f'{cad_class}=={cad_filename}')
            shutil.copy(cad_file_path, dst_file_path)
            count += 1
    print(f"\n--- Total number of data being selected: {count}")


def my_draw(args, shape_path, out_path):
    if shape_path.lower().endswith(".stl"):
        shape = read_stl_file(shape_path)
    else:
        shape = read_step_file(shape_path)
    display, _, _, _ = init_display(size=(args.width, args.height), display_triedron=False, background_gradient_color1=[255,255,255], background_gradient_color2=[255,255,255])

    if args.out_type == "snapshot":
        ais_shapes = display.DisplayShape(shape)
    else:
        ais_shapes = display.DisplayShape(shape, color="black")

    for shp in ais_shapes:
        shp.SetWidth(args.line_width)

    if args.out_type == "hlr":
        display.SetModeHLR()
    elif args.out_type == "wireframe":
        display.SetModeWireFrame()
    display.EnableAntiAliasing()

    done_count = 0
    for i in range(len(PROJ_LIST)):
        out_path_i = out_path.format(i)
        if args.continue_task and os.path.exists(out_path_i):
            continue
        view = display.GetView()
        view.SetProj(PROJ_LIST[i])
        display.FitAll()
        view.Dump(out_path_i)
        done_count += 1

    return done_count


def main(args):
    remove_all_temp_folders(args.input_path)

    if args.statistics_report:
        if os.path.exists(args.input_path):
            statistics_report(args.input_path)
        elif os.path.exists(args.output_root_path):
            statistics_report(args.output_root_path)
        return

    if args.random_select:
        random_select_files(args.input_path, "out_rs", args.random_select_N)
        return

    if args.remove_duplicate == 2:
        dir_list = []
        for root, dirs, files in os.walk(args.input_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if is_image_folder(dir_path):
                    dir_list.append(dir_path)

        for dir_path in tqdm(dir_list):
            remove_duplicate_data(args, dir_path)
        return

    if args.extra_processing_mode:
        file_list = []
        for root, dirs, files in os.walk(args.input_path):
            for filename in files:
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                parts = os.path.normpath(root).split(os.sep)
                parts[0] = "out_ep"
                out_path = os.sep.join(parts)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                out_path = os.path.join(out_path, filename)
                out_path = f"{os.path.splitext(out_path)[0]}.{args.out_format}"
                file_list.append((os.path.join(root, filename), out_path))

        if args.extra_processing_mode == 3:
            for in_path, out_path in tqdm(file_list):
                resize_and_save(in_path, out_path, args)
            return

        for in_path, out_path in tqdm(file_list):
            try:
                extra_processing(in_path, out_path, args)
            except Exception as err:
                print(err, "---", in_path)
        return

    if args.format_svg_file:
        file_list = []
        for root, dirs, files in os.walk(args.input_path):
            for filename in files:
                if not filename.lower().endswith(".svg"):
                    continue
                in_path = os.path.join(root, filename)
                parts = os.path.normpath(root).split(os.sep)
                parts[0] = "out_fsvg"
                out_path = os.sep.join(parts)
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                out_path = os.path.join(out_path, filename)
                file_list.append((in_path, out_path))

        for in_path, out_path in tqdm(file_list):
            format_svg(in_path, out_path, args)
        return

    if args.viewpoint_selector_mode:
        dir_list = []
        for root, dirs, files in os.walk(args.input_path):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if is_image_folder(dir_path):
                    dir_list.append(dir_path)

        for dir_path in tqdm(dir_list):
            selected = viewpoint_selector.select(args, dir_path)
            parts = os.path.normpath(dir_path).split(os.sep)
            parts[0] = "out_vs"
            out_path = os.sep.join(parts)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            for file in selected:
                shutil.copy2(file, out_path)
        return

    done_count = count_image_files(args.output_root_path)
    final_total = len(PROJ_LIST) * count_cad_files(args.input_path)
    if args.continue_task and final_total == done_count:
        print(f"\n=== {done_count}/{final_total} === Done!")
        return

    for root, dirs, files in os.walk(args.input_path):
        for filename in files:
            if not filename.lower().endswith(".step") and \
               not filename.lower().endswith(".stp") and \
               not filename.lower().endswith(".stl"):
                continue

            classes = get_class_info(root)
            shape_path = os.path.join(root, filename)
            shape_name = os.path.splitext(filename)[0]
            out_folder = os.path.join(args.output_root_path, root, shape_name)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)

            classes.append(shape_name)
            prefix_name = "==".join(classes)
            out_path = os.path.join(out_folder, prefix_name + "_{}.png")

            if args.continue_task and check_shape_already_done(out_folder):
                print(f"\n=== {done_count}/{final_total} ===== Skip {shape_path} ==============================\n")
                continue

            print(f"\n=== {done_count}/{final_total} ===== Drawing {shape_path} ==============================\n")
            done_count += my_draw(args, shape_path, out_path)

            if args.remove_duplicate:
                remove_duplicate_data(args, out_folder)

    done_count = count_image_files(args.output_root_path)
    print(f"\n=== {done_count}/{final_total} === Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="folder to be processed.")
    parser.add_argument("-O", "--output_root_path", type=str, default="out", help="output path.")
    parser.add_argument("-W", "--width", type=int, default=512, help="width.")
    parser.add_argument("-H", "--height", type=int, default=512, help="height.")
    parser.add_argument("-lw", "--line_width", type=float, default=1.0, help="line width.")
    parser.add_argument("-st", "--sim_tol", type=float, default=20.0, help="similarity tolerance.")
    parser.add_argument("-rd", "--remove_duplicate", type=int, default=0, help="0: nothing; 1: remove duplicate during generation; 2: only remove duplicate.")
    parser.add_argument("-rdm", "--remove_duplicate_method", type=str, default="hash", help="mse, hash.")
    parser.add_argument("-D", "--debug", type=int, default=0, help="debug mode.")
    parser.add_argument("-con", "--continue_task", type=int, default=1, help="continue task.")
    parser.add_argument("-epm", "--extra_processing_mode", type=int, default=0, help="0 for nothing, 1 for padding-first, 2 for line-width-first, 3 for resize & save")
    parser.add_argument("-pad", "--padding_size", type=int, default=80, help="total padding size including both sides.")
    parser.add_argument("-F", "--out_format", type=str, default="png", help="output format (png, jpg).")
    parser.add_argument("-T", "--out_type", type=str, default="hlr", help="output type (hlr, wireframe, snapshot).")
    parser.add_argument("-vsm", "--viewpoint_selector_mode", type=int, default=0, help="0 for nothing, 1 for using ICNet.")
    parser.add_argument("-vst", "--viewpoint_selector_topN", type=int, default=5, help="top N views to be selected.")
    parser.add_argument("-fsvg", "--format_svg_file", type=int, default=0, help="format svg files.")
    parser.add_argument("-stat", "--statistics_report", type=int, default=0, help="output statistics report.")
    parser.add_argument("-RS", "--random_select", type=int, default=0, help="randomly select data.")
    parser.add_argument("-RSN", "--random_select_N", type=int, default=5, help="randomly selected number of data.")
    args = parser.parse_args()

    args.line_width = min(args.width, args.height) / 256 * args.line_width

    set_seed(0)
    main(args)

