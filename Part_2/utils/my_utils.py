import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import xml.etree.ElementTree as ET
import re
import random


def tensor2image(tensor, out="my_output_image.jpg"):
    tensor_to_pil = transforms.ToPILImage()
    image = tensor_to_pil(tensor)
    image.save(out)


def swap_pairs(lst):
    for i in range(0, len(lst) - 1, 2):
        lst[i], lst[i + 1] = lst[i + 1], lst[i]


def extra_process(lst):
    for i in range(len(lst)):
        lst[i] = lst[i] / 224
        lst[i] = -lst[i] + 1
        lst[i] = 1 - lst[i] * 2


def get_pos_from_svg(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    namespaces = {
        'svg': 'http://www.w3.org/2000/svg'
    }

    paths = root.findall('.//svg:path', namespaces)

    coords_list = [path.get('d') for path in paths]

    pattern = r'(-?\d+(\.\d+)?)\s+(-?\d+(\.\d+)?)'
    ret = []

    for coord_str in coords_list:
        matches = re.findall(pattern, coord_str)
        coord_lst = [float(coord) for match in matches for coord in match[::2]]
        swap_pairs(coord_lst)
        extra_process(coord_lst)
        ret.append(coord_lst)

    return ret


def get_path_dict(args, input_path):
    # iter_list = args.key_steps
    iter_list = ['0', '100', '200', '300', '400', '500', '600', '700', '800'] # TEMP
    path_dict = {}
    seed = 0 # seed set to be 0 by default
    idx = 0

    for folder_name in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder_name, 'svg_logs')
        if not os.path.isdir(folder_path):
            continue

        path_dict_ = {k: {} for k in iter_list}
        for svg_file in os.listdir(folder_path):
            start = svg_file.find("iter") + len("iter")
            end = svg_file.find(".svg", start)
            iter_number = svg_file[start:end]
            if iter_number in iter_list:
                svg_path = os.path.join(folder_path, svg_file)
                pos_list = get_pos_from_svg(svg_path)
                path_dict_[iter_number]['pos'] = pos_list
                path_dict_[iter_number]['color'] = [[0.0, 0.0, 0.0]] * len(pos_list)
                path_dict_[iter_number]['radius'] = [args.radius] * len(pos_list)

        img_path = os.path.join(input_path, folder_name, 'input.png')
        path_dict[f'{idx}_{seed}'] = {'img_path': img_path, 'iterations': path_dict_}
        idx += 1

    return path_dict


def my_save_plots(batch, save_path):
    merged_image = torchvision.utils.make_grid(batch, nrow=4)
    merged_image_numpy = merged_image.permute(1, 2, 0).cpu().numpy()
    pil_image = Image.fromarray((merged_image_numpy * 255).astype('uint8'))
    pil_image.save(save_path)


if __name__ == "__main__":
    pass
