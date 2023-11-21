import pickle
import os
import argparse
from glob import glob
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str)
parser.add_argument('--data_files', type=str, nargs='+')
parser.add_argument('--maskarea_files', type=str, nargs='+')
parser.add_argument('--min_maskarea', type=float, default=0.02)
parser.add_argument('--max_maskarea', type=float, default=0.9)
parser.add_argument('--no_tqdm', action='store_true')


def main(args):
    assert args.output_file is not None, "argument \'output_file\' must be specified!"
    assert args.data_files is not None, "argument \'data_files\' must be specified!"

    use_mask_area = (args.maskarea_files is not None and len(args.maskarea_files) > 0)\
        and (args.min_maskarea > 0.0 or args.max_maskarea < 1.0)

    data_files = []
    for path_format in args.data_files:
        data_files += glob(path_format, recursive='**' in path_format)
    
    if use_mask_area:
        maskarea_files = []
        for path_format in args.maskarea_files:
            maskarea_files += glob(path_format, recursive='**' in path_format)

        mask_areas = {}
        for chunk_path in maskarea_files:
            with open(chunk_path, 'rb') as file:
                mask_areas.update(pickle.load(file))
    
    if not args.no_tqdm:
        data_files = tqdm(data_files)

    path_dicts = {}
    for chunk_path in data_files:
        with open(chunk_path, 'rb') as file:
            chunk_path_dicts = pickle.load(file)

        if use_mask_area:
            for sample_name in list(chunk_path_dicts.keys()):
                assert sample_name in mask_areas,\
                    f"cannot find the mask area of sample \'{sample_name}\' in the mask-area files!\n"\
                    f"mask-area files:\n  {' '.join(sorted(maskarea_files))}."

                if not (args.min_maskarea < mask_areas[sample_name] < args.max_maskarea):
                    del chunk_path_dicts[sample_name]

        path_dicts.update(chunk_path_dicts)

    with open(args.output_file, 'wb') as file:
        pickle.dump(path_dicts, file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)