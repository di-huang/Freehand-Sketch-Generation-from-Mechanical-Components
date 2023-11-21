import os


def select(args, dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        return

    if args.viewpoint_selector_mode == 1:
        from models.IC_viewpoint_selector import IC_score

        topN = args.viewpoint_selector_topN
        img_dict = {}

        for file in files:
            score = IC_score(file)
            img_dict[file] = score

        sorted_items = sorted(img_dict.items(), key=lambda x: x[1], reverse=True)
        return [item[0] for item in sorted_items[:topN]]

    return []

