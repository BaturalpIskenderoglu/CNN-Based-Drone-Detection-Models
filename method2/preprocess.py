import cv2
import os
import json

def preprocess(src_folder, target_folder, max_size):
    """
        Resize images with size higher than max_size
        Only update the fields where read_annotations() method in DroneDataset.py reads
    """

    os.makedirs(target_folder, exist_ok=True)

    coco_path  = os.path.join(src_folder, "_annotations.coco.json")

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    new_coco = coco.copy()

    new_coco["images"] = []
    new_coco["annotations"] = []

    for index, image_info in enumerate(coco["images"]):

        if index % 500 == 0:
            print(f"{index} images preprocessed")

        image_id = image_info["id"]
        filename = image_info["file_name"]

        input_image_path = os.path.join(src_folder, filename)

        output_image_path = os.path.join(target_folder, filename)

        image = cv2.imread(input_image_path)

        if image is None:
            print(f"Can not read: {input_image_path}")
            continue

        # change sizes if larges than max_size
        original_h, original_w = image.shape[:2]

        new_width = original_w
        new_height = original_h

        max_original_size = max(original_w, original_h)

        scale = 1

        if max_original_size > max_size:
            scale = max_size / max_original_size
            new_width = int(original_w * scale)
            new_height = int(original_h * scale)

        if scale == 1:
            resized_image = image
        else:
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_image_path, resized_image)

        # change new_coco
        
        new_image_info = image_info.copy()

        new_image_info["width"] = new_width
        new_image_info["height"] = new_height

        new_coco["images"].append(new_image_info)

        for annotation in coco["annotations"]:

            if annotation["image_id"] != image_id:
                continue

            new_annotation = annotation.copy()

            x, y, bbox_w, bbox_h = annotation["bbox"]

            new_x = x * scale
            new_y = y * scale

            new_bbox_w = bbox_w * scale
            new_bbox_h = bbox_h * scale

            new_annotation["bbox"] = [
                new_x,
                new_y,
                new_bbox_w,
                new_bbox_h
            ]

            new_coco["annotations"].append(new_annotation)

    target_coco_path = os.path.join(target_folder, "_annotations.coco.json")
    with open(target_coco_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=4, ensure_ascii=False)

    print("Finished")

