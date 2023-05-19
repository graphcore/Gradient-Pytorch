# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import yacs
import torch
import poptorch
import numpy as np
from PIL import Image
from poptorch import inferenceModel
from run import ipu_options
from typing import Dict, List
from utils.config import get_cfg_defaults
from torchvision.transforms import Compose
from notebook.visualization import plotting_tool
from utils.postprocessing import post_processing
from utils.tools import load_and_fuse_pretrained_weights
from utils.preprocessing import ResizeImage, Pad, ToNumpy, ToTensor


class YOLOv4InferencePipeline:
    def __init__(self, model, checkpoint_path):
        self.cfg = get_cfg_defaults()
        self.config = "configs/inference-yolov4p5.yaml"
        self.cfg.merge_from_file(self.config)
        self.cfg.freeze()
        self.checkpoint = checkpoint_path
        self.model = self.prepareInferenceYoloV4ModelForIPU(model)

    def __call__(self, image):
        transformed_image, size = self.preprocessImage(image)
        y = self.model(transformed_image)
        processed_batch = self.postprocessImage(y, size)

        return processed_batch

    def preprocessImage(self, img: Image):
        height, width = img.size
        print("original image dimensions:\nh:", height, "w:", width)

        img_conv = img.convert("RGB")

        # Change the data type of the dataloader depending of the options
        if self.cfg.model.uint_io:
            image_type = "uint"
        elif not self.cfg.model.ipu or not self.cfg.model.half:
            image_type = "float"
        else:
            image_type = "half"

        size = torch.as_tensor(img_conv.size)
        resize_img_mthd = ResizeImage(self.cfg.model.image_size)
        print("img_size: ", self.cfg.model.image_size)
        pad_mthd = Pad(self.cfg.model.image_size)
        image_to_tensor_mthd = Compose([ToNumpy(), ToTensor(int(self.cfg.dataset.max_bbox_per_scale), image_type)])
        dummy_label = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
        transformed_image, transformed_labels = resize_img_mthd((img_conv, dummy_label))
        transformed_image, transformed_labels = pad_mthd((transformed_image, transformed_labels))
        transformed_image, transformed_labels = image_to_tensor_mthd((transformed_image, transformed_labels))
        transformed_image, _ = torch.unsqueeze(transformed_image, dim=0), torch.unsqueeze(transformed_labels, dim=0)

        return transformed_image, size

    def prepareInferenceYoloV4ModelForIPU(self, original_model):
        mode = "inference"
        model = original_model(self.cfg)

        # Insert the pipeline splits if using pipeline
        if self.cfg.model.pipeline_splits:
            named_layers = {name: layer for name, layer in model.named_modules()}
            for ipu_idx, split in enumerate(self.cfg.model.pipeline_splits):
                named_layers[split] = poptorch.BeginBlock(ipu_id=ipu_idx + 1, layer_to_call=named_layers[split])

        model = load_and_fuse_pretrained_weights(model, self.checkpoint)
        model.optimize_for_inference()
        model.eval()

        # Create the specific ipu options if self.cfg.model.ipu
        ipu_opts = ipu_options(self.cfg, model, mode) if self.cfg.model.ipu else None

        model = inferenceModel(model, ipu_opts)

        return model

    def postprocessImage(self, y, size):
        dummy_label = []
        processed_batch = post_processing(self.cfg, y, [size], dummy_label)

        return processed_batch

    def plotImg(self, processed_batch, original_img):
        img_paths = plotting_tool(self.cfg, processed_batch[0], [original_img])
        return img_paths

    def detach(self):
        if self.model.isAttachedToDevice():
            self.model.detachFromDevice()
