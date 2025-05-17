import os
import time

import cv2
import numpy as np
from requests.exceptions import ChunkedEncodingError
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Florence2Segmenter:
    def __init__(self,  max_retries=3, retry_delay=2, device="cpu"):
        self.device = device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._initialize_model()

    def _initialize_model(self):
        for attempt in range(self.max_retries):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/Florence-2-base",
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    "microsoft/Florence-2-base",
                    trust_remote_code=True
                )
                break
            except ChunkedEncodingError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def set_mask_prompt(self, mask_prompt):
        self.mask_prompt = mask_prompt

    def process_image(self, image_input):
        image = cv2.cvtColor(image_input.copy(), cv2.COLOR_BGR2RGB)
        for attempt in range(self.max_retries):
            try:
                inputs = self.processor(
                    text=f"<REFERRING_EXPRESSION_SEGMENTATION>{self.mask_prompt}",
                    images=image,
                    return_tensors="pt"
                ).to(self.device, self.torch_dtype)

                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=4096,
                    num_beams=3,
                    do_sample=False
                )

                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                return self.processor.post_process_generation(
                    generated_text,
                    task="<REFERRING_EXPRESSION_SEGMENTATION>",
                    image_size=(image.shape[1], image.shape[0])  # width, height
                ), image

            except ChunkedEncodingError:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)

    def create_mask(self, parsed_answer, image_shape):
        """セグメンテーション領域を示すマスクを生成 (領域内=255, 領域外=0)"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        task_key = "<REFERRING_EXPRESSION_SEGMENTATION>"

        if task_key in parsed_answer:
            results = parsed_answer[task_key]
            for polygon_group in results.get('polygons', []):
                for coords in polygon_group:
                    if len(coords) >= 6 and len(coords) % 2 == 0:
                        points = np.array([(coords[j], coords[j+1]) for j in range(0, len(coords), 2)], dtype=np.int32)
                        cv2.fillPoly(mask, [points], color=255)
        return mask
