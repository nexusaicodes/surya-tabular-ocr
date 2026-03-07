from io import BytesIO

from PIL import Image

from surya.common.util import expand_bbox
from surya.models import load_predictors


class TableExtractionPipeline:
    """High-level pipeline for extracting table structure (and optionally text) from images.

    Usage:
        pipeline = TableExtractionPipeline()
        result = pipeline.extract_tables(pil_image, ocr=True)
    """

    def __init__(self, device=None, dtype=None):
        predictors = load_predictors(device=device, dtype=dtype)
        self.layout_predictor = predictors["layout"]
        self.table_rec_predictor = predictors["table_rec"]
        self.recognition_predictor = predictors["recognition"]
        self.detection_predictor = predictors["detection"]

    def extract_tables(
        self,
        image: Image.Image | bytes,
        ocr: bool = True,
        skip_table_detection: bool = False,
    ) -> dict:
        """Extract table structure from an image.

        Args:
            image: PIL Image or raw bytes (PNG/JPEG/etc.)
            ocr: If True, also run text recognition on table regions.
            skip_table_detection: If True, treat the entire image as a single table
                (useful when the image is already a cropped table).

        Returns:
            Dict with "tables" list and "image_size" [width, height].
            Each table entry contains cells, rows, cols, image_bbox from TableResult,
            plus "text_lines" if ocr=True.
        """
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image)).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        table_imgs = []

        if skip_table_detection:
            table_imgs.append(image)
        else:
            layout_results = self.layout_predictor([image])
            layout_pred = layout_results[0]

            table_bboxes = [
                line.bbox
                for line in layout_pred.bboxes
                if line.label in ["Table", "TableOfContents"]
            ]

            for bbox in table_bboxes:
                expanded = expand_bbox(bbox)
                table_imgs.append(image.crop(expanded))

        if not table_imgs:
            return {"tables": [], "image_size": list(image.size)}

        table_preds = self.table_rec_predictor(table_imgs)

        tables = []
        for i, pred in enumerate(table_preds):
            table_dict = pred.model_dump()

            if ocr:
                ocr_results = self.recognition_predictor(
                    [table_imgs[i]], None, self.detection_predictor
                )
                table_dict["text_lines"] = [
                    line.model_dump() for line in ocr_results[0].text_lines
                ]

            tables.append(table_dict)

        return {"tables": tables, "image_size": list(image.size)}
