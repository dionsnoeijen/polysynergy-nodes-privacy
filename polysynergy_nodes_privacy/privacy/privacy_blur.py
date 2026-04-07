import os
import io
import asyncio
import logging
import requests
import numpy as np
from PIL import Image as PILImage, ImageFilter
from datetime import datetime
from ultralytics import YOLO
import supervision as sv

from polysynergy_node_runner.setup_context.node import Node
from polysynergy_node_runner.setup_context.node_decorator import node
from polysynergy_node_runner.setup_context.node_error import NodeError
from polysynergy_node_runner.setup_context.node_variable_settings import NodeVariableSettings
from polysynergy_node_runner.setup_context.dock_property import dock_property
from polysynergy_node_runner.setup_context.path_settings import PathSettings
from polysynergy_node_runner.services.s3_service import S3Service
from polysynergy_nodes.image.types import Image

logger = logging.getLogger(__name__)


@node(
    name="Privacy Blur",
    category="privacy",
    icon="privacy.svg",
    version=1.0
)
class PrivacyBlur(Node):
    """
    All-in-one privacy protection node.

    Combines detection and blurring in a single node for convenience.
    Detects faces, persons, or license plates and automatically blurs them.

    Perfect for quick privacy protection without needing to connect
    separate detection and blur nodes.
    """

    input_image: Image = NodeVariableSettings(
        label="Input Image",
        info="Image to process (URL or image object)",
        dock=True,
        has_in=True,
        required=True
    )

    detection_type: str = NodeVariableSettings(
        label="Detection Type",
        info="Type of privacy-sensitive objects to detect and blur",
        dock=dock_property(
            select_values={
                "faces": "Faces only",
                "persons": "Full persons",
                "license_plates": "License plates",
                "all": "All privacy objects"
            }
        ),
        has_in=True,
        default="faces"
    )

    confidence_threshold: float = NodeVariableSettings(
        label="Confidence",
        info="Minimum confidence threshold (0.0 - 1.0)",
        dock=True,
        has_in=True,
        default=0.5
    )

    blur_type: str = NodeVariableSettings(
        label="Blur Type",
        info="Type of blur effect to apply",
        dock=dock_property(
            select_values={
                "gaussian": "Gaussian blur",
                "pixelate": "Pixelate",
                "solid": "Solid color"
            }
        ),
        has_in=True,
        default="gaussian"
    )

    blur_intensity: int = NodeVariableSettings(
        label="Intensity",
        info="Blur radius or pixel size (1-50)",
        dock=True,
        has_in=True,
        default=20
    )

    padding: int = NodeVariableSettings(
        label="Padding",
        info="Extra pixels around each detection",
        dock=True,
        has_in=True,
        default=5
    )

    solid_color: str = NodeVariableSettings(
        label="Solid Color",
        info="Color for solid blur type (hex format)",
        dock=True,
        has_in=True,
        default="#000000"
    )

    # Outputs
    blurred_image: Image = NodeVariableSettings(
        label="Blurred Image",
        info="Image with privacy regions blurred",
        has_out=True
    )

    detections: list = NodeVariableSettings(
        label="Detections",
        info="List of detected objects (for logging/audit)",
        has_out=True
    )

    detection_count: int = NodeVariableSettings(
        label="Count",
        info="Number of objects detected and blurred",
        has_out=True
    )

    true_path: dict = PathSettings(
        label="Success",
        info="Privacy blur completed successfully"
    )

    false_path: dict = PathSettings(
        label="Error",
        info="Error during privacy blur"
    )

    COCO_PERSON_CLASS = 0

    # Model cache
    _models = {}

    def get_image_from_input(self):
        """Extract image URL from various input formats"""
        import json

        def extract_url(value):
            """Recursively extract URL from potentially nested structures"""
            if isinstance(value, str):
                # Check if it's a JSON string
                if value.startswith('{'):
                    try:
                        parsed = json.loads(value)
                        return extract_url(parsed)
                    except json.JSONDecodeError:
                        pass
                # It's a direct URL string
                if value.startswith('http'):
                    return value
            elif isinstance(value, dict):
                # Check for url field
                if 'url' in value:
                    return extract_url(value['url'])
                elif 'image_url' in value:
                    return extract_url(value['image_url'])
            return None

        url = extract_url(self.input_image)
        if url:
            return url

        raise ValueError(f"Could not extract URL from input: {type(self.input_image)}")

    async def download_image(self, url: str) -> bytes:
        """Download image from URL and return bytes"""
        def _sync_download():
            # Check if this is a MinIO/S3 URL that we can access directly via S3 client
            if 'polysynergy-' in url and '-media' in url:
                # Extract bucket name and key from URL
                # URL format: http://host:port/bucket-name/key?params
                try:
                    from urllib.parse import urlparse, unquote
                    parsed = urlparse(url)
                    path_parts = parsed.path.lstrip('/').split('/', 1)
                    if len(path_parts) == 2:
                        bucket_name = path_parts[0]
                        key = unquote(path_parts[1])

                        # Use S3 client directly
                        s3_service = S3Service()
                        response = s3_service.s3_client.get_object(Bucket=bucket_name, Key=key)
                        return response['Body'].read()
                except Exception as e:
                    logger.warning(f"Failed to download via S3 client, falling back to HTTP: {e}")

            # Fallback: try HTTP request (replace localhost for Docker)
            download_url = url.replace('localhost:9000', 'minio:9000')
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()
            return response.content
        return await asyncio.to_thread(_sync_download)

    def _get_class_name(self, class_id: int, detection_type: str) -> str:
        """Convert class ID to human-readable name"""
        if detection_type == "persons" or class_id == self.COCO_PERSON_CLASS:
            return "person"
        elif detection_type == "faces":
            return "face"
        elif detection_type == "license_plates":
            return "license_plate"
        return f"class_{class_id}"

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _apply_blur(self, region_crop: PILImage.Image, blur_type: str, intensity: int) -> PILImage.Image:
        """Apply blur effect to a cropped region"""
        if blur_type == "gaussian":
            return region_crop.filter(ImageFilter.GaussianBlur(radius=intensity))
        elif blur_type == "pixelate":
            width, height = region_crop.size
            small_width = max(1, width // intensity)
            small_height = max(1, height // intensity)
            small = region_crop.resize((small_width, small_height), PILImage.Resampling.NEAREST)
            return small.resize((width, height), PILImage.Resampling.NEAREST)
        elif blur_type == "solid":
            rgb_color = self._hex_to_rgb(self.solid_color)
            return PILImage.new('RGB', region_crop.size, rgb_color)
        else:
            return region_crop.filter(ImageFilter.GaussianBlur(radius=intensity))

    def _get_model(self, model_name: str = "yolov8n.pt"):
        """Get or load a cached YOLO model"""
        if model_name not in self._models:
            self._models[model_name] = YOLO(model_name)
        return self._models[model_name]

    async def execute(self):
        try:
            # Validate inputs
            if self.confidence_threshold < 0 or self.confidence_threshold > 1:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")

            if self.blur_intensity < 1 or self.blur_intensity > 50:
                raise ValueError("Blur intensity must be between 1 and 50")

            # Get image
            image_url = self.get_image_from_input()
            image_bytes = await self.download_image(image_url)

            # Load image
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            def _run_detection():
                image_array = np.array(pil_image)
                model = self._get_model("yolov8n.pt")
                results = model(image_array, conf=self.confidence_threshold, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)
                return detections

            detections = await asyncio.to_thread(_run_detection)

            # Filter by class
            if len(detections) > 0:
                if self.detection_type in ["persons", "faces"]:
                    mask = detections.class_id == self.COCO_PERSON_CLASS
                    detections = detections[mask]

            # Convert to output format
            detection_list = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    bbox = detections.xyxy[i].tolist()
                    conf = float(detections.confidence[i])
                    cls_id = int(detections.class_id[i])

                    detection_list.append({
                        "bbox": bbox,
                        "confidence": conf,
                        "class": self._get_class_name(cls_id, self.detection_type),
                        "class_id": cls_id
                    })

            self.detections = detection_list
            self.detection_count = len(detection_list)

            # Apply blur to detections
            def _apply_blurs():
                img = pil_image.copy()

                for detection in detection_list:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(c) for c in bbox]

                    # Add padding
                    x1 = max(0, x1 - self.padding)
                    y1 = max(0, y1 - self.padding)
                    x2 = min(img.width, x2 + self.padding)
                    y2 = min(img.height, y2 + self.padding)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop, blur, paste
                    region_crop = img.crop((x1, y1, x2, y2))
                    blurred_region = self._apply_blur(
                        region_crop,
                        self.blur_type,
                        self.blur_intensity
                    )
                    img.paste(blurred_region, (x1, y1))

                return img

            processed_image = await asyncio.to_thread(_apply_blurs)

            # Upload result
            def _upload_image():
                img_bytes = io.BytesIO()
                processed_image.save(img_bytes, format='JPEG', quality=90)
                img_bytes.seek(0)
                image_data = img_bytes.getvalue()

                s3_service = S3Service()
                timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                tenant_id = os.getenv('TENANT_ID', 'unknown')
                project_id = os.getenv('PROJECT_ID', 'unknown')
                s3_key = f"{tenant_id}/{project_id}/privacy_blur/blurred_{timestamp}.jpg"

                result = s3_service.upload_image(
                    image_data=image_data,
                    key=s3_key,
                    content_type='image/jpeg'
                )
                return result

            upload_result = await asyncio.to_thread(_upload_image)

            if upload_result['success']:
                self.blurred_image = {
                    "url": upload_result['url'],
                    "mime_type": "image/jpeg",
                    "width": processed_image.width,
                    "height": processed_image.height
                }

                self.true_path = {
                    "detection_count": self.detection_count,
                    "detection_type": self.detection_type,
                    "blur_type": self.blur_type,
                    "image_url": upload_result['url']
                }
            else:
                raise Exception("Failed to upload blurred image")

        except Exception as e:
            self.false_path = NodeError.format(e)
            self.blurred_image = None
            self.detections = []
            self.detection_count = 0
