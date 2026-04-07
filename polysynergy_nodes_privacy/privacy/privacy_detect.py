import os
import io
import asyncio
import logging
import requests
import numpy as np
from PIL import Image as PILImage
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
    name="Privacy Detect",
    category="privacy",
    icon="detect.svg",
    version=1.0
)
class PrivacyDetect(Node):
    """
    Detect privacy-sensitive objects in images using YOLOv8.

    Detects faces, persons, and license plates and returns their
    bounding box coordinates for further processing (e.g., blurring).

    Uses Ultralytics YOLOv8 models for object detection.
    """

    input_image: Image = NodeVariableSettings(
        label="Input Image",
        info="Image to analyze (URL or image object)",
        dock=True,
        has_in=True,
        required=True
    )

    base64_data: str = NodeVariableSettings(
        label="Base64 Data",
        info="Base64-encoded image data (takes priority over Input Image)",
        dock=True,
        has_in=True
    )

    detection_type: str = NodeVariableSettings(
        label="Detection Type",
        info="Type of privacy-sensitive objects to detect",
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

    annotate_image: bool = NodeVariableSettings(
        label="Annotate Image",
        info="Output an annotated image with bounding boxes",
        dock=True,
        has_in=True,
        default=False
    )

    # Outputs
    detections: list = NodeVariableSettings(
        label="Detections",
        info="List of detected objects with bbox coordinates",
        has_out=True
    )

    detection_count: int = NodeVariableSettings(
        label="Count",
        info="Number of objects detected",
        has_out=True
    )

    annotated_image: Image = NodeVariableSettings(
        label="Annotated Image",
        info="Image with bounding boxes drawn",
        has_out=True
    )

    true_path: dict = PathSettings(
        label="Success",
        info="Detection completed successfully"
    )

    false_path: dict = PathSettings(
        label="Error",
        info="Error during detection"
    )

    # COCO class IDs
    COCO_PERSON_CLASS = 0

    # Model cache
    _models = {}

    def get_base64_from_input(self):
        """Extract base64 data from input if available, returns bytes or None"""
        import base64 as b64

        def extract_b64(value):
            if isinstance(value, str):
                if value.startswith('data:image/'):
                    _, data = value.split(',', 1)
                    return b64.b64decode(data)
                if not value.startswith(('http://', 'https://', '{')):
                    try:
                        decoded = b64.b64decode(value, validate=True)
                        if len(decoded) > 100:
                            return decoded
                    except Exception:
                        pass
            elif isinstance(value, dict):
                if 'base64' in value:
                    return extract_b64(value['base64'])
            return None

        return extract_b64(self.input_image)

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

    # Custom class IDs for our combined detection
    CLASS_PERSON = 0
    CLASS_LICENSE_PLATE = 100  # Custom ID for license plates

    def _get_class_name(self, class_id: int, detection_type: str) -> str:
        """Convert class ID to human-readable name"""
        if class_id == self.CLASS_LICENSE_PLATE:
            return "license_plate"
        elif detection_type == "persons" or class_id == self.COCO_PERSON_CLASS:
            return "person"
        elif detection_type == "faces":
            return "face"
        return f"class_{class_id}"

    def _get_model(self, model_name: str = "yolov8n.pt"):
        """Get or load a cached YOLO model"""
        if model_name not in self._models:
            self._models[model_name] = YOLO(model_name)
        return self._models[model_name]

    def _run_person_detection(self, image_array: np.ndarray, conf: float) -> sv.Detections:
        """Run person/face detection using YOLOv8n"""
        model = self._get_model("yolov8n.pt")
        results = model(image_array, conf=conf, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter to only persons (class 0)
        if len(detections) > 0:
            mask = detections.class_id == self.COCO_PERSON_CLASS
            detections = detections[mask]
        return detections

    # License plate model URL and path
    LICENSE_PLATE_MODEL_URL = 'https://github.com/ablanco1950/LicensePlate_Yolov8_MaxFilters/raw/main/best.pt'
    LICENSE_PLATE_MODEL_PATH = '/tmp/license_plate_best.pt'

    def _get_license_plate_model(self):
        """Download and cache the license plate detection model"""
        import urllib.request
        import os

        if 'license_plate' not in self._models:
            # Download model if not exists
            if not os.path.exists(self.LICENSE_PLATE_MODEL_PATH):
                logger.info("Downloading license plate detection model...")
                urllib.request.urlretrieve(self.LICENSE_PLATE_MODEL_URL, self.LICENSE_PLATE_MODEL_PATH)
                logger.info("License plate model downloaded successfully")

            self._models['license_plate'] = YOLO(self.LICENSE_PLATE_MODEL_PATH)

        return self._models['license_plate']

    def _run_license_plate_detection(self, image_array: np.ndarray, conf: float) -> sv.Detections:
        """Run license plate detection using a dedicated license plate model"""
        try:
            # Use dedicated license plate model
            model = self._get_license_plate_model()
            results = model(image_array, conf=conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Filter to only license plates (class 0 in this model)
            if len(detections) > 0:
                plate_mask = detections.class_id == 0  # 'license-plate' class
                detections = detections[plate_mask]

                # Remap class IDs to our custom LICENSE_PLATE class
                if len(detections) > 0:
                    detections.class_id = np.full(len(detections), self.CLASS_LICENSE_PLATE)

            return detections

        except Exception as e:
            logger.warning(f"License plate detection failed: {e}")
            return sv.Detections.empty()

    async def execute(self):
        try:
            # Validate inputs
            if self.confidence_threshold < 0 or self.confidence_threshold > 1:
                raise ValueError("Confidence threshold must be between 0.0 and 1.0")

            # Get image — base64_data input takes priority
            image_bytes = None
            if self.base64_data:
                import base64 as b64
                data = self.base64_data
                if data.startswith('data:'):
                    _, data = data.split(',', 1)
                image_bytes = b64.b64decode(data)
            if image_bytes is None:
                image_bytes = self.get_base64_from_input()
            if image_bytes is None:
                image_url = self.get_image_from_input()
                image_bytes = await self.download_image(image_url)

            # Load image for inference
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            def _run_inference():
                # Convert PIL to numpy array for inference
                image_array = np.array(pil_image)

                all_detections = []

                # Run appropriate detection based on type
                if self.detection_type in ["persons", "faces", "all"]:
                    person_detections = self._run_person_detection(image_array, self.confidence_threshold)
                    if len(person_detections) > 0:
                        all_detections.append(person_detections)

                if self.detection_type in ["license_plates", "all"]:
                    plate_detections = self._run_license_plate_detection(image_array, self.confidence_threshold)
                    if len(plate_detections) > 0:
                        all_detections.append(plate_detections)

                # Combine all detections
                if len(all_detections) == 0:
                    return sv.Detections.empty(), image_array
                elif len(all_detections) == 1:
                    return all_detections[0], image_array
                else:
                    # Merge multiple detection results
                    combined = sv.Detections.merge(all_detections)
                    return combined, image_array

            detections, image_array = await asyncio.to_thread(_run_inference)

            # Convert to output format
            detection_list = []
            if len(detections) > 0:
                for i in range(len(detections)):
                    bbox = detections.xyxy[i].tolist()
                    conf = float(detections.confidence[i])
                    cls_id = int(detections.class_id[i])

                    detection_list.append({
                        "bbox": bbox,  # [x1, y1, x2, y2]
                        "confidence": conf,
                        "class": self._get_class_name(cls_id, self.detection_type),
                        "class_id": cls_id
                    })

            self.detections = detection_list
            self.detection_count = len(detection_list)

            # Create annotated image if requested
            if self.annotate_image and len(detections) > 0:
                def _annotate():
                    box_annotator = sv.BoxAnnotator(thickness=2)
                    label_annotator = sv.LabelAnnotator()

                    labels = [
                        f"{self._get_class_name(int(cls), self.detection_type)} {conf:.2f}"
                        for cls, conf in zip(detections.class_id, detections.confidence)
                    ]

                    annotated = box_annotator.annotate(
                        scene=image_array.copy(),
                        detections=detections
                    )
                    annotated = label_annotator.annotate(
                        scene=annotated,
                        detections=detections,
                        labels=labels
                    )

                    return PILImage.fromarray(annotated)

                annotated_pil = await asyncio.to_thread(_annotate)

                # Upload annotated image
                def _upload_annotated():
                    img_bytes = io.BytesIO()
                    annotated_pil.save(img_bytes, format='JPEG', quality=85)
                    img_bytes.seek(0)
                    image_data = img_bytes.getvalue()

                    s3_service = S3Service()
                    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    tenant_id = os.getenv('TENANT_ID', 'unknown')
                    project_id = os.getenv('PROJECT_ID', 'unknown')
                    s3_key = f"{tenant_id}/{project_id}/privacy_detect/annotated_{timestamp}.jpg"

                    result = s3_service.upload_image(
                        image_data=image_data,
                        key=s3_key,
                        content_type='image/jpeg'
                    )
                    return result

                upload_result = await asyncio.to_thread(_upload_annotated)

                if upload_result['success']:
                    self.annotated_image = {
                        "url": upload_result['url'],
                        "mime_type": "image/jpeg",
                        "width": pil_image.width,
                        "height": pil_image.height
                    }

            self.true_path = {
                "detection_count": self.detection_count,
                "detection_type": self.detection_type,
                "detections": self.detections
            }

        except Exception as e:
            self.false_path = NodeError.format(e)
            self.detections = []
            self.detection_count = 0
            self.annotated_image = None
