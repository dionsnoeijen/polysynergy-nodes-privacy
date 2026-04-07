import os
import io
import asyncio
import logging
import requests
from PIL import Image as PILImage, ImageFilter
from datetime import datetime

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
    name="Blur Regions",
    category="privacy",
    icon="blur.svg",
    version=1.0
)
class BlurRegions(Node):
    """
    Blur specific regions in an image based on bounding box coordinates.

    Takes an image and a list of regions (from PrivacyDetect or manual input)
    and applies blur effects to those regions.

    Supports gaussian blur, pixelation, and solid color masking.
    """

    input_image: Image = NodeVariableSettings(
        label="Input Image",
        info="Image to process (URL or image object)",
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

    regions: list = NodeVariableSettings(
        label="Regions",
        info="List of regions with bbox coordinates [x1, y1, x2, y2]",
        dock=True,
        has_in=True,
        required=True
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
        info="Extra pixels around each region",
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
        info="Image with blurred regions",
        has_out=True
    )

    regions_blurred: int = NodeVariableSettings(
        label="Regions Blurred",
        info="Number of regions that were blurred",
        has_out=True
    )

    true_path: dict = PathSettings(
        label="Success",
        info="Blur completed successfully"
    )

    false_path: dict = PathSettings(
        label="Error",
        info="Error during blur operation"
    )

    def get_base64_from_input(self):
        """Extract base64 data from input if available, returns bytes or None"""
        import base64

        def extract_b64(value):
            if isinstance(value, str):
                if value.startswith('data:image/'):
                    _, data = value.split(',', 1)
                    return base64.b64decode(data)
                if not value.startswith(('http://', 'https://', '{')):
                    try:
                        decoded = base64.b64decode(value, validate=True)
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

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _apply_blur(self, region_crop: PILImage.Image, blur_type: str, intensity: int) -> PILImage.Image:
        """Apply blur effect to a cropped region"""
        if blur_type == "gaussian":
            return region_crop.filter(ImageFilter.GaussianBlur(radius=intensity))
        elif blur_type == "pixelate":
            # Shrink and enlarge to create pixelation effect
            width, height = region_crop.size
            small_width = max(1, width // intensity)
            small_height = max(1, height // intensity)
            small = region_crop.resize((small_width, small_height), PILImage.Resampling.NEAREST)
            return small.resize((width, height), PILImage.Resampling.NEAREST)
        elif blur_type == "solid":
            rgb_color = self._hex_to_rgb(self.solid_color)
            return PILImage.new('RGB', region_crop.size, rgb_color)
        else:
            # Default to gaussian
            return region_crop.filter(ImageFilter.GaussianBlur(radius=intensity))

    async def execute(self):
        try:
            # Validate inputs
            if not self.regions:
                raise ValueError("No regions provided for blurring")

            if self.blur_intensity < 1 or self.blur_intensity > 50:
                raise ValueError("Blur intensity must be between 1 and 50")

            # Get image — base64_data input takes priority
            image_bytes = None
            if self.base64_data:
                import base64
                data = self.base64_data
                if data.startswith('data:'):
                    _, data = data.split(',', 1)
                image_bytes = base64.b64decode(data)
            if image_bytes is None:
                image_bytes = self.get_base64_from_input()
            if image_bytes is None:
                image_url = self.get_image_from_input()
                image_bytes = await self.download_image(image_url)

            # Load image
            pil_image = PILImage.open(io.BytesIO(image_bytes))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Process each region
            def _process_regions():
                img = pil_image.copy()
                blurred_count = 0

                for region in self.regions:
                    # Extract bbox - support both direct bbox and detection object format
                    if isinstance(region, dict) and 'bbox' in region:
                        bbox = region['bbox']
                    elif isinstance(region, (list, tuple)) and len(region) == 4:
                        bbox = region
                    else:
                        continue

                    x1, y1, x2, y2 = [int(c) for c in bbox]

                    # Add padding
                    x1 = max(0, x1 - self.padding)
                    y1 = max(0, y1 - self.padding)
                    x2 = min(img.width, x2 + self.padding)
                    y2 = min(img.height, y2 + self.padding)

                    # Skip invalid regions
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop region
                    region_crop = img.crop((x1, y1, x2, y2))

                    # Apply blur
                    blurred_region = self._apply_blur(
                        region_crop,
                        self.blur_type,
                        self.blur_intensity
                    )

                    # Paste back
                    img.paste(blurred_region, (x1, y1))
                    blurred_count += 1

                return img, blurred_count

            processed_image, blurred_count = await asyncio.to_thread(_process_regions)

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
                s3_key = f"{tenant_id}/{project_id}/blur_regions/blurred_{timestamp}.jpg"

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
                self.regions_blurred = blurred_count

                self.true_path = {
                    "regions_blurred": blurred_count,
                    "blur_type": self.blur_type,
                    "image_url": upload_result['url']
                }
            else:
                raise Exception("Failed to upload blurred image")

        except Exception as e:
            self.false_path = NodeError.format(e)
            self.blurred_image = None
            self.regions_blurred = 0
