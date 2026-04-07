"""API Platform Remote Pipeline Client.

Calls the API Platform service via the OpenAI-compatible
``/v1/chat/completions`` endpoint.  The pipeline (layout detection,
parallel OCR, post-processing) runs entirely on the server side.

Request format::

    POST /v1/chat/completions
    {
        "model": "<model>",
        "extra": {"return_crop_images": false, ...},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "<data_uri>"}},
                {"type": "text",      "text": "<|PIPELINE_DOCUMENT_RECOGNITION|>"}
            ]
        }],
        "stream": false
    }

Response format::

    {
        "choices": [
            {"message": {"role": "assistant", "content": "<json_string>"}},
            {"message": {"role": "assistant", "content": "<markdown_string>"}}
        ]
    }

``choices[0].message.content`` — JSON result (list-of-pages, normalised
0-1000 bbox coords, same schema as the self-hosted pipeline).
``choices[1].message.content`` — Markdown result with
``![](page=N,bbox=[...])`` image placeholders.
"""

from __future__ import annotations

import base64
import os
import random
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter

from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.config import ApiPlatformConfig

logger = get_logger(__name__)

_PIPELINE_TRIGGER = "<|PIPELINE_DOCUMENT_RECOGNITION|>"
_DEFAULT_API_URL = "https://api.zhipuai-infra.cn/v1/chat/completions"
_DEFAULT_MODEL = "tob-glm-ocr-dev-test"


# ── helpers ──────────────────────────────────────────────────────────────────


def _sniff_mime(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "application/pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "application/octet-stream"


def _to_data_uri(mime: str, b64: str) -> str:
    return f"data:{mime};base64,{b64}"


# ── client ────────────────────────────────────────────────────────────────────


class ApiPlatformClient:
    """Client for the API Platform remote pipeline (chat completions format).

    Usage::

        from glmocr.api_platform_client import ApiPlatformClient
        from glmocr.config import ApiPlatformConfig

        cfg = ApiPlatformConfig(api_key="Bearer-token-here")
        client = ApiPlatformClient(cfg)
        client.start()

        response = client.parse("document.png")
        # response["choices"][0]["message"]["content"] → JSON string
        # response["choices"][1]["message"]["content"] → Markdown string
    """

    def __init__(self, config: "ApiPlatformConfig") -> None:
        self.api_url = config.api_url or _DEFAULT_API_URL
        self.model = config.model or _DEFAULT_MODEL
        self.api_key = (
            config.api_key
            or os.getenv("GLMOCR_API_PLATFORM_API_KEY")
        )
        self.verify_ssl = config.verify_ssl
        self.connect_timeout = config.connect_timeout
        self.request_timeout = config.request_timeout

        self.retry_max_attempts = config.retry_max_attempts
        self.retry_backoff_base = config.retry_backoff_base_seconds
        self.retry_backoff_max = config.retry_backoff_max_seconds
        self.retry_jitter_ratio = config.retry_jitter_ratio
        self.retry_status_codes = set(config.retry_status_codes)
        self._pool_maxsize = config.connection_pool_size or 16
        self._session: Optional[requests.Session] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=1,
                pool_maxsize=self._pool_maxsize,
                max_retries=0,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        logger.debug("ApiPlatformClient ready for %s", self.api_url)

    def stop(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self) -> "ApiPlatformClient":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    # ── file preparation ─────────────────────────────────────────────────────

    def _prepare_source(self, source: Union[str, Path, bytes]) -> str:
        """Return a data URI suitable for the ``image_url.url`` field.

        Accepts:
        * ``bytes``        – raw image / PDF bytes
        * ``str`` / Path   – local path, http(s) URL, or data URI
        """
        if isinstance(source, bytes):
            b64 = base64.b64encode(source).decode()
            return _to_data_uri(_sniff_mime(source), b64)

        s = str(source)

        if s.startswith(("http://", "https://")):
            # Remote URL: pass through directly
            return s

        if s.startswith("data:"):
            return s

        # Local file path
        path = Path(s)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        raw = path.read_bytes()

        # PDFs → always wrap as data URI
        if path.suffix.lower() == ".pdf" or raw[:5] == b"%PDF-":
            b64 = base64.b64encode(raw).decode()
            return _to_data_uri("application/pdf", b64)

        # Images: re-encode unsupported formats to JPEG/PNG via Pillow
        try:
            from PIL import Image

            img = Image.open(BytesIO(raw))
            fmt = (img.format or "").upper()
            if fmt in ("JPEG", "JPG", "PNG"):
                b64 = base64.b64encode(raw).decode()
                return _to_data_uri(_sniff_mime(raw), b64)

            # Convert to JPEG (lossy but compact) for anything else
            out = BytesIO()
            img.convert("RGB").save(out, format="JPEG", quality=92, optimize=True)
            converted = out.getvalue()
            b64 = base64.b64encode(converted).decode()
            return _to_data_uri("image/jpeg", b64)
        except Exception:
            b64 = base64.b64encode(raw).decode()
            return _to_data_uri(_sniff_mime(raw), b64)

    # ── request ───────────────────────────────────────────────────────────────

    def parse(
        self,
        source: Union[str, Path, bytes],
        request_id: Optional[str] = None,
        return_crop_images: bool = False,
        return_all_crop_images: bool = False,
        need_layout_visualization: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Parse a document using the remote API Platform pipeline.

        Args:
            source: Local file path, HTTP URL, data URI, or raw bytes.
            request_id: Optional request identifier forwarded as ``Request-Id``
                header (useful for server-side tracing).
            return_crop_images: Ask the server to return image-type region crops.
            return_all_crop_images: Ask the server to return all region crops.
            need_layout_visualization: Ask the server for layout visualisation.
            **kwargs: Additional fields merged into the ``extra`` payload dict.

        Returns:
            Raw API response dict.  Key fields:

            * ``choices[0].message.content`` – JSON string (list-of-pages)
            * ``choices[1].message.content`` – Markdown string
        """
        if self._session is None:
            self.start()

        image_url = self._prepare_source(source)

        extra: Dict[str, Any] = {
            "return_crop_images": return_crop_images,
            "return_all_crop_images": return_all_crop_images,
            "need_layout_visualization": need_layout_visualization,
        }
        extra.update(kwargs)

        payload: Dict[str, Any] = {
            "model": self.model,
            "extra": extra,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": _PIPELINE_TRIGGER},
                    ],
                }
            ],
            "stream": False,
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if request_id:
            headers["Request-Id"] = request_id

        return self._send_request(payload, headers)

    # ── HTTP + retry ──────────────────────────────────────────────────────────

    def _sleep_backoff(
        self,
        attempt: int,
        retry_after: Optional[float] = None,
    ) -> None:
        if retry_after is not None and retry_after > 0:
            secs = min(float(retry_after), self.retry_backoff_max)
        else:
            secs = min(
                self.retry_backoff_base * (2**attempt),
                self.retry_backoff_max,
            )
        jitter = secs * self.retry_jitter_ratio
        if jitter > 0:
            secs = max(0.0, secs + random.uniform(-jitter, jitter))
        time.sleep(secs)

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
        ra = resp.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    def _send_request(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        total = int(self.retry_max_attempts) + 1
        last_error: Optional[str] = None

        for attempt in range(total):
            try:
                resp = self._session.post(  # type: ignore[union-attr]
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout, self.request_timeout),
                    verify=self.verify_ssl,
                )

                if resp.status_code == 200:
                    return resp.json()

                status = resp.status_code
                preview = (resp.text or "")[:500]

                if status in self.retry_status_codes and attempt < total - 1:
                    logger.warning(
                        "API Platform returned %s (attempt %d/%d). "
                        "Retrying... response: %s",
                        status, attempt + 1, total, preview,
                    )
                    self._sleep_backoff(attempt, self._retry_after(resp))
                    continue

                logger.error(
                    "API Platform request failed with status %s: %s",
                    status, preview,
                )
                raise ValueError(
                    f"API Platform request failed with status {status}: {preview}"
                )

            except requests.exceptions.RequestException as exc:
                last_error = str(exc)
                if attempt < total - 1:
                    logger.warning(
                        "API Platform request error (attempt %d/%d): %s. Retrying...",
                        attempt + 1, total, last_error,
                    )
                    self._sleep_backoff(attempt)
                    continue
                logger.error("API Platform request failed: %s", last_error)
                logger.debug(traceback.format_exc())
                raise

            except Exception as exc:
                logger.error("Unexpected error during API Platform request: %s", exc)
                logger.debug(traceback.format_exc())
                raise

        raise ValueError(
            f"API Platform request failed after {total} attempts: {last_error}"
        )
