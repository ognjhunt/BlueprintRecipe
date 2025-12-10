"""
Gemini Client - Wrapper for Google Gemini API.

This module provides a client for interacting with Google's Gemini
models for multimodal scene understanding.
"""

import base64
import json
import os
from typing import Any, Optional


class GeminiClient:
    """
    Client for Google Gemini API.

    Supports:
    - Text-only generation
    - Multimodal generation (text + image)
    - Structured JSON output
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.0-pro",
        project_id: Optional[str] = None
    ):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._client = None

    def _get_client(self):
        """Get or create the Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model_name)
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install google-generativeai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        response_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text response from prompt.

        Args:
            prompt: The text prompt
            response_schema: Optional JSON schema to enforce
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        client = self._get_client()

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 8192,
        }

        if response_schema:
            generation_config["response_mime_type"] = "application/json"

        response = client.generate_content(
            prompt,
            generation_config=generation_config
        )

        return response.text

    def generate_with_image(
        self,
        prompt: str,
        image_data: bytes,
        image_mime_type: str = "image/jpeg",
        response_schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text response from prompt and image.

        Args:
            prompt: The text prompt
            image_data: Raw image bytes
            image_mime_type: MIME type of the image
            response_schema: Optional JSON schema to enforce
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        client = self._get_client()

        # Detect mime type from image data if not specified
        if image_mime_type == "image/jpeg" and image_data[:4] == b"\x89PNG":
            image_mime_type = "image/png"

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 8192,
        }

        if response_schema:
            generation_config["response_mime_type"] = "application/json"

        # Build multimodal content
        import google.generativeai as genai

        response = client.generate_content(
            [
                prompt,
                {"mime_type": image_mime_type, "data": image_data}
            ],
            generation_config=generation_config
        )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        output_schema: dict[str, Any],
        image_data: Optional[bytes] = None
    ) -> dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: The text prompt
            output_schema: JSON schema for the output
            image_data: Optional image bytes

        Returns:
            Parsed JSON response
        """
        # Add schema to prompt
        schema_prompt = f"""
{prompt}

You must respond with valid JSON that matches this schema:
{json.dumps(output_schema, indent=2)}

Only output the JSON, no additional text.
"""

        if image_data:
            response = self.generate_with_image(
                schema_prompt,
                image_data,
                response_schema=output_schema
            )
        else:
            response = self.generate(
                schema_prompt,
                response_schema=output_schema
            )

        # Parse JSON response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                json_str = response.strip()

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")


class MockGeminiClient(GeminiClient):
    """Mock Gemini client for testing without API access."""

    def __init__(self):
        super().__init__(api_key="mock", model_name="mock")

    def generate(self, prompt: str, **kwargs) -> str:
        """Return mock response."""
        return json.dumps({
            "environment_analysis": {
                "detected_type": "kitchen",
                "confidence": 0.9,
                "estimated_dimensions": {
                    "width": 4.0,
                    "depth": 3.5,
                    "height": 2.8
                }
            },
            "object_inventory": [
                {
                    "id": "refrigerator_01",
                    "category": "appliance",
                    "description": "Stainless steel French door refrigerator",
                    "is_articulated": True,
                    "articulation_type": "door",
                    "is_manipulable": False
                },
                {
                    "id": "counter_01",
                    "category": "furniture",
                    "description": "Kitchen counter with granite top",
                    "is_articulated": False,
                    "is_manipulable": False
                },
                {
                    "id": "mug_01",
                    "category": "dishes",
                    "description": "White ceramic coffee mug",
                    "is_articulated": False,
                    "is_manipulable": True,
                    "is_variation_candidate": True
                }
            ],
            "spatial_layout": {
                "placements": [
                    {"object_id": "refrigerator_01", "position": {"x": -1.5, "y": 0, "z": 0}, "rotation_degrees": 0},
                    {"object_id": "counter_01", "position": {"x": 0, "y": 0, "z": 0}, "rotation_degrees": 0},
                    {"object_id": "mug_01", "position": {"x": 0.2, "y": 0.9, "z": 0.1}, "rotation_degrees": 45}
                ],
                "relationships": [
                    {"type": "adjacent_to", "subject_id": "refrigerator_01", "object_id": "counter_01"},
                    {"type": "on_top_of", "subject_id": "mug_01", "object_id": "counter_01"}
                ]
            },
            "suggested_policies": [
                {"policy_id": "dexterous_pick_place", "relevance_score": 0.9, "rationale": "Mugs on counter suitable for pick-place"},
                {"policy_id": "articulated_access", "relevance_score": 0.8, "rationale": "Refrigerator doors for articulation training"}
            ]
        })

    def generate_with_image(self, prompt: str, image_data: bytes, **kwargs) -> str:
        """Return mock response."""
        return self.generate(prompt)
