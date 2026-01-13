"""
Ollama service for LLM inference
"""
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    """Service for interacting with Ollama API"""

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.default_model = settings.DEFAULT_MODEL
        self.timeout = aiohttp.ClientTimeout(total=60)  # 60 second timeout

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response from Ollama

        Args:
            prompt: User prompt
            system_prompt: System prompt for persona
            model: Model name (defaults to config)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        if model is None:
            model = self.default_model

        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }

        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{self.base_url}/api/generate"
                logger.info(f"Sending request to Ollama: {url} with model {model}")

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise Exception(f"Ollama API error: {response.status}")

                    result = await response.json()
                    return result.get("response", "").strip()

        except aiohttp.ClientError as e:
            logger.error(f"Network error communicating with Ollama: {str(e)}")
            raise Exception(f"Failed to communicate with Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama service: {str(e)}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Ollama

        Returns:
            List of available models
        """
        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{self.base_url}/api/tags"
                logger.info(f"Fetching models from: {url}")

                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise Exception(f"Ollama API error: {response.status}")

                    result = await response.json()
                    return result.get("models", [])

        except aiohttp.ClientError as e:
            logger.error(f"Network error communicating with Ollama: {str(e)}")
            raise Exception(f"Failed to communicate with Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error listing models: {str(e)}")
            raise

    async def check_health(self) -> bool:
        """
        Check if Ollama service is healthy

        Returns:
            True if healthy, False otherwise
        """
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception as e:
            logger.warning(f"Ollama health check failed: {str(e)}")
            return False

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama library

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:  # 5 minute timeout for pulling
                url = f"{self.base_url}/api/pull"
                payload = {"name": model_name}

                logger.info(f"Pulling model {model_name} from Ollama")

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Failed to pull model: {response.status} - {error_text}")
                        return False

                    return True

        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False