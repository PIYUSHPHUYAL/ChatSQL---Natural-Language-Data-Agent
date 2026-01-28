"""
Ollama LLM Client - Custom wrapper for local Llama model
Built from scratch for ChatSQL project
"""

import requests
import json
from typing import Optional, Dict, Any


class OllamaClient:
    """
    Custom LLM client for Ollama.
    Handles communication with local Llama 3.1 model.
    """

    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama API endpoint
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

        # Test connection
        self._test_connection()

    def _test_connection(self) -> None:
        """Test if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding. Is it running?")

            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]

            if self.model_name not in model_names:
                raise ValueError(
                    f"Model {self.model_name} not found. "
                    f"Available: {model_names}"
                )

            print(f"✅ Connected to Ollama - Model: {self.model_name}")

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Cannot connect to Ollama. "
                "Make sure Ollama is running (check system tray)"
            )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate text from prompt using Llama model.

        Args:
            prompt: Input text prompt
            temperature: Randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum response length
            stream: Whether to stream response (not implemented yet)

        Returns:
            Generated text response
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()

            result = response.json()
            return result['response'].strip()

        except requests.exceptions.Timeout:
            raise TimeoutError("LLM took too long to respond (>60s)")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Ollama API: {e}")

    def generate_with_system(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate with system + user prompts (better for instructions).

        Args:
            system_prompt: System role instructions
            user_prompt: User's actual question
            temperature: Randomness level
            max_tokens: Max response length

        Returns:
            Generated response
        """
        # Combine system and user prompts
        full_prompt = f"""<|system|>
{system_prompt}

<|user|>
{user_prompt}

<|assistant|>
"""

        return self.generate(
            full_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )


# Test function
def test_ollama():
    """Quick test of Ollama client."""
    print("Testing Ollama client...")

    client = OllamaClient()

    # Simple test
    response = client.generate("Say 'Hello from Llama!' and nothing else.")
    print(f"\n🤖 LLM Response: {response}\n")

    # System prompt test
    response = client.generate_with_system(
        system_prompt="You are a helpful SQL expert.",
        user_prompt="What does SELECT * FROM users LIMIT 10; do?"
    )
    print(f"\n🤖 SQL Expert Response:\n{response}\n")

    print("✅ Ollama client working!")


if __name__ == "__main__":
    test_ollama()