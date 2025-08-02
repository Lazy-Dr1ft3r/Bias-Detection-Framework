


"""
Intersectional Bias Detection Framework for Large Language Models: Phase 2
Model Integration Module - Open Source Models Edition
"""

import os
import json
import time
import pandas as pd
import requests
import logging
import asyncio
from typing import List, Dict, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import aiohttp
from tqdm import tqdm
import backoff


def convert_json_to_csv(json_path, csv_path):
    """Convert the JSON prompts to a CSV file format for Phase 2 processing."""
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract prompts and metadata
    rows = []
    
    # Handle the nested structure from Phase 1
    if isinstance(data, dict):
        for prompt_type, prompts in data.items():
            for i, item in enumerate(prompts):
                if isinstance(item, dict) and "prompt" in item:
                    row = {
                        "prompt_id": f"{prompt_type}_{i}",
                        "prompt_text": item["prompt"],
                        "prompt_type": item["type"],
                        "domain": item["domain"]
                    }
                    
                    # Add dimension values
                    if "values" in item:
                        for dim, value in item["values"].items():
                            row[f"dimension_{dim}"] = value
                    
                    rows.append(row)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Converted {len(rows)} prompts to CSV at {csv_path}")

# Example usage
json_path = "data/prompts/complete_test_suite.json"
csv_path = "bias_test_prompts.csv"
convert_json_to_csv(json_path, csv_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("model_integration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ModelIntegration")


class LLMInterface(ABC):
    """
    Abstract base class for LLM API interfaces.
    """
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the LLM model
            api_key: API key for authentication (if needed)
        """
        self.model_name = model_name
        # Create logger first before using it
        self.logger = logging.getLogger(f"LLMInterface.{model_name}")
        # Now get the API key
        self.api_key = api_key or self._get_api_key_from_env()
    
    def _get_api_key_from_env(self) -> str:
        """
        Get API key from environment variables.
        
        Returns:
            API key as string
        """
        env_var_name = f"{self.__class__.__name__.upper()}_API_KEY"
        api_key = os.environ.get(env_var_name)
        
        if not api_key:
            self.logger.warning(f"API key not found in environment variable {env_var_name}")
        
        return api_key
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the LLM for a given prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        pass


class MistralInterface(LLMInterface):
    """
    Interface for Mistral AI's open models via HuggingFace Inference API (free tier).
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", api_key: str = None):
        """
        Initialize the Mistral interface via HuggingFace Inference API.
        
        Args:
            model_name: Name of the Mistral model on HuggingFace
            api_key: HuggingFace API key (optional for some models)
        """
        super().__init__(model_name, api_key)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if API key is provided
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=5,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Mistral model via HuggingFace Inference API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        # Format prompt for Mistral instruction format
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Prepare the request payload
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status} - {await response.text()}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text
                    response_text = response_data[0]["generated_text"]
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text for Mistral models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words (actual tokenization may vary)
        return len(text.split()) * 1.3  # ~1.3 tokens per word as a rough estimate


class LLaMALocalInterface(LLMInterface):
    """
    Interface for locally hosted LLaMA models via standard endpoints.
    """
    
    def __init__(self, model_name: str = "llama-2-7b", api_url: str = "http://localhost:8000/v1"):
        """
        Initialize the LLaMA local interface.
        
        Args:
            model_name: Name/identifier of the local LLaMA model
            api_url: URL where the model is being served
        """
        super().__init__(model_name, api_key=None)  # No API key for local models
        self.api_url = f"{api_url.rstrip('/')}/completions"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=3,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using a locally hosted LLaMA model.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        # Prepare the request payload (following OpenAI-like API)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text
                    response_text = response_data["choices"][0]["text"]
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text for LLaMA models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words (LLaMA uses different tokenization)
        return len(text.split()) * 1.5  # ~1.5 tokens per word as a rough estimate


class OllamaInterface(LLMInterface):
    """
    Interface for Ollama API, which provides easy access to multiple open-source models.
    """
    
    def __init__(self, model_name: str = "mistral", api_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama interface.
        
        Args:
            model_name: Name of the model in Ollama (e.g., llama2, mistral, falcon)
            api_url: URL where Ollama is being served
        """
        super().__init__(model_name, api_key=None)  # No API key needed for Ollama
        self.api_url = f"{api_url.rstrip('/')}/api/generate"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=3,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the Ollama API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        # Prepare the request payload for Ollama
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text from Ollama
                    response_text = response_data["response"]
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time,
                        "tokens": {
                            "total": response_data.get("eval_count", 0)
                        }
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text for Ollama models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words
        return len(text.split()) * 1.4  # ~1.4 tokens per word as a rough estimate


class DeepSeekInterface(LLMInterface):
    """
    Interface for DeepSeek AI's open-source models via Ollama or HuggingFace.
    """
    
    def __init__(self, model_name: str = "deepseek-coder", api_url: str = "http://localhost:11434", use_ollama: bool = True):
        """
        Initialize the DeepSeek interface.
        
        Args:
            model_name: Name of the DeepSeek model (e.g., deepseek-coder, deepseek-llm)
            api_url: URL for API access
            use_ollama: Whether to use Ollama (True) or direct API (False)
        """
        self.use_ollama = use_ollama
        
        # Adjust model name for Ollama if needed
        if use_ollama and not model_name.startswith("deepseek"):
            model_name = f"deepseek-{model_name}"
        
        super().__init__(model_name)
        
        if use_ollama:
            self.api_url = f"{api_url.rstrip('/')}/api/generate"
        else:
            # Use HuggingFace style endpoint for direct API
            self.api_url = f"{api_url.rstrip('/')}/generate"
        
        self.headers = {
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=5,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using DeepSeek models.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        if self.use_ollama:
            # Use Ollama API format
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False
            }
        else:
            # Use standard API format (similar to HuggingFace)
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "do_sample": True
                }
            }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status} - {await response.text()}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text (format depends on API)
                    if self.use_ollama:
                        response_text = response_data.get("response", "")
                    else:
                        response_text = response_data.get("generated_text", "")
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text for DeepSeek models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words
        return len(text.split()) * 1.5  # ~1.5 tokens per word as a rough estimate


class HuggingFaceInferenceInterface(LLMInterface):
    """
    Interface for HuggingFace Inference API, which provides access to many open-source models.
    """
    
    def __init__(self, model_name: str = "google/flan-t5-large", api_key: str = None):
        """
        Initialize the HuggingFace Inference API interface.
        
        Args:
            model_name: Name of the model on HuggingFace
            api_key: HuggingFace API key (optional for some models)
        """
        super().__init__(model_name, api_key)
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {
            "Content-Type": "application/json"
        }
        
        # Add authorization header if API key is provided
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=5,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using the HuggingFace Inference API.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        # Prepare the request payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status} - {await response.text()}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text - format varies between models
                    if isinstance(response_data, list):
                        # Text generation models
                        response_text = response_data[0].get("generated_text", "")
                    elif isinstance(response_data, dict):
                        # Some models return a direct result
                        response_text = response_data.get("generated_text", str(response_data))
                    else:
                        response_text = str(response_data)
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text for HuggingFace models.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words
        return len(text.split()) * 1.3  # ~1.3 tokens per word as a rough estimate


class LocalInferenceServer(LLMInterface):
    """
    Interface for models running on a local inference server like TGI (Text Generation Inference).
    """
    
    def __init__(self, model_name: str = "local-model", api_url: str = "http://localhost:8080"):
        """
        Initialize the local inference server interface.
        
        Args:
            model_name: Name/identifier of the local model
            api_url: URL where the inference server is running
        """
        super().__init__(model_name, api_key=None)  # No API key for local models
        self.api_url = f"{api_url.rstrip('/')}/generate"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, aiohttp.ClientError),
        max_tries=3,
        factor=2
    )
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response using a local inference server.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Default parameters
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 500)
        
        # Prepare the request payload for TGI-compatible servers
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "do_sample": True
            }
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = f"API Error: {response.status}"
                        self.logger.error(error_msg)
                        return {
                            "success": False,
                            "error": error_msg,
                            "prompt": prompt,
                            "response_text": None,
                            "model": self.model_name,
                            "elapsed_time": time.time() - start_time
                        }
                    
                    response_data = await response.json()
                    
                    # Extract response text
                    response_text = response_data.get("generated_text", "")
                    
                    return {
                        "success": True,
                        "prompt": prompt,
                        "response_text": response_text,
                        "model": self.model_name,
                        "elapsed_time": time.time() - start_time
                    }
                    
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": self.model_name,
                "elapsed_time": time.time() - start_time
            }
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a given text.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation based on words
        return len(text.split()) * 1.5  # ~1.5 tokens per word as a rough estimate


class ModelManager:
    """
    Class for managing multiple LLM interfaces and collecting responses.
    """
    
    def __init__(self, models: List[LLMInterface] = None):
        """
        Initialize the ModelManager with a list of LLM interfaces.
        
        Args:
            models: List of LLM interface instances
        """
        self.models = models or []
        self.logger = logging.getLogger("ModelManager")
    
    def add_model(self, model: LLMInterface) -> None:
        """
        Add an LLM interface to the manager.
        
        Args:
            model: LLM interface instance
        """
        self.models.append(model)
    
    def get_model_by_name(self, model_name: str) -> Optional[LLMInterface]:
        """
        Get an LLM interface by model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            LLM interface instance if found, None otherwise
        """
        for model in self.models:
            if model.model_name == model_name:
                return model
        return None
    
    async def generate_single_response(self, model: LLMInterface, prompt: str, 
                                      **kwargs) -> Dict[str, Any]:
        """
        Generate a response from a single model.
        
        Args:
            model: LLM interface instance
            prompt: Input prompt text
            **kwargs: Additional parameters for the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            return await model.generate_response(prompt, **kwargs)
        except Exception as e:
            error_msg = f"Error generating response from {model.model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "prompt": prompt,
                "response_text": None,
                "model": model.model_name,
                "elapsed_time": 0
            }
    
    async def generate_responses(self, prompt: str, models: List[LLMInterface] = None, 
                                **kwargs) -> List[Dict[str, Any]]:
        """
        Generate responses from multiple models for a single prompt.
        
        Args:
            prompt: Input prompt text
            models: List of LLM interface instances to use (default: all)
            **kwargs: Additional parameters for the APIs
            
        Returns:
            List of dictionaries containing responses and metadata
        """
        models_to_use = models or self.models
        
        tasks = [self.generate_single_response(model, prompt, **kwargs) for model in models_to_use]
        responses = await asyncio.gather(*tasks)
        
        return responses
    
    async def batch_process_prompts(self, prompts: List[str], model_names: List[str] = None,
                                   max_concurrency: int = 5, **kwargs) -> List[List[Dict[str, Any]]]:
        """
        Process a batch of prompts with selected models.
        
        Args:
            prompts: List of prompt texts
            model_names: List of model names to use (default: all)
            max_concurrency: Maximum number of concurrent requests
            **kwargs: Additional parameters for the APIs
            
        Returns:
            List of lists of response dictionaries
        """
        # Select models to use
        if model_names:
            models_to_use = [model for model in self.models if model.model_name in model_names]
        else:
            models_to_use = self.models
        
        self.logger.info(f"Batch processing {len(prompts)} prompts with {len(models_to_use)} models")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def process_with_semaphore(prompt):
            async with semaphore:
                return await self.generate_responses(prompt, models=models_to_use, **kwargs)
        
        # Process prompts with limited concurrency
        tasks = [process_with_semaphore(prompt) for prompt in prompts]
        
        # Show progress bar
        all_responses = []
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing prompts"):
            response = await task
            all_responses.append(response)
        
        # Sort results to match input order
        # (as_completed may return results in a different order)
        prompt_to_responses = {responses[0]["prompt"]: responses for responses in all_responses}
        ordered_responses = [prompt_to_responses[prompt] for prompt in prompts]
        
        return ordered_responses
    
    async def process_dataset(self, dataset_path: str, output_path: str,
                             model_names: List[str] = None, prompt_column: str = "prompt_text",
                             max_concurrency: int = 5, **kwargs) -> pd.DataFrame:
        """
        Process all prompts in a dataset and save responses.
        
        Args:
            dataset_path: Path to the dataset CSV file
            output_path: Path where to save the results
            model_names: List of model names to use (default: all)
            prompt_column: Name of the column containing prompts
            max_concurrency: Maximum number of concurrent requests
            **kwargs: Additional parameters for the APIs
            
        Returns:
            DataFrame with the results
        """
        # Load dataset
        df = pd.read_csv(dataset_path)
        prompts = df[prompt_column].tolist()
        
        # Process prompts
        all_responses = await self.batch_process_prompts(
            prompts=prompts,
            model_names=model_names,
            max_concurrency=max_concurrency,
            **kwargs
        )
        
        # Prepare results DataFrame
        results = []
        
        for i, prompt_responses in enumerate(all_responses):
            for response in prompt_responses:
                # Create a row for each model response
                row = {
                    "prompt_id": df.iloc[i].get("prompt_id", f"prompt_{i}"),
                    "prompt_type": df.iloc[i].get("prompt_type", ""),
                    "domain": df.iloc[i].get("domain", ""),
                    "prompt_text": response["prompt"],
                    "model": response["model"],
                    "success": response["success"],
                    "response_text": response["response_text"],
                    "elapsed_time": response.get("elapsed_time", 0)
                }
                
                # Add additional metadata if available
                if "tokens" in response:
                    row["prompt_tokens"] = response["tokens"].get("prompt", 0)
                    row["completion_tokens"] = response["tokens"].get("completion", 0)
                    row["total_tokens"] = response["tokens"].get("total", 0)
                
                if "error" in response:
                    row["error"] = response["error"]
                
                # Add dimension values if available in the dataset
                for col in df.columns:
                    if col.startswith("dimension_") and col not in row:
                        row[col] = df.iloc[i][col]
                
                results.append(row)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Saved {len(results_df)} model responses to {output_path}")
        
        return results_df
# Example usage
async def main():
    """
    Example of how to use the ModelManager to process a dataset.
    """
    # Just use Ollama models
    ollama_interface = OllamaInterface(model_name="deepseek-llm")
    
    # Create model manager and add models
    manager = ModelManager()
    manager.add_model(ollama_interface)
    
    # Process dataset
    results_df = await manager.process_dataset(
        dataset_path="bias_test_prompts.csv",
        output_path="model_responses(deepseek-llm).csv",
        prompt_column="prompt_text",
        max_concurrency=3, 
        temperature=0.2,
        max_tokens=1000
    )
    
    print(f"Processed {len(results_df)} responses")

if __name__ == "__main__":
    # Run the main function
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())