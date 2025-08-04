import asyncio
import openai
from tqdm.asyncio import tqdm_asyncio # Progress bar for async tasks
from typing import List, Dict, Any


class LLMHandler:
    """
    Dedicated diplomat responsible for interacting with all large language model APIs.
    It encapsulates client initialization, concurrency control, API calls, and error handling.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration and set up API client and concurrency controller.
        """
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')

        if not self.api_key:
            raise ValueError("API key not set in configuration file or secrets.yaml.")

        # Initialize async OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # Get concurrency count from configuration and create semaphore
        max_workers = self.config.get('max_workers', 5)
        self.semaphore = asyncio.Semaphore(max_workers)

    async def _query_single(self, message: List[Dict[str, str]]) -> str:
        """
        Make API call for a single message, including error handling and concurrency control.
        This logic comes from the call_openai function in step1.py.
        """
        data = {
            "model": self.model_name,
            "messages": message,
            "temperature": 0.0
        }

        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(**data)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error occurred during API call: {e}")
                print(f"Request data that caused error: {data}")
                return "__API_ERROR__"  # Return a clear error identifier

    async def batch_query(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        Receive a list of messages, execute all API calls concurrently, and return results in original order.
        This is the core external method that encapsulates the entire concurrent execution loop.

        Args:
            messages: A list of messages, where each element is a message list conforming to OpenAI format.

        Returns:
            A list of strings containing all API responses in corresponding order.
        """
        if not messages:
            return []

        print(f"Starting batch concurrent query for {len(messages)} messages...")

        # Create all async tasks that need to be executed
        tasks = [self._query_single(msg) for msg in messages]

        # Use tqdm_asyncio.gather to execute tasks and display progress bar
        results = await tqdm_asyncio.gather(*tasks, desc=f"Querying {self.model_name}")

        print("All queries completed.")
        return results