import asyncio
import openai
from typing import List, Dict, Any
from tqdm.asyncio import tqdm_asyncio # 用于异步任务的进度条

class LLMHandler:
    """
    负责与所有大语言模型API进行交互的专职外交官。
    它封装了客户端初始化、并发控制、API调用和错误处理。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化时接收配置，并设置好API客户端和并发控制器。
        """
        self.config = config
        self.api_key = config.get('api_key')
        self.base_url = config.get('base_url')
        self.model_name = config.get('model_name')

        if not self.api_key:
            raise ValueError("API密钥未在配置文件或secrets.yaml中设置。")

        # 初始化异步OpenAI客户端
        self.client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        # 从配置中获取并发数，并创建信号量
        max_workers = self.config.get('max_workers', 5)
        self.semaphore = asyncio.Semaphore(max_workers)

    async def _query_single(self, message: List[Dict[str, str]]) -> str:
        """
        对单条消息进行API调用，包含错误处理和并发控制。
        这部分逻辑来自于 step1.py 的 call_openai 函数。
        """
        data = {
            "model": self.model_name,
            "messages": message,
            "temperature": 0.0,
            # 您可以在这里添加更多API参数，如top_p等
        }

        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(**data)
                return resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"API调用时发生错误: {e}")
                print(f"出错的请求数据: {data}")
                return "__API_ERROR__"  # 返回一个明确的错误标识

    async def batch_query(self, messages: List[List[Dict[str, str]]]) -> List[str]:
        """
        接收一个消息列表，并发地执行所有API调用，并按原始顺序返回结果列表。
        这是对外的核心方法，它封装了整个并发执行的循环。

        Args:
            messages: 一个消息列表，其中每个元素都是一个符合OpenAI格式的message list。

        Returns:
            一个字符串列表，包含了按顺序对应的所有API响应。
        """
        if not messages:
            return []

        print(f"开始对 {len(messages)} 条消息进行批量并发查询...")

        # 创建所有需要执行的异步任务
        tasks = [self._query_single(msg) for msg in messages]

        # 使用tqdm_asyncio.gather来执行任务并显示进度条
        results = await tqdm_asyncio.gather(*tasks, desc=f"Querying {self.model_name}")

        print("所有查询已完成。")
        return results