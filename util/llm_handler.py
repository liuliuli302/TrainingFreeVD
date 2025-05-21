from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

class LLMHandlerConfig:
    def __init__(
        self,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        request_timeout: int = 60,
        **kwargs
    ):
        """
        LLM处理器的基础配置
        
        参数:
            model_name: 模型名称
            max_tokens: 生成的最大token数
            temperature: 温度参数，控制生成的随机性
            request_timeout: 请求超时时间（秒）
            **kwargs: 额外的模型特定参数
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.extra_params = kwargs
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置参数，如果不存在返回默认值"""
        if key in self.__dict__:
            return self.__dict__[key]
        return self.extra_params.get(key, default)


class LLMHandler(ABC):
    """
    LLM处理器基类，定义统一接口
    可被扩展以支持不同类型的LLM实现
    """
    def __init__(self, config: LLMHandlerConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """初始化LLM客户端，子类必须实现"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        """
        生成文本的统一接口，子类必须实现
        
        参数:
            prompt: 用户输入提示
            system_message: 系统提示（如果模型支持）
            **kwargs: 额外的生成参数
            
        返回:
            生成的文本
        """
        pass
    
    def format_prompt(self, prompt: str, system_message: Optional[str] = None) -> Union[str, List[Dict[str, str]]]:
        """
        根据模型格式化提示，子类可重写此方法
        
        参数:
            prompt: 用户输入提示
            system_message: 系统提示
            
        返回:
            格式化后的提示（字符串或消息列表）
        """
        if system_message:
            return f"{system_message}\n\n{prompt}"
        return prompt
    
    def cleanup(self) -> None:
        """释放资源，子类可重写此方法"""
        self.client = None


class APILLMHandler(LLMHandler):
    """API型LLM处理器基类"""
    
    def __init__(self, config: LLMHandlerConfig):
        """
        初始化API型LLM处理器
        
        参数:
            config: 配置对象，需要包含api_key
        """
        if not config.get("api_key"):
            raise ValueError(f"{self.__class__.__name__}需要提供api_key")
        super().__init__(config)


class LocalLLMHandler(LLMHandler):
    """本地LLM处理器基类"""
    
    def __init__(self, config: LLMHandlerConfig):
        """
        初始化本地LLM处理器
        
        参数:
            config: 配置对象，需要包含model_path
        """
        if not config.get("model_path"):
            raise ValueError(f"{self.__class__.__name__}需要提供model_path")
        super().__init__(config)


# 示例实现子类
class OpenAIHandler(APILLMHandler):
    """OpenAI API实现"""
    
    def _initialize_client(self) -> None:
        try:
            from openai import OpenAI
            import os
            
            api_key = self.config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            api_base = self.config.get("api_base") or os.environ.get("OPENAI_API_BASE")
            
            client_kwargs = {"api_key": api_key}
            if api_base:
                client_kwargs["base_url"] = api_base
            
            self.client = OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("请安装OpenAI客户端: pip install openai")
    
    def format_prompt(self, prompt: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        messages = self.format_prompt(prompt, system_message)
        
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            timeout=self.config.request_timeout
        )
        return response.choices[0].message.content


class LlamaHandler(LocalLLMHandler):
    """Llama本地模型实现"""
    
    def _initialize_client(self) -> None:
        try:
            from llama_cpp import Llama
            
            self.client = Llama(
                model_path=self.config.get("model_path"),
                n_ctx=self.config.get("context_length", 4096),
                n_threads=self.config.get("threads", 4)
            )
        except ImportError:
            raise ImportError("请安装llama-cpp-python: pip install llama-cpp-python")
    
    def format_prompt(self, prompt: str, system_message: Optional[str] = None) -> str:
        if system_message:
            return f"<|system|>\n{system_message}\n\n<|user|>\n{prompt}\n\n<|assistant|>\n"
        return f"<|user|>\n{prompt}\n\n<|assistant|>\n"
    
    def generate(self, prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
        formatted_prompt = self.format_prompt(prompt, system_message)
        
        response = self.client.create_completion(
            prompt=formatted_prompt,
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            stop=["<|user|>", "</s>"]
        )
        return response["choices"][0]["text"]


# 工厂函数，简化LLM处理器创建
def create_llm_handler(handler_type: str, **config_params) -> LLMHandler:
    """
    创建LLM处理器实例
    
    参数:
        handler_type: 处理器类型，如 'openai', 'anthropic', 'llama', 等
        **config_params: 配置参数
        
    返回:
        LLMHandler实例
    """
    config = LLMHandlerConfig(**config_params)
    
    if handler_type.lower() == "openai":
        return OpenAIHandler(config)
    elif handler_type.lower() == "llama":
        return LlamaHandler(config)
    else:
        raise ValueError(f"不支持的处理器类型: {handler_type}")


# 使用示例
if __name__ == "__main__":
    # 使用OpenAI
    openai_handler = create_llm_handler(
        "openai",
        model_name="gpt-3.5-turbo",
        api_key="your-api-key-here",
    )
    
    response = openai_handler.generate(
        prompt="写一个Python函数，计算斐波那契数列",
        system_message="你是一个专业的Python程序员"
    )
    print(f"OpenAI Response: {response}")
    
    # 使用本地Llama模型
    # llama_handler = create_llm_handler(
    #     "llama",
    #     model_name="llama3",
    #     model_path="/path/to/model.gguf",
    #     threads=8
    # )
    # 
    # response = llama_handler.generate(
    #     prompt="写一个Python函数，计算斐波那契数列",
    #     system_message="你是一个专业的Python程序员"
    # )
    # print(f"Llama Response: {response}")
    
    # 扩展实现自定义Handler
    # class MyCustomLLMHandler(LLMHandler):
    #     def _initialize_client(self):
    #         # 自定义初始化
    #         pass
    #     
    #     def generate(self, prompt, system_message=None, **kwargs):
    #         # 自定义生成逻辑
    #         pass









