from typing import Dict, List, Any, Union, Optional, Callable, TypeVar
from pathlib import Path
import os
import json
from datetime import datetime
from tqdm import tqdm

T = TypeVar('T')  # 用于泛型类型标注

class StepResult:
    """
    统一的Step输出包装类，用于标准化Step输出并提供元数据
    """
    def __init__(
        self,
        data: T,
        step_name: str,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        execution_time: Optional[float] = None
    ):
        """
        初始化Step结果
        
        Args:
            data: Step处理后的实际数据
            step_name: 产生此结果的Step名称
            message: 处理结果的消息或描述
            metadata: 与处理相关的元数据
            execution_time: 处理耗时(秒)
        """
        self.data = data
        self.step_name = step_name
        self.message = message
        self.metadata = metadata or {}
        self.execution_time = execution_time
        self.timestamp = datetime.now().isoformat()
    
    @classmethod
    def create(cls, data: T, step_name: str, message: str = "处理完成", **kwargs) -> 'StepResult[T]':
        """
        创建结果的便捷方法
        
        Args:
            data: 处理后的数据
            step_name: Step名称
            message: 结果消息
            **kwargs: 传递给构造函数的其他参数
        """
        return cls(data, step_name, message, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将结果转换为字典形式
        """
        return {
            "step_name": self.step_name,
            "message": self.message,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            # 注意：data字段可能不是可序列化的，因此这里不包含
        }
    
    def get_data(self) -> T:
        """
        获取数据
        
        Returns:
            处理后的数据
        """
        return self.data
    
    def then(self, func: Callable[[T], Any]) -> 'StepResult':
        """
        允许链式处理结果
        
        Args:
            func: 对结果数据进行进一步处理的函数
            
        Returns:
            处理后的新结果
        """
        import time
        start_time = time.time()
        
        new_data = func(self.data)
        execution_time = time.time() - start_time
        
        return StepResult.create(
            data=new_data,
            step_name=self.step_name,
            message=f"{self.message} -> 链式处理",
            metadata={**self.metadata, "chain_source": self.step_name},
            execution_time=execution_time
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        time_info = f", 耗时: {self.execution_time:.2f}s" if self.execution_time else ""
        return f"StepResult({self.step_name}{time_info}, 消息: {self.message})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Step:
    """Pipeline步骤的类"""
    def __init__(self, name: str):
        """
        初始化Step
        
        Args:
            name: Step名称
        """
        self.name = name
    
    def process(
            self,
            data: Any,
            context: dict
        ) -> StepResult:
        """处理数据的抽象方法
        Args:
            data: 输入数据
            context: pipeline上下文
        Returns:
            StepResult: 包含处理结果的包装对象
        """
        raise NotImplementedError
        
    def _wrap_process(self, data: Any, context: dict) -> StepResult:
        """
        包装process方法，处理计时
        
        Args:
            data: 输入数据
            context: pipeline上下文
            
        Returns:
            StepResult: 处理结果
        """
        import time
        start_time = time.time()
        
        result = self.process(data, context)
        
        # 如果返回的不是StepResult，则包装为StepResult
        if not isinstance(result, StepResult):
            result = StepResult.create(
                data=result,
                step_name=self.name,
                execution_time=time.time() - start_time
            )
        elif result.execution_time is None:
            # 如果是StepResult但没有执行时间，添加执行时间
            result.execution_time = time.time() - start_time
            
        return result


# 示例：如何实现具体的Step子类
class ExampleStep(Step):
    """示例Step实现"""
    
    def __init__(self, name: str, multiply_factor: int = 2):
        super().__init__(name)
        self.multiply_factor = multiply_factor
    
    def process(self, data: Any, context: dict) -> StepResult:
        """处理数据示例：将数值乘以一个系数"""
        # 执行实际处理逻辑
        if isinstance(data, (int, float)):
            result = data * self.multiply_factor
            message = f"将输入值 {data} 乘以 {self.multiply_factor}"
        else:
            # 对于非数值类型，转换为字符串并重复
            result = str(data) * self.multiply_factor
            message = f"将输入值 '{data}' 重复 {self.multiply_factor} 次"
        
        # 返回结果
        return StepResult.create(
            data=result,
            step_name=self.name,
            message=message,
            metadata={"input_value": data, "factor": self.multiply_factor}
        )


# 使用示例
if __name__ == "__main__":
    # 创建并执行一个示例Step
    step = ExampleStep(name="数值处理", multiply_factor=3)
    
    # 处理数值数据
    result = step._wrap_process(10, {})
    
    # 输出结果
    print(result)
    print(f"处理后的数据: {result.get_data()}")
    
    # 链式处理
    chained_result = result.then(lambda x: x + 5)
    print(f"链式处理后的数据: {chained_result.get_data()}")
    
    # 处理非数值数据
    string_result = step._wrap_process("Hello", {})
    print(string_result)
    print(f"字符串处理结果: {string_result.get_data()}")