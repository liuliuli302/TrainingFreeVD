import os
import cv2
import numpy as np
from typing import List, Tuple, Union, Optional
from PIL import Image
import logging

class VideoProcessor:
    """
    视频和图像处理工具类，用于视频帧提取和加载图像/视频到内存
    """
    
    def __init__(self, logging_level: int = logging.INFO):
        """
        初始化VideoProcessor
        
        参数:
            logging_level: 日志级别，默认为INFO
        """
        self.logger = self._setup_logger(logging_level)
    
    def _setup_logger(self, level: int) -> logging.Logger:
        """设置日志器"""
        logger = logging.getLogger("VideoProcessor")
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: Optional[str] = None,
        fps: Optional[float] = None, 
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
        frame_format: str = 'jpg',
        quality: int = 95,
        return_frames: bool = False
    ) -> Optional[List[np.ndarray]]:
        """
        从视频中提取帧
        
        参数:
            video_path: 视频文件路径
            output_dir: 输出目录，如果不提供则不保存到磁盘
            fps: 提取的帧率，如果为None则使用原视频帧率
            max_frames: 最大帧数，如果为None则提取所有帧
            resize: 调整大小为(宽, 高)，如果为None则不调整大小
            frame_format: 帧保存格式，'jpg'或'png'
            quality: 图像质量(1-100)，仅适用于jpg格式
            return_frames: 是否返回提取的帧，如果为True则返回帧列表
            
        返回:
            如果return_frames为True，返回提取的帧列表；否则返回None
        """
        if not os.path.exists(video_path):
            self.logger.error(f"视频文件不存在: {video_path}")
            return None
            
        # 创建输出目录
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"无法打开视频: {video_path}")
            return None
            
        # 获取视频属性
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / original_fps
        
        # 确定提取帧的间隔
        target_fps = fps if fps is not None else original_fps
        frame_interval = max(1, int(original_fps / target_fps))
        
        extracted_frames = []
        frame_idx = 0
        saved_count = 0
        
        self.logger.info(f"开始处理视频: {os.path.basename(video_path)}")
        self.logger.info(f"视频信息: {frame_count}帧, {original_fps:.2f}fps, 时长: {duration:.2f}秒")
        self.logger.info(f"目标提取帧率: {target_fps:.2f}fps, 帧间隔: {frame_interval}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 按照指定帧率提取
            if frame_idx % frame_interval == 0:
                # 调整大小
                if resize:
                    frame = cv2.resize(frame, resize)
                
                # 保存到磁盘
                if output_dir:
                    frame_filename = f"frame_{saved_count:06d}.{frame_format}"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    if frame_format.lower() == 'jpg':
                        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    else:
                        cv2.imwrite(frame_path, frame)
                
                # 添加到返回列表
                if return_frames:
                    # OpenCV使用BGR，转换为RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    extracted_frames.append(rgb_frame)
                
                saved_count += 1
                
                # 检查是否达到最大帧数
                if max_frames and saved_count >= max_frames:
                    self.logger.info(f"已达到最大帧数 {max_frames}")
                    break
            
            frame_idx += 1
            
            # 输出进度
            if frame_idx % 100 == 0:
                progress = (frame_idx / frame_count) * 100
                self.logger.debug(f"处理进度: {progress:.1f}% ({frame_idx}/{frame_count})")
        
        cap.release()
        self.logger.info(f"视频处理完成，提取了 {saved_count} 帧")
        
        if return_frames:
            return extracted_frames
        return None
    
    def load_video(
        self, 
        video_path: str, 
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None
    ) -> Tuple[List[np.ndarray], float]:
        """
        将视频加载到内存
        
        参数:
            video_path: 视频文件路径
            fps: 提取的帧率，如果为None则使用原视频帧率
            max_frames: 最大帧数，如果为None则加载所有帧
            resize: 调整大小为(宽, 高)，如果为None则不调整大小
            
        返回:
            (帧列表, 原始帧率)的元组
        """
        frames = self.extract_frames(
            video_path=video_path,
            fps=fps,
            max_frames=max_frames,
            resize=resize,
            return_frames=True
        )
        
        if frames is None:
            return [], 0.0
        
        # 获取原始帧率
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return frames, original_fps
    
    def load_image(
        self, 
        image_path: str, 
        resize: Optional[Tuple[int, int]] = None,
        return_format: str = 'numpy'
    ) -> Union[np.ndarray, Image.Image, None]:
        """
        加载图像到内存
        
        参数:
            image_path: 图像文件路径
            resize: 调整大小为(宽, 高)，如果为None则不调整大小
            return_format: 返回格式，'numpy'为numpy数组，'pil'为PIL图像
            
        返回:
            根据return_format返回相应格式的图像，加载失败返回None
        """
        if not os.path.exists(image_path):
            self.logger.error(f"图像文件不存在: {image_path}")
            return None
        
        try:
            if return_format.lower() == 'pil':
                # 使用PIL加载
                image = Image.open(image_path)
                
                # 调整大小
                if resize:
                    image = image.resize(resize)
                
                return image
            else:
                # 使用OpenCV加载
                image = cv2.imread(image_path)
                
                if image is None:
                    self.logger.error(f"无法加载图像: {image_path}")
                    return None
                
                # 调整大小
                if resize:
                    image = cv2.resize(image, resize)
                
                # 转换为RGB (OpenCV加载的是BGR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                return image
                
        except Exception as e:
            self.logger.error(f"加载图像时出错: {str(e)}")
            return None
    
    def load_images_from_dir(
        self, 
        directory: str, 
        extensions: List[str] = ['jpg', 'jpeg', 'png'],
        resize: Optional[Tuple[int, int]] = None,
        max_images: Optional[int] = None,
        return_format: str = 'numpy'
    ) -> List[Union[np.ndarray, Image.Image]]:
        """
        从目录中加载多个图像
        
        参数:
            directory: 图像目录
            extensions: 要加载的文件扩展名列表
            resize: 调整大小为(宽, 高)，如果为None则不调整大小
            max_images: 最大加载图像数，如果为None则加载所有图像
            return_format: 返回格式，'numpy'为numpy数组，'pil'为PIL图像
            
        返回:
            加载的图像列表
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            self.logger.error(f"目录不存在: {directory}")
            return []
        
        images = []
        extensions = [ext.lower() if not ext.startswith('.') else ext[1:].lower() for ext in extensions]
        
        # 获取所有图像文件
        image_files = []
        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1][1:].lower()
            if ext in extensions:
                image_files.append(os.path.join(directory, filename))
        
        # 排序以确保顺序一致
        image_files.sort()
        
        # 限制数量
        if max_images:
            image_files = image_files[:max_images]
        
        self.logger.info(f"开始加载 {len(image_files)} 张图像...")
        
        # 加载图像
        for i, image_path in enumerate(image_files):
            image = self.load_image(image_path, resize, return_format)
            if image is not None:
                images.append(image)
            
            # 输出进度
            if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
                self.logger.debug(f"已加载 {i+1}/{len(image_files)} 张图像")
        
        self.logger.info(f"成功加载了 {len(images)}/{len(image_files)} 张图像")
        return images


# 使用示例
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # 示例1: 提取视频帧并保存到磁盘
    # processor.extract_frames(
    #     video_path="path/to/video.mp4",
    #     output_dir="output/frames",
    #     fps=5,  # 每秒提取5帧
    #     resize=(640, 360)  # 调整为640x360
    # )
    
    # 示例2: 加载视频到内存
    # frames, fps = processor.load_video(
    #     video_path="path/to/video.mp4",
    #     fps=10,  # 每秒提取10帧
    #     max_frames=100,  # 最多提取100帧
    #     resize=(320, 240)  # 调整为320x240
    # )
    # print(f"加载了 {len(frames)} 帧视频, 原始帧率: {fps}")
    
    # 示例3: 加载单张图像
    # image = processor.load_image(
    #     image_path="path/to/image.jpg",
    #     resize=(800, 600)
    # )
    # if image is not None:
    #     print(f"图像形状: {image.shape}")
    
    # 示例4: 加载目录中的所有图像
    # images = processor.load_images_from_dir(
    #     directory="path/to/images",
    #     resize=(640, 480),
    #     max_images=50
    # )
    # print(f"加载了 {len(images)} 张图像")