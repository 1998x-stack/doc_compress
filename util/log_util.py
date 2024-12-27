# coding=utf-8
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import time, sys, traceback, threading
from  pprint import PrettyPrinter
pprint = PrettyPrinter(indent=4, width=80, sort_dicts=False).pprint

class Log:
    """
    日志系统
    用于记录系统各种信息与异常
    """
    def __init__(self, dir_name='', additional_info: str = ''):
        """
        初始化函数
        检查是否存在指定的日志目录，如果没有则创建
        """
        # 使用绝对路径来定位日志文件夹
        if dir_name:
            self.dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../logs', dir_name)
        else:
            self.dir_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../logs')
        self.additional_info = additional_info
        try:
            # 检测是否存在日志目录
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)
        except Exception as e:
            print(f"创建日志目录 {self.dir_name} 发生错误: {e}")

    def log_info(self, message: str, print_screen: bool = True):
        """
        记录一般信息
        :param message: 要记录的信息
        :param print_screen: 是否在屏幕上也显示该信息
        """
        # 获取当前时间，并格式化
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        message = f"{current_time} {self.additional_info}  === {message}"

        # 打印到屏幕
        if print_screen:
            pprint(message)

        message += '\n'
        file_name = os.path.join(self.dir_name, time.strftime("%Y-%m-%d", time.localtime()) + '.log')
        try:
            # 锁定资源，防止多线程问题
            lock = threading.Lock()
            with lock:
                with open(file_name, "a") as fa:
                    fa.write(message)
        except Exception as e:
            pprint(f"写入日志信息 {message} 发生错误: {e}")

    def log_exception(self, print_screen: bool = True):
        """
        记录异常信息
        :param print_screen: 是否在屏幕上也显示该信息
        """
        # 获取异常信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_message = repr(traceback.format_exception(exc_type, exc_value, exc_traceback))

        # 记录异常信息
        self.log_info(f"Exception: {error_message}", print_screen)
        
logger = Log()