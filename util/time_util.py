import sys,os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import time
from functools import wraps
from threading import Lock
from datetime import datetime

from util.log_util import logger


def print_func_time(function):
    """
    计算程序运行时间
    :param function:
    :return:
    """
    def f(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        spend = t1 - t0
        logger.log_info("运行耗时%.3f 毫秒：函数%s" % (spend*1000, function.__name__))
        return result
    return f


# 全局字典用于记录函数使用信息
class FuncUsage:
    def __init__(self):
        self.usage_stats = {} # 初始化字典, 用于记录函数使用信息
        self.lock = Lock()  # 初始化锁

    def update_usage(
            self, 
            func_key, 
            execution_time
        ):
        with self.lock:
            usage = self.usage_stats.get(
                func_key, 
                {
                    'count': 0, 
                    'last_used': None, 
                    'total_time': 0, 
                    'mean_time': 0, 
                    'max_time': None,
                }
            )
            usage['count'] += 1
            usage['last_used'] = str(datetime.now())
            usage['total_time'] += execution_time
            usage['mean_time'] = usage['total_time'] / usage['count']
            usage['max_time'] = max(usage['max_time'], execution_time) if usage['max_time'] else execution_time
            self.usage_stats[func_key] = usage

func_usage_tracker = FuncUsage()

def calculate_execution_time(func_id: str='default'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            # 计算执行时间并记录
            execution_time = (end_time - start_time) * 1000  # 将秒转换为毫秒
            func_key = f'【{func_id}】{func.__qualname__}'
            if logger:
                print_args = [str(arg)[:30] for arg in args]
                print_kwargs = {k: str(v)[:30] for k, v in kwargs.items()}
                logger.log_info(f"[=== 《当前运行时间》 {func_key}: {round(execution_time, 4)} milliseconds, [ --- args & kwargs --- ] with args: {print_args}, kwargs: {print_kwargs} ===]")

            # 更新函数使用次数和最后使用时间
            func_usage_tracker.update_usage(func.__name__, execution_time)
            if logger:
                usage = func_usage_tracker.usage_stats[func.__name__]
                logger.log_info(f"[=== 《统计运行时间》 {func_key}: MEAN TIME: {round(usage['mean_time'], 4)} milliseconds, MAX TIME: {round(usage['max_time'], 4)} milliseconds, TOTAL TIME: {round(usage['total_time'], 4)} milliseconds, USAGE COUNTS: {usage['count']} times ===]")
            return result
        return wrapper
    return decorator