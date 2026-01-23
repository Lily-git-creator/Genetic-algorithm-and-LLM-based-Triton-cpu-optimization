import time
import functools
from collections import defaultdict

class GlobalProfiler:
    """
    全局单例性能分析器。
    使用字典存储每种操作的总耗时和调用次数。
    """
    def __init__(self):
        self.stats = defaultdict(lambda: {"total_time": 0.0, "count": 0})
        self.start_time = time.time() # 记录程序启动时间

    def record(self, tag, elapsed):
        self.stats[tag]["total_time"] += elapsed
        self.stats[tag]["count"] += 1

    def print_summary(self):
        total_program_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print(f"📊 Performance Profiling Report (Total Runtime: {total_program_time:.2f}s)")
        print("="*60)
        print(f"{'Task Name':<20} | {'Calls':<6} | {'Total(s)':<10} | {'Avg(s)':<8} | {'% of Total':<8}")
        print("-" * 60)
        
        # 按总耗时降序排列
        sorted_stats = sorted(self.stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for tag, data in sorted_stats:
            avg_time = data['total_time'] / data['count'] if data['count'] > 0 else 0
            pct = (data['total_time'] / total_program_time) * 100
            print(f"{tag:<20} | {data['count']:<6} | {data['total_time']:<10.2f} | {avg_time:<8.2f} | {pct:<6.1f}%")
        
        print("="*60 + "\n")

# 全局单例对象
_profiler_instance = GlobalProfiler()

def profile(tag):
    """
    装饰器：用于测量函数执行时间。
    用法：
    @profile("llm_api")
    def my_func(): ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                _profiler_instance.record(tag, elapsed)
        return wrapper
    return decorator

def print_stats():
    _profiler_instance.print_summary()
