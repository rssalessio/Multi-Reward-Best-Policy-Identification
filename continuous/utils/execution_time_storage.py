import time
import numpy as np
from typing import Dict, List, Tuple

class ExecutionTimeStorage:
    _instance = None
    execution_times: Dict[str, List[float]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExecutionTimeStorage, cls).__new__(cls)
        return cls._instance

    @classmethod
    def record_execution_time(cls, identifier: str, time_taken: float):
        if identifier not in cls.execution_times:
            cls.execution_times[identifier] = []
        cls.execution_times[identifier].append(time_taken)

    @classmethod
    def get_mean_execution_time(cls, identifier: str) -> float:
        times = cls.execution_times.get(identifier, [])
        if times:
            return np.mean(times)
        return 0
    
    @classmethod
    def get_all_mean_execution_time(cls) -> List[Tuple[str, float]]:
        return [[key, np.mean(val)]  for key, val in cls.execution_times.items()]


def execution_time(method):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        
        #Determine if this is a method of a class or a standalone function
        if args and hasattr(args[0], method.__name__):
            identifier = "-".join([args[0].__class__.__name__, method.__name__])
        else:
            identifier = method.__name__
        # Record execution time
        ExecutionTimeStorage.record_execution_time(
            identifier, end_time - start_time)
        
        return result
    return wrapper

