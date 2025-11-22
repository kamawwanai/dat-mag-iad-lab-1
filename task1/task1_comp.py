import numpy as np
import time
from heapq import heappush, heappop
from collections import defaultdict
from numba import njit
from typing import List, Tuple, Callable
import pandas as pd


# НАИВНЫЙ АЛГОРИТМ

def sliding_window_median_naive(nums, k):
    """
    Наивный алгоритм - сортируем каждое окно
    Сложность: O(n * k log k)
    """
    n = len(nums)
    if n == 0 or k == 0:
        return []
    
    result = []
    
    for i in range(n - k + 1):
        # Взяли срез окна и сортируем
        window = sorted(nums[i:i+k])
        
        # Находим медиану
        if k & 1:  # нечетное k
            median = float(window[k // 2])
        else:  # четное k
            median = (window[k // 2 - 1] + window[k // 2]) / 2.0
        
        result.append(median)
    
    return result

@njit
def sliding_window_median_naive_numba(nums, k):
    """
    Тот же наивный алгоритм с Numba
    """
    n = len(nums)
    if n == 0 or k == 0:
        return np.empty(0, dtype=np.float64)
    
    result = np.empty(n - k + 1, dtype=np.float64)
    
    for i in range(n - k + 1):
        # Срез окна и сортипровка
        window = nums[i:i+k].copy()
        window.sort()  # здесь будет оптимизация Numba
        
        # Находим медиану
        if k & 1:
            result[i] = float(window[k // 2])
        else:
            result[i] = (float(window[k // 2 - 1]) + float(window[k // 2])) / 2.0
    
    return result

# ДВЕ КУЧИ
class MedianFinder:
    """Две кучи с удалением"""
    def __init__(self):
        # максимальная куча (хранит меньшую половину элементов, вершина - максимум)
        self.max_heap = []
        # минимальная куча (хранит большую половину элементов, вершина - минимум)
        self.min_heap = []
        self.delayed = defaultdict(int) # словарь для удаления элементов
        self.max_heap_size = 0
        self.min_heap_size = 0

    def add_num(self, num):
        # элемент в max_heap, если он меньше или равен текущему максимуму левой половины
        if not self.max_heap or num <= -self.max_heap[0]:
            heappush(self.max_heap, -num)
            self.max_heap_size += 1
        # в min_heap (правая половина, большие элементы)
        else:
            heappush(self.min_heap, num)
            self.min_heap_size += 1
        # балансировка
        self._rebalance()

    def remove_num(self, num):
        self.delayed[num] += 1
        if num <= -self.max_heap[0]:
            self.max_heap_size -= 1
            # элемент на вершине max_heap, очищаем кучу
            if num == -self.max_heap[0]:
                self._prune(self.max_heap)
        else:
            self.min_heap_size -= 1
            # элемент на вершине min_heap, очищаем кучу
            if num == self.min_heap[0]:
                self._prune(self.min_heap)
        # балансировка
        self._rebalance()

    def _prune(self, heap):
        # очищаем вершину кучи от помеченных на удаление элементов
        sign = -1 if heap is self.max_heap else 1
        while heap and sign * heap[0] in self.delayed:
            val = sign * heap[0]
            self.delayed[val] -= 1
            # удаляем из словаря, если больше нет копий
            if self.delayed[val] == 0:
                del self.delayed[val]
            heappop(heap)

    def _rebalance(self):
        if self.max_heap_size > self.min_heap_size + 1: # если левая куча больше правой более чем на 1, переносим элемент
            heappush(self.min_heap, -heappop(self.max_heap))
            self.max_heap_size -= 1
            self.min_heap_size += 1
            self._prune(self.max_heap)
        elif self.max_heap_size < self.min_heap_size: # иначе
            heappush(self.max_heap, -heappop(self.min_heap))
            self.min_heap_size -= 1
            self.max_heap_size += 1
            self._prune(self.min_heap)

    def find_median(self, k):
        # для нечетного количества элементов медиана - вершина max_heap
        if k & 1:
            return float(-self.max_heap[0])
        # иначе среднее между двумя кучами
        return float((-self.max_heap[0] + self.min_heap[0]) / 2.0)

def sliding_window_median_heaps(nums, k):
    """Две кучи // Сложность: O(n log k)"""
    n = len(nums)
    if n == 0 or k == 0:
        return []
    
    finder = MedianFinder()
    result = []
    
    for i in range(k):
        finder.add_num(nums[i])
    result.append(finder.find_median(k))
    
    # при движении окна: добавляем новый элемент справа, удаляем старый слева
    for i in range(k, n):
        finder.add_num(nums[i])
        finder.remove_num(nums[i - k])
        result.append(finder.find_median(k))
    
    return result

# Numba incremental
@njit
def sliding_window_median_numba(nums, k):
    """Numba инкрементальная сортировка - O(nk)"""
    n = len(nums)
    if n == 0 or k == 0:
        return np.empty(0, dtype=np.float64)
    
    result = np.empty(n - k + 1, dtype=np.float64)
    # отсортированное окно из первых k элементов
    window = np.sort(nums[:k].copy())
    mid1, mid2 = (k - 1) // 2, k // 2
    
    # первое окно
    if k & 1:
        result[0] = float(window[mid2])
    else:
        result[0] = (float(window[mid1]) + float(window[mid2])) / 2.0
    
    # двигаем окно
    for i in range(k, n):
        old_val = nums[i - k]
        new_val = nums[i]
        
        # удаление (old_val в отсортированном окне)
        idx = -1
        for j in range(k):
            if window[j] == old_val:
                idx = j
                break
        # сдвигаем элементы влево, затирая удаленный
        if idx != -1:
            for j in range(idx, k - 1):
                window[j] = window[j + 1]
        
        # вставка (находим позицию для нового)
        insert_pos = k - 1 # по умолчанию вставляем в конец
        for j in range(k - 1):
            if window[j] > new_val:
                insert_pos = j
                break
        # сдвигаем элементы вправо, освобождая место для нового
        for j in range(k - 1, insert_pos, -1):
            window[j] = window[j - 1]
        window[insert_pos] = new_val
        
        # медиана обновленного окна
        if k & 1:
            result[i - k + 1] = float(window[mid2])
        else:
            result[i - k + 1] = (float(window[mid1]) + float(window[mid2])) / 2.0
    
    return result


# ТЕСТЫ И БЕНЧМАРКИ
class TestCase:
    def __init__(self, name: str, nums: List, k: int, expected: List):
        self.name = name
        self.nums = nums
        self.k = k
        self.expected = expected

def get_test_cases() -> List[TestCase]:
    return [
        TestCase("Example leetcode 1", [1, 3, -1, -3, 5, 3, 6, 7], 3, 
                 [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]),
        TestCase("Example leetcode 2", [1, 2, 3, 4, 2, 3, 1, 4, 2], 3, 
                 [2.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0]),
        TestCase("Even k", [1, 2, 3, 4, 5, 6], 2, 
                 [1.5, 2.5, 3.5, 4.5, 5.5]),
        TestCase("k=1", [5, 2, 8, 1], 1, 
                 [5.0, 2.0, 8.0, 1.0]),
        TestCase("k=n", [3, 1, 4, 1, 5], 5, 
                 [3.0]),
        TestCase("All same", [5, 5, 5, 5, 5], 3, 
                 [5.0, 5.0, 5.0]),
    ]

def generate_test_data(n: int, k: int, data_type: str = "random") -> np.ndarray:
    return np.random.randint(-10000, 10000, n)


def run_test_case(test_case: TestCase, algorithm_func, algorithm_name: str) -> Tuple[bool, str]:
    try:
        if "numba" in algorithm_name.lower():
            result = algorithm_func(np.array(test_case.nums), test_case.k)
        else:
            result = algorithm_func(test_case.nums, test_case.k)
        
        if np.allclose(result, test_case.expected, rtol=1e-9, atol=1e-9):
            return True, ""
        else:
            return False, f"Expected {test_case.expected}. Got {list(result)}"
    except Exception as e:
        return False, f"Exception: {str(e)}"
    
def run_correctness_tests(algorithms: dict) -> Tuple[dict, bool]:
    test_cases = get_test_cases()

    print("ПРОВЕРКА КОРРЕКТНОСТИ")
    
    for test_case in test_cases:
        print(f"Тест: {test_case.name}")
        for algo_name, algo_func in algorithms.items():
            passed, error_msg = run_test_case(test_case, algo_func, algo_name)
            
            if passed:
                print(f"[OK] {algo_name}")
            else:
                print(f"[FAILED] {algo_name}: {error_msg}")

def benchmark_single(func: Callable, nums: np.ndarray, k: int, 
                     warmup: int = 1, runs: int = 3) -> float:
    for _ in range(warmup):
        _ = func(nums, k)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = func(nums, k)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return np.mean(times)

def get_extended_benchmark_configs():
    """БЕНЧМАРКИ ДЛЯ СРАВНЕНИЯ"""
    return {
        "Маленькие массивы, разные k": [
            (500, 5, "n=500, k=5"),
            (500, 10, "n=500, k=10"),
            (500, 50, "n=500, k=50"),
            (500, 100, "n=500, k=100"),
        ],
        
        "Средние массивы, маленькие k": [
            (1000, 1, "n=1k, k=1"),
            (1000, 5, "n=1k, k=5"),
            (1000, 10, "n=1k, k=10"),
            (1000, 25, "n=1k, k=25"),
            (1000, 50, "n=1k, k=50"),
            (1000, 100, "n=1k, k=100"),
        ],
        
        "Средние массивы, большие k": [
            (5000, 10, "n=5k, k=10"),
            (5000, 50, "n=5k, k=50"),
            (5000, 100, "n=5k, k=100"),
            (5000, 250, "n=5k, k=250"),
            (5000, 500, "n=5k, k=500"),
        ],
        
        "Большие массивы": [
            (10000, 1, "n=10k, k=1"),
            (10000, 10, "n=10k, k=10"),
            (10000, 50, "n=10k, k=50"),
            (10000, 100, "n=10k, k=100"),
            (10000, 250, "n=10k, k=250"),
            (10000, 500, "n=10k, k=500"),
            (10000, 1000, "n=10k, k=1k"),
        ],
        
        "Очень большие массивы": [
            (25000, 50, "n=25k, k=50"),
            (25000, 100, "n=25k, k=100"),
            (25000, 500, "n=25k, k=500"),
            (25000, 1000, "n=25k, k=1k"),
        ],
        
        "Экстремальные случаи": [
            (50000, 10, "n=50k, k=10"),
            (50000, 100, "n=50k, k=100"),
            (50000, 1000, "n=50k, k=1k"),
            (10000, 5000, "n=10k, k=5k"),
        ]
    }

def benchmark_suite(config_groups: dict, 
                    algorithms: dict,
                    data_type: str = "random") -> pd.DataFrame:
    print("\n")
    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")

    all_results = []
    
    for group_name, configs in config_groups.items():
        print(f"\n")
        print(f"Группа: {group_name}")
        
        for n, k, desc in configs:
            print(f"\n{desc}:")
            nums = generate_test_data(n, k, data_type)
            
            for name, func in algorithms.items():
                try:
                    avg_time = benchmark_single(func, nums, k, warmup=1, runs=3)
                    all_results.append({
                        "Group": group_name,
                        "Configuration": desc,
                        "n": n,
                        "k": k,
                        "Algorithm": name,
                        "Time (ms)": avg_time
                    })
                    print(f"  {name:20s}: {avg_time:8.2f} ms")
                except Exception as e:
                    print(f"  {name:20s}: ERROR - {str(e)}")
                    all_results.append({
                        "Group": group_name,
                        "Configuration": desc,
                        "n": n,
                        "k": k,
                        "Algorithm": name,
                        "Time (ms)": float('inf')
                    })
    
    df = pd.DataFrame(all_results)

    return df

def main():
    algorithms = {
        "Naive Python": sliding_window_median_naive,
        "Naive Numba": sliding_window_median_naive_numba,
        "Two Heaps": sliding_window_median_heaps,
        "Numba Incremental": sliding_window_median_numba,
    }
    
    # Тесты корректности
    run_correctness_tests(algorithms)

    
    # Бенчмарки
    config_groups = get_extended_benchmark_configs()
    df_results = benchmark_suite(config_groups, algorithms, data_type="random")

    df_results.to_csv("task1_results.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    main()