import numpy as np
import time
from heapq import heappush, heappop
from collections import defaultdict

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

def generate_test_data(n: int, k: int) -> np.ndarray:
    """Генерация случайных данных"""
    return np.random.randint(-10000, 10000, n)

def main():
    n = int(input("Введите размер массива (n): "))
    k = int(input("Введите размер окна (k): "))

    data = generate_test_data(n, k)

    start_time = time.perf_counter()
    result = sliding_window_median_heaps(data, k)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print(f"Время выполнения: {execution_time * 1000:.2f} ms")
    print(f"\nПервые 5 медиан: {result[:5]}")
    print(f"Последние 5 медиан: {result[-5:]}")


if __name__ == "__main__":
    main()