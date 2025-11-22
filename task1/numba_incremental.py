import numpy as np
import time
from numba import njit


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
        insert_pos = k - 1  # по умолчанию вставляем в конец
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


def generate_test_data(n: int, k: int) -> np.ndarray:
    """Генерация случайных данных"""
    return np.random.randint(-10000, 10000, n)

def main():
    n = int(input("Введите размер массива (n): "))
    k = int(input("Введите размер окна (k): "))

    data = generate_test_data(n, k)

    # Прогрев JIT-компилятора Numba (первый запуск компилирует функцию)
    _ = sliding_window_median_numba(np.array([1, 2, 3, 4, 5]), 3)

    start_time = time.perf_counter()
    result = sliding_window_median_numba(data, k)
    end_time = time.perf_counter()

    execution_time = end_time - start_time

    print(f"Время выполнения: {execution_time * 1000:.2f} ms")
    print(f"\nПервые 5 медиан: {result[:5]}")
    print(f"Последние 5 медиан: {result[-5:]}")


if __name__ == "__main__":
    main()