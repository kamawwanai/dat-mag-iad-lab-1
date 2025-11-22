import time
import numpy as np
from numba import njit, prange

# ВАРИАНТ 1
@njit(parallel=True)
def match_timestamps_searchsorted(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    бинарный поиск с параллельностью через numba.prange
    Complexity: O(n log m)
    """
    n = len(timestamps1)
    m = len(timestamps2)
    matching = np.empty(n, dtype=np.uint32)
    
    for i in prange(n):  #  параллельный цикл
        target = timestamps1[i]
        
        # бинарный поиск
        left, right = 0, m
        while left < right:
            mid = (left + right) // 2
            if timestamps2[mid] < target:
                left = mid + 1
            else:
                right = mid
        
        # ближайший сосед
        idx = left
        if idx == 0:
            matching[i] = 0
        elif idx == m:
            matching[i] = m - 1
        else:
            left_diff = abs(target - timestamps2[idx - 1])
            right_diff = abs(target - timestamps2[idx])
            matching[i] = idx - 1 if left_diff <= right_diff else idx
    
    return matching


# ВАРИАНТ 2 (ЧЕРЕЗ УКАЗАТЕЛЬ)
@njit
def match_timestamps(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    Example:
        timestamps1 = [0, 0.091, 0.5]
        timestamps2 = [0.001, 0.09, 0.12, 0.6]
        => matching = [0, 1, 3]
    """
    n = len(timestamps1)
    m = len(timestamps2)
    matching = np.empty(n, dtype=np.uint32)

    j = 0  # указатель timestamps2

    for i in range(n):
        target = timestamps1[i]

        # двигаем указатель до ближайшего
        while j < m - 1:
            curr_diff = abs(timestamps2[j] - target)
            next_diff = abs(timestamps2[j + 1] - target)
            if next_diff < curr_diff:
                j += 1
            else:
                break

        matching[i] = j

    return matching


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps

def perf_measurement():
    """
    Performance measurement procedure
    """
    st_ts = time.time()
    fn_ts = st_ts + 3600 * 2
    fps = 30
    ts1 = make_timestamps(fps, st_ts, fn_ts)
    ts2 = make_timestamps(fps, st_ts + 200, fn_ts)
    # warmup
    for _ in range(10):
        match_timestamps(ts1, ts2)
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        match_timestamps(ts1, ts2)
    print(f"Perf time: {(time.perf_counter() - t0) / n_iter} seconds")


def run_benchmarks():
    """
    Запускает бенчмарки на разных размерах данных
    """
    print("Сравнение разных бенчмарок")
    print()

    # Определяем различные сценарии
    scenarios = [
        # (название, длительность_часы, fps_cam1, fps_cam2, n_iter)
        ("(10 min, 30fps)", 10/60, 30, 30, 50),
        ("(1 hour, 30fps)", 1, 30, 30, 30),
        ("(2 hours, 30fps)", 2, 30, 30, 20),
        ("(3 hours, 30fps)", 3, 30, 30, 10),
        ("(2 hours, 60fps)", 2, 60, 60, 20),
        ("(2 hours, 30+60fps)", 2, 30, 60, 20),
    ]

    for scenario_name, duration_hours, fps1, fps2, n_iter in scenarios:
        print(f"\n{scenario_name}")
        
        st_ts = time.time()
        fn_ts = st_ts + 3600 * duration_hours
        ts1 = make_timestamps(fps1, st_ts, fn_ts)
        ts2 = make_timestamps(fps2, st_ts + 200, fn_ts)
        
        print(f"  Dataset: {len(ts1)} frames (cam1), {len(ts2)} frames (cam2)")
        
        # прогрев для обеих функций
        match_timestamps(ts1, ts2)
        match_timestamps_searchsorted(ts1, ts2)
        
        # бенчмарк match_timestamps_searchsorted
        t0 = time.perf_counter()
        for _ in range(n_iter):
            match_timestamps_searchsorted(ts1, ts2)
        time_searchsorted = (time.perf_counter() - t0) / n_iter
        
        # бенчмарк match_timestamps
        t0 = time.perf_counter()
        for _ in range(n_iter):
            match_timestamps(ts1, ts2)
        time_pointer = (time.perf_counter() - t0) / n_iter
        
        print(f"  match_timestamps_searchsorted: {time_searchsorted:.6f} seconds")
        print(f"  match_timestamps: {time_pointer:.6f} seconds")


def main():
    """
    Setup:
        Say we have two videocameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.

    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.

    Estimation criteria:
        - The solution has to be optimal algorithmically. As an example, let's assume that
    the best solution has O(n^3) complexity. In this case, the O(n^3 * logn) solution will add -1 point penalty,
    O(n^4) will add -2 points and so on.
        - The solution has to be optimal python-wise.
    If it can be optimized ~x5 times by rewriting the algorithm in Python,
    this will add -1 point. x20 times optimization will result in -2 points.
    You may use any optimization library!
        - All corner cases must be handled correctly. A wrong solution will add -3 points.
        - The base score is 6.
        - Parallel implementation adds +1 point, provided it is effective (cannot be optimized x5 times)
        - 3 points for this homework are added by completing the second problem (the one with the medians).
    Optimize the solution to work with ~2-3 hours of data.
    Good luck!
    """
    # generate timestamps for the first camera
    timestamps1 = make_timestamps(30, time.time() - 100, time.time() + 3600 * 2)
    # generate timestamps for the second camera
    timestamps2 = make_timestamps(60, time.time() + 200, time.time() + 3600 * 2.5)

    t_match = time.perf_counter()
    matching = match_timestamps(timestamps1, timestamps2)
    elapsed = time.perf_counter() - t_match

    print(f"Matching time:{elapsed*1000:.2f} ms")

    perf_measurement()

    print()

    run_benchmarks()

if __name__ == '__main__':
    main()