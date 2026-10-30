import math

import numpy as np
from scipy.spatial import KDTree


def random_directions(n, rng):
    #n изотропных единичных направлений; результат имеет форму (n, 3)
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n должно быть целым неотрицательным числом")
    u = rng.random((n, 2))
    z = 2 * u[:, 0] - 1
    phi = 2 * np.pi * u[:, 1]
    rho = np.sqrt(np.maximum(0.0, 1 - z**2))
    return np.column_stack((rho * np.cos(phi), rho * np.sin(phi), z))


def random_direction(rng):
    #Одно направление: скалярный вариант для сравнения времени.
    u, v = rng.random(2)
    z = 2 * u - 1
    phi = 2 * math.pi * v
    rho = math.sqrt(max(0.0, 1 - z * z))
    return np.array((rho * math.cos(phi), rho * math.sin(phi), z))


class Source:
    def __init__(self, Y=10_000.0, position=(0.0, 0.0, 0.0)):
        if not np.isfinite(Y) or Y < 0:
            raise ValueError("Y должно быть конечным и неотрицательным")
        self.Y = float(Y)
        self.position = np.array(position, dtype=float, copy=True)
        if self.position.shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("position должна содержать три конечные координаты")

    def photon_count(self, E, rng):
        #Число фотонов одного события
        if not np.isfinite(E) or E < 0:
            raise ValueError("E должно быть конечным и неотрицательным")
        return int(rng.poisson(E * self.Y))

    def emit(self, E, rng):
        #Сразу весь массив направлений одного события
        n = self.photon_count(E, rng)
        return random_directions(n, rng)

    def emit_generator(self, E, rng):
        #Генератор: по одному направлению, без хранения всего события
        n = self.photon_count(E, rng)
        for _ in range(n):
            yield random_direction(rng)

    def emit_batches(self, E, rng, batch_size=100):
        #Генератор массивов до batch_size фотонов. Последний пакет короче
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size должно быть целым положительным")
        n = self.photon_count(E, rng)
        for start in range(0, n, batch_size):
            size = min(batch_size, n - start)
            yield random_directions(size, rng)


def sphere_intersections(position, directions, R):
    #Точки выхода на сферу и длины пути для единичных направлений
    x = np.asarray(position, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError("position должна иметь форму (3,)")
    if not np.isfinite(R) or R <= 0 or np.dot(x, x) >= R**2:
        raise ValueError("R > 0; источник должен быть строго внутри сферы")
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError("directions должна иметь форму (n, 3)")
    if not np.isfinite(directions).all() or not np.allclose(
        np.sum(directions**2, axis=1), 1.0, rtol=0.0, atol=1e-10
    ):
        raise ValueError("Направления должны быть конечными единичными векторами")
    b = directions @ x
    distance = -b + np.sqrt(b**2 + R**2 - np.dot(x, x))
    points = x + distance[:, None] * directions
    return points, distance


def sphere_intersections_loop(position, directions, R):
    #Скалярная версия формулы, для корректных входных данных
    x, y, z = position
    points = np.empty((len(directions), 3), dtype=float)
    distance = np.empty(len(directions), dtype=float)
    c = R * R - x * x - y * y - z * z
    for i, (nx, ny, nz) in enumerate(directions):
        b = x * nx + y * ny + z * nz
        d = -b + math.sqrt(b * b + c)
        distance[i] = d
        points[i] = (x + d * nx, y + d * ny, z + d * nz)
    return points, distance


def survival_probability(distance, attenuation_length=np.inf):
    #Вероятность дойти до стенки: exp(-distance / attenuation_length)
    distance = np.asarray(distance, dtype=float)
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("Длина пути должна быть конечной и неотрицательной")
    if np.isnan(attenuation_length) or attenuation_length <= 0:
        raise ValueError("Длина затухания должна быть положительной")
    return np.exp(-distance / attenuation_length)


class Detector:
    def __init__(self, R=2.0, N=40, r=0.10, a=0.5):
        if not np.isfinite(R) or R <= 0:
            raise ValueError("R должно быть положительным")
        if not isinstance(N, (int, np.integer)) or N <= 0:
            raise ValueError("N должно быть целым положительным")
        if not np.isfinite(r) or not 0 < r <= 2 * R:
            raise ValueError("Требуется 0 < r <= 2R")
        if not np.isfinite(a) or a < 0:
            raise ValueError("a должно быть конечным и неотрицательным")
        self.R, self.N, self.r, self.a = float(R), int(N), float(r), float(a)
        i = np.arange(N)
        z = 1 - 2 * (i + 0.5) / N
        phi = np.pi * (3 - np.sqrt(5)) * i
        rho = np.sqrt(1 - z**2)
        self.pmt_positions = R * np.column_stack(
            (rho * np.cos(phi), rho * np.sin(phi), z)
        )
        self.pmt_positions.setflags(write=False)
        # Дерево строится один раз для неизменной геометрии.
        self.tree = KDTree(self.pmt_positions, copy_data=True)

    def find_pmt(self, points, method="tree", block_size=512):
        #Номер ФЭУ или -1. points лежат на поверхности сферы R
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.empty(0, dtype=int)
        points = np.atleast_2d(points)
        if points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points должна иметь форму (n, 3)")
        if method not in ("tree", "full"):
            raise ValueError("method: 'tree' или 'full'")
        if method == "tree":
            #eps=0: точный ближайший сосед; порог проверяем сами.
            _, nearest = self.tree.query(points, k=1, eps=0.0)
        else:
            if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
                raise ValueError("block_size должно быть целым положительным")
            nearest = np.empty(len(points), dtype=int)
            for start in range(0, len(points), block_size):
                block = points[start:start + block_size]
                dots = block @ self.pmt_positions.T
                nearest[start:start + len(block)] = np.argmax(dots, axis=1)
        #Единый критерий для обоих способов поиска; прежняя геометрия.
        delta = points - self.pmt_positions[nearest]
        inside = np.sum(delta**2, axis=1) <= self.r**2
        return np.where(inside, nearest, -1)

    def find_pmt_loop(self, points):
        #Два цикла Python: по фотонам и по всем ФЭУ (эталон скорости)
        result = np.full(len(points), -1, dtype=int)
        for i, (x, y, z) in enumerate(points):
            best_distance2 = math.inf
            nearest = -1
            for j, (px, py, pz) in enumerate(self.pmt_positions):
                d2 = (x - px)**2 + (y - py)**2 + (z - pz)**2
                if d2 < best_distance2:
                    best_distance2, nearest = d2, j
            if best_distance2 <= self.r**2:
                result[i] = nearest
        return result

    def photoelectrons(self, n_hits, rng):
        #Для H попаданий K ~ Poisson(a*H); поддерживает массив по ФЭУ
        n_hits = np.asarray(n_hits)
        if not np.isfinite(n_hits).all() or np.any(n_hits < 0):
            raise ValueError("Число попаданий должно быть неотрицательным")
        if np.any(n_hits != np.floor(n_hits)):
            raise ValueError("Число попаданий должно быть целым")
        return rng.poisson(self.a * n_hits)


def simulate_event(detector, source, E, rng, attenuation_length=np.inf,
                   batch_size=100, method="tree"):
    #Одно событие; возвращает массив фотоэлектронов формы (N,)
    # Проверка геометрии и длины затухания также при E=0.
    sphere_intersections(source.position, np.empty((0, 3)), detector.R)
    survival_probability(np.empty(0), attenuation_length)
    hits = np.zeros(detector.N, dtype=np.int64)
    for directions in source.emit_batches(E, rng, batch_size):
        points, distance = sphere_intersections(source.position, directions, detector.R)
        survive = rng.random(len(distance)) < survival_probability(
            distance, attenuation_length
        )
        ids = detector.find_pmt(points[survive], method=method)
        hits += np.bincount(ids[ids >= 0], minlength=detector.N)
    return detector.photoelectrons(hits, rng)


def simulate_response(detector, source, E=1.0, n_events=500, seed=2026,
                      attenuation_length=np.inf, batch_size=1000, method="tree"):
    #Полный сигнал S=sum(K_j) для независимых событий
    rng = np.random.default_rng(seed)
    signal = np.empty(n_events, dtype=np.int64)
    for i in range(n_events):
        signal[i] = simulate_event(
            detector, source, E, rng, attenuation_length, batch_size, method
        ).sum()
    return signal


def response_pair(detector, source, E=1.0, n_events=500, seed=2026,
                  attenuation_length=8.0, batch_size=1000, method="tree"):
    #Отклик без/с затуханием на одних фотонах и одних флуктуациях ФЭУ. Общие случайные числа уменьшают шум разности. Каждый столбец сохраняет правильное распределение; столбцы между собой зависимы.
    
    sphere_intersections(source.position, np.empty((0, 3)), detector.R)
    survival_probability(np.empty(0), attenuation_length)
    rng = np.random.default_rng(seed)
    signal = np.zeros((n_events, 2), dtype=np.int64)
    for event in range(n_events):
        kept_hits = np.zeros(detector.N, dtype=np.int64)
        lost_hits = np.zeros(detector.N, dtype=np.int64)
        for directions in source.emit_batches(E, rng, batch_size):
            points, distance = sphere_intersections(source.position, directions, detector.R)
            ids = detector.find_pmt(points, method=method)
            survive = rng.random(len(ids)) < survival_probability(
                distance, attenuation_length
            )
            hit = ids >= 0
            kept_hits += np.bincount(ids[hit & survive], minlength=detector.N)
            lost_hits += np.bincount(ids[hit & ~survive], minlength=detector.N)
        kept_pe = detector.photoelectrons(kept_hits, rng)
        lost_pe = detector.photoelectrons(lost_hits, rng)
        signal[event, 0] = (kept_pe + lost_pe).sum()
        signal[event, 1] = kept_pe.sum()
    return signal


def central_expectation(detector, E=1.0, Y=10_000.0, attenuation_length=np.inf):
    #Точный средний сигнал в центре для неперекрывающихся сферических шапок
    coverage = detector.N * detector.r**2 / (4 * detector.R**2)
    return E * Y * detector.a * coverage * math.exp(-detector.R / attenuation_length)