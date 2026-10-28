import numpy as np


class Detector:
    def __init__(self, R=2.0, N=40, r=0.10, a=0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a

        i = np.arange(N)

        z = 1 - 2 * (i + 0.5) / N
        phi = np.pi * (3 - np.sqrt(5)) * i

        rho = np.sqrt(1 - z**2)

        self.pmt_positions = R * np.column_stack([
            rho * np.cos(phi),
            rho * np.sin(phi),
            z
        ])

    # Метод определения ФЭУ по точке поверхности
    def find_pmt(self, points):
        points = np.atleast_2d(points)

        dots = points @ self.pmt_positions.T

        nearest = np.argmax(dots, axis=1)

        best = dots[
            np.arange(len(points)),
            nearest
        ]

        inside = (
            best >= self.R**2 - self.r**2 / 2
        )

        return np.where(inside, nearest, -1)

#Добавим генерацию фотоэлектронов. Этот метод логично отнести к Detector, потому что преобразование попавшего света в сигнал зависит от фотокатодов и параметра a, а не от источника.
    
    def photoelectrons(self, n_hits, rng):
        return rng.poisson(
            self.a * np.asarray(n_hits)
        )

class Source:
    def __init__(self, Y=10_000, position=(0.0, 0.0, 0.0)):
        self.Y = Y
        self.position = np.asarray(position, dtype=float)
    def emit(self, E, rng):
        if E < 0:
            raise ValueError("Энергия не может быть отрицательной")

        n_photons = int(round(self.Y * E))

        phi = rng.uniform(0, 2 * np.pi, n_photons)
        cos_theta = rng.uniform(-1, 1, n_photons)

        sin_theta = np.sqrt(1 - cos_theta**2)

        return np.column_stack([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])

#Найдём точку выхода фотона на сферу
    
def sphere_intersections(position, directions, R):
    position = np.asarray(position, dtype=float)

    position_squared = np.dot(position, position)

    if position_squared >= R**2:
        raise ValueError("Источник должен находиться внутри сферы")

    b = directions @ position

    t = -b + np.sqrt(
        b**2 + R**2 - position_squared
    )

    return position + t[:, None] * directions

def simulate_event(detector, source, E, rng):
    directions = source.emit(E, rng)

    points = sphere_intersections(
        source.position,
        directions,
        detector.R
    )

#Выделяет реальные попадания    
    pmt_ids = detector.find_pmt(points)

#Считаем количество попаданий в каждый канал
    n_hits = np.bincount(
        pmt_ids[pmt_ids >= 0],
        minlength=detector.N
    )
# minlength=detector.N гарантирует массив из 40 элементов, включая ФЭУ с нулём попаданий
    
    photoelectrons = detector.photoelectrons(
        n_hits,
        rng
    )

    return photoelectrons

#Смоделируем 1000 событий

def simulate_sample(detector, source, E, n_events, rng):
    charges = np.empty(n_events, dtype=int)

    for i in range(n_events):
        pe = simulate_event(
            detector,
            source,
            E,
            rng
        )

        charges[i] = pe.sum()

    return charges