import math
import random

vector3 = tuple[float, float, float]
vector4 = tuple[float, vector3]


def momentum_in_CM(M: float, m: float) -> float:
    """Вычисляет модуль импульса в системе покоя родителя.

    Рассматривается распад M -> m + безмассовая частица.
    Используются естественные единицы: c = 1.
    """
    P_cm = (M**2 - m**2) / (2 * M)
    return P_cm

#Генерируем изотропное направление

def random_direction():
    phi = random.uniform(0, 2 * math.pi)
    cos_theta = random.uniform(-1, 1)
    sin_theta = math.sqrt(1 - cos_theta**2)

    return (
        sin_theta * math.cos(phi),
        sin_theta * math.sin(phi),
        cos_theta
    )        

#Преобразование Лоренца вдоль z    
                
def lorentz_boost(beta, p4):
    E, (px, py, pz) = p4

    gamma = 1 / math.sqrt(1 - beta**2)

    E_new = gamma * (E - beta * pz)
    pz_new = gamma * (pz - beta * E)

    return (E_new, (px, py, pz_new))
    
def norm2(p4):
    E, (px, py, pz) = p4

    return E**2 - px**2 - py**2 - pz**2

#Соберём генератор нейтрино для фиксированного импульса пиона

def make_vector4(mass, momentum, direction):
    x, y, z = direction

    E = math.sqrt(mass**2 + momentum**2)

    return (
        E,
        (
            momentum * x,
            momentum * y,
            momentum * z
        )
    )

#Теперь опишем один распад:

def decay_at_momentum(M, P, m):
    parent = make_vector4(M, P, (0, 0, 1))

    nu_cm = make_vector4(
        0,
        momentum_in_CM(M, m),
        random_direction()
    )

    beta = parent[1][2] / parent[0]

    return lorentz_boost(-beta, nu_cm)

def neutrino_beam(M_beam, P_beam, m_lepton):
    while True:
        yield decay_at_momentum(
            M_beam,
            P_beam,
            m_lepton
        )

def neutrino_beam_uniform(M_beam, P_beam_max, m_lepton):
    while True:
        P = random.uniform(0, P_beam_max)

        p4_nu = decay_at_momentum(
            M_beam,
            P,
            m_lepton
        )

        yield p4_nu, P