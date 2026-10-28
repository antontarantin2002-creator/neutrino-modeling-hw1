import numpy as np

from beam_python import momentum_in_CM

#Генерация направлений

def random_direction(N, rng):
    phi = rng.uniform(0, 2 * np.pi, N)
    cos_theta = rng.uniform(-1, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2)

    return np.array([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ])

#Создание четырёхимпульсов    

def make_vector4(mass, momentum, direction):
    momentum = np.asarray(momentum).reshape(1, -1)

    p3 = momentum * direction

    E = np.sqrt(
        mass**2 + np.sum(p3**2, axis=0)
    )

    return np.vstack([E, p3])   
    
#Добавим преобразование вдоль z

def lorentz_boost_z(beta, p4):
    E, px, py, pz = p4

    gamma = 1 / np.sqrt(1 - beta**2)

    return np.array([
        gamma * (E - beta * pz),
        px,
        py,
        gamma * (pz - beta * E)
    ])    

#Запустим векторизованную модель    

def neutrino_beam_z(N, M, Pmax, m, rng):
    P = rng.uniform(0, Pmax, N)

    parent = make_vector4(
        M,
        P,
        np.array([[0], [0], [1]])
    )

    nu_cm = make_vector4(
        0,
        np.full(N, momentum_in_CM(M, m)),
        random_direction(N, rng)
    )

    beta = parent[3] / parent[0]

    return lorentz_boost_z(-beta, nu_cm)

#Преобразование Лоренца для произвольного направления    

def lorentz_boost(beta, p4):
    beta = np.asarray(beta).reshape(3, -1)

    E = p4[0]
    p3 = p4[1:]

    beta_squared = np.sum(beta**2, axis=0)

    if np.any(beta_squared >= 1):
        raise ValueError("Скорость должна быть меньше c")

    gamma = 1 / np.sqrt(1 - beta_squared)

    beta_dot_p = np.sum(beta * p3, axis=0)

    E_new = gamma * (E - beta_dot_p)

    coefficient = (
        gamma**2 / (gamma + 1) * beta_dot_p
        - gamma * E
    )

    p_new = p3 + beta * coefficient

    return np.vstack([E_new, p_new])    

#Добавим разброс направлений родительских адронов
    
def hadron_beam(N, mass, Pmax, rng, theta_rms=0):
    phi = rng.uniform(0, 2 * np.pi, N)
    theta = rng.normal(0, theta_rms, N)

    direction = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    P = rng.uniform(0, Pmax, N)

    return make_vector4(mass, P, direction)

#Теперь функция распада. Сразу сохраним также мюон, он указан в дополнительной постановке задачи:        

def decay_beam(parent, m_lepton, rng):
    E = parent[0]
    P = parent[1:]

    M = np.sqrt(
        E**2 - np.sum(P**2, axis=0)
    )

    p_star = momentum_in_CM(M, m_lepton)

    direction = random_direction(len(E), rng)

    nu_cm = make_vector4(
        0,
        p_star,
        direction
    )

    mu_cm = make_vector4(
        m_lepton,
        p_star,
        -direction
    )

    beta = P / E

    nu_lab = lorentz_boost(-beta, nu_cm)
    mu_lab = lorentz_boost(-beta, mu_cm)

    return nu_lab, mu_lab