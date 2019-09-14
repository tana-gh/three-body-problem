import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

def f(t, p, m):
    return [
        p[6],
        p[7],
        p[8],
        p[9],
        p[10],
        p[11],
        -fac3(p[0], p[1], p[2], p[3], p[4], p[5], m[1], m[2]),
        -fac3(p[1], p[0], p[3], p[2], p[5], p[4], m[1], m[2]),
        -fac3(p[2], p[3], p[4], p[5], p[0], p[1], m[2], m[0]),
        -fac3(p[3], p[2], p[5], p[4], p[1], p[0], m[2], m[0]),
        -fac3(p[4], p[5], p[0], p[1], p[2], p[3], m[0], m[1]),
        -fac3(p[5], p[4], p[1], p[0], p[3], p[2], m[0], m[1])
    ]

def fac3(p00, p01, p10, p11, p20, p21, m1, m2):
    return m1 * fac2(p00, p01, p10, p11) + m2 * fac2(p00, p01, p20, p21)

def fac2(p00, p01, p10, p11):
    r0 = p00 - p10
    r1 = p01 - p11
    return r0 / np.power(r0 ** 2.0 + r1 ** 2.0, 1.5)

def main():
    p0 = [
        3.0, 4.0,
        0.0, 0.0,
        3.0, 0.0,
        0.0, 0.0,
        0.0, 0.0,
        0.0, 0.0
    ]

    t0 = 0.0
    t1 = 70.0
    dt = 0.0001

    m = [
        3.0,
        4.0,
        5.0
    ]

    p = integrate(f, p0, t0, t1, dt, m)
    plot_p(p, 'figure.png')

def integrate(f, p0, t0, t1, dt, m):
    r = ode(f)
    r.set_integrator('dopri5')
    r.set_initial_value(p0, t0)
    r.set_f_params(m)
    
    arr = []
    
    while r.successful() and r.t < t1:
        arr.append(r.integrate(r.t + dt))
    
    return np.array(arr)

def plot_p(p, filename):
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.set_title('Three-Body Problem')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(True)
    ax.axhline(0.0, color='gray')
    ax.axvline(0.0, color='gray')
    ax.set_xlim(-4.0, 8.0)
    ax.set_ylim(-6.0, 6.0)
    
    view = np.reshape(p, (p.shape[0], 2, 3, 2))
    
    ax.plot(view[:, 0, 0, 0], view[:, 0, 0, 1])
    ax.plot(view[:, 0, 1, 0], view[:, 0, 1, 1])
    ax.plot(view[:, 0, 2, 0], view[:, 0, 2, 1])

    plt.savefig(filename)

if __name__ == '__main__':
    main()
