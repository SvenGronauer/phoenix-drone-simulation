import pandas as pd
import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
import phoenix_drone_simulation  # noqa
import time
from scipy import signal

# 2.130295e-11 * PWM ** 2 + 1.032633e-6 * PWM + 5.484560e-4
PWM_FORCE_FACTOR_0 = 5.484560e-4
PWM_FORCE_FACTOR_1 = 1.032633e-6
PWM_FORCE_FACTOR_2 = 2.130295e-11


MAX_THRUST = 0.028 * 9.81 * 1.9 / 4


def create_pwm_signal(N):
    a = 4
    ts = np.linspace(0, 2*np.pi, N)
    signal = np.cos(ts * a)
    # signal = np.ones_like(signal)
    return signal


def get_system(dt=0.002):
    # === System 1
    num = [0, 7.2345374e-8]
    den = [1, -0.9695404]
    # den = [1, -0.9466]
    tf_1 = signal.TransferFunction(num, den, dt=dt)
    sys_1 = tf_1.to_ss()
    A_1, B_1, C_1 = float(sys_1.A), float(sys_1.B), float(sys_1.C)
    return sys_1


def transfer_func(u, i):
    a = 0.9745210
    a = 1
    K = 6.0705967e-8
    K = 1

    return K * a**i * u


def plot_heaviside():
    sim_time = 0.5  # in seconds
    dt = 0.002
    N = int(sim_time / dt)
    sys = get_system(dt)
    sig = np.zeros(N)
    sig[:] = 6e4
    ts = np.arange(N) * dt

    tout1, yout1, xout1 = signal.dlsim(sys, u=sig, t=ts)
    # tout2, yout2, xout2 = signal.dlsim(sys, sig2, ts_2)

    plt.plot(tout1, yout1)
    plt.plot(tout1, 6e4*0.67*np.ones_like(tout1)*sys.C[0]/(1-sys.A[0]))
    # plt.plot(tout2, yout2)
    # plt.plot(ts_1, sig1/5e5)
    # plt.plot(ts_2, PWM_2)
    plt.show()


def my_incremental_approach(inputs, N):
    T_s = 1/500
    T = 0.150 / 4  # == 180ms / 4

    A = 1 - T_s / T
    print(f'A={A}')
    K = MAX_THRUST / 65535  # maximum thrust per motor when PWM = 65535
    print(f'MAX_THRUST={MAX_THRUST}')
    # K = 1
    B = T_s / T  # * K

    x = 0.
    outputs = np.zeros(N)
    for i in range(N-1):
        x = A * x + B * inputs[i]
        outputs[i+1] = x * K
    return outputs


def angvel2thrust(w, linearity=0.424):
    return (1 - linearity) * w**2 + linearity * w


def setting_from_NN_KW_26_model_3(inputs, sim_time=1, dt=1/500):
    N = int(sim_time / dt)
    ts = np.arange(N) * dt
    num = [0, 7.23e-08]
    den = [1, -0.95]
    tf = signal.TransferFunction(num, den, dt=dt)
    sys = tf.to_ss()
    tout, yout, xout = signal.dlsim(sys, u=inputs, t=ts)
    # tout2, yout2, xout2 = signal.dlsim(sys, u=inputs, t=ts)
    return tout, yout


def main():
    max_thrust = 0.028 * 9.81 * 1.9 / 4
    dt = 0.002  # in seconds
    dt_alt = 0.002  # in seconds
    sim_time = 1  # in seconds
    N = int(sim_time / dt)

    input_jump = np.ones(N) * 65535

    xs, ys = setting_from_NN_KW_26_model_3(input_jump, sim_time, dt=dt)
    ys_new = my_incremental_approach(inputs=input_jump, N=N)
    plt.plot(xs, ys, xs, np.ones_like(ys) * max_thrust, xs, ys_new)
    plt.show()

    xs = np.arange(100) / 100
    ys = angvel2thrust(xs)
    plt.plot(xs, ys)
    plt.show()

    # plot_heaviside()
    dt1 = 0.002  # in seconds
    dt2 = 0.002  # in seconds
    sim_time = 1  # in seconds

    N1 = int(sim_time / dt1)
    N2 = int(sim_time / dt2)

    ts_1 = np.arange(N1) * dt1
    ts_2 = np.arange(N2) * dt2
    sig1 = create_pwm_signal(N1) * 30000 + 30000
    sig2 = np.ones_like(create_pwm_signal(N2) * 30000 + 30000) * 65535

    # plot_heaviside()
    # my_y = my_incremental_approach(inputs=sig2, N=N1)
    # plt.plot(ts_1, my_y)
    # # plt.plot(ts_2, PWM_2)
    # plt.show()
    return

    # === System 1
    num = [0, 6.8e-8]
    den = [1, -0.9695404]
    tf_1 = signal.TransferFunction(num, den, dt=dt1)
    sys_1 = tf_1.to_ss()
    A_1, B_1, C_1 = float(sys_1.A), float(sys_1.B), float(sys_1.C)

    # === System 2
    # num = [0, 6.0705967e-8]
    den = [1, -0.97]
    tf_2 = signal.TransferFunction(num, den, dt=dt2)
    sys_2 = tf_2.to_ss()
    A_2, B_2, C_2 = float(sys_2.A), float(sys_2.B), float(sys_2.C)

    f = 0.1 * np.ones(4) / 6.0705967e-08

    # print('tf:')
    # print(tf)
    # print('sys:')
    # print(sys)

    tout1, yout1, xout1 = signal.dlsim(sys_1, u=sig1, t=ts_1)
    tout2, yout2, xout2 = signal.dlsim(sys_2, sig2, ts_2)

    plt.plot(tout1, yout1)
    plt.plot(tout2, yout2)
    plt.plot(ts_1, sig1/5e5)
    # plt.plot(ts_2, PWM_2)
    plt.show()

    # test my own implementation:
    # x(k+1) = (1-T_s/T) x(k) + K (T_s/T) u(k)
    # x = 0
    x1, x2 = 0, 0
    K = 1
    # T_s_T = 1 - 0.9745210
    PWM_1 = np.zeros(N1)
    PWM_2 = np.zeros(N2)
    # forces = 0.1 * np.ones((N, 4)) / 6.0705967e-08
    for i in range(N1):
        PWM_1[i] = x1 * C_1
        x1 = A_1 * x1 + B_1 * (sig1[i])
    for i in range(N2):
        PWM_2[i] = x2 * C_2
        x2 = A_2 * x2 + B_2 * (sig2[i])

    # plt.plot(ts_1, PWM_1)
    # plt.scatter(ts_1, sig1)
    # # plt.plot(ts_2, PWM_2)
    # plt.scatter(ts_2, sig2)
    # plt.show()


if __name__ == '__main__':
    main()

