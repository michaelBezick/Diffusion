import numpy as np
import math as m

beta_start = 1e-5
beta_stop = 0.02

beta_schedule = np.linspace(beta_start, beta_stop, 1000)
alpha_schedule = np.ones_like(beta_schedule) - beta_schedule
alpha_bar = np.cumprod(alpha_schedule)
beta_bar = np.cumprod(beta_schedule)
print(m.sqrt(alpha_bar[0]))
print(m.sqrt(alpha_bar[-1]))
print(1-alpha_bar[0])
print(1-alpha_bar[-1])
print(beta_bar[0])
print(beta_bar[-1])
