import numpy as np
import math
import matplotlib.pyplot as plt

a = .0005
b = 1000

c = 0

V = 6584 * 3.5152625 # To account for image resizing from (240,240,155) to (224,224,155)​ #replace
t = 0
dt = 1 #Keeep it at 1, or you have to change the ticks
t_final = 13
initial_amounts = np.array([10892, 903, 752, 11972, 495, 3161, 8684, 12600, 9766, 1017, 8723, 1611, 1934])

gompertz_results = []

def V_prime(V):
  return a * (math.log(b) - math.log(V)) * V - c*V

for value in initial_amounts:
  V = value
  
  t_values = []
  V_values = []

  t = 0
  dt = 1 #Keeep it at 1, or you have to change the ticks
  t_final = 13

  while t <= t_final:
    t_values.append(t)
    V_values.append(V)
    V1 = V_prime(V)
    V2 = V_prime(V + dt * V1/2)
    V3 = V_prime(V + dt * V2/2)
    V4 = V_prime(V + dt * V3)


    V += V1/6 + V2/3 + V3/3 + V4/6
    t += dt

  gompertz_results.append(V_values[-1])

gompertz_results = np.array([gompertz_results])
gompertz_results

