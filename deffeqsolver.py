import numpy as np 
import math 
import matplotlib.pyplot as plt

#a = .0005
#b = 2000
#38.4382700

a = .0125
b = 25000

#c = 262.5
c = 0

#a = 0.9
#b = 13900
#c = 12000

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
  dt = 0.1 #Keeep it at 1, or you have to change the ticks
  t_final = 13

  while t <= t_final:
    t_values.append(t)
    V_values.append(V)
    V1 = V_prime(V)
    V2 = V_prime(V + dt * V1/2)
    V3 = V_prime(V + dt * V2/2)
    V4 = V_prime(V + dt * V3)


    V += (V1/6 + V2/3 + V3/3 + V4/6) * 0.1
    t += dt

  gompertz_results.append(V_values[-1])

gompertz_results = np.array([gompertz_results])
gompertz_results


exponential_pred = []

for x in range(13):
  if initial_amounts[x] < 3.88 * 1000:
    r = 3.9 / 100
  elif initial_amounts[x] < 36.88 * 1000:
    r = 1.2 / 100
  else:
    r = 0.3 / 100

  exponential_pred.append(initial_amounts[x]*((1+r)**13))

exponential_pred = np.array([exponential_pred])
exponential_pred

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

MAPE(exponential_pred, gompertz_results)

print (exponential_pred)
print (gompertz_results)

x=np.linspace(0,14,14)

plt.scatter(np.array(gompertz_results) / 1000, np.array(exponential_pred) / 1000, color='red')
plt.plot(x,x, color='green')
plt.xlabel("Gompertz Model Predictions (mL)")
plt.ylabel("In Vivo Exponential Growth Simulation (mL)")
plt.title("Gompertz Predictions vs. In Vivo Exponential Growth Data")

print (x)

V = 12600
  
t_values = []
V_values = []

t = 0
dt = 0.1 #Keeep it at 1, or you have to change the ticks
t_final = 13

while t <= t_final:
  t_values.append(t)
  V_values.append(V)
  V1 = V_prime(V)
  V2 = V_prime(V + dt * V1/2)
  V3 = V_prime(V + dt * V2/2)
  V4 = V_prime(V + dt * V3)


  V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
  t += dt

  t_values = []
exp_values = []

t = 0
dt = 0.1
t_final = 13
V_init = 1017

while t <= t_final:
  t_values.append(t)
  exp_values.append(V_init*((1+r)**t))

  if V_init < 3.88 * 1000:
    r = 3.9 / 100
  elif V_init < 36.88 * 1000:
    r = 1.2 / 100
  else:
    r = 0.3 / 100
  
  t += dt


plt.plot(t_values, np.array(V_values) / 1000, label = "Gompertz Model Predicted Growth")
plt.plot(t_values, np.array(exp_values) / 1000, label = "Exponential Model Predicted Growth")
plt.ylabel("Predicted Tumor Size (mL)")
plt.xlabel("Time (days)")
leg = plt.legend(loc='upper center')
#plt.title("Gompertz vs. Exponential Model Predicted Growth for 3.161 mL tumor")

print (V_values[-1])

GAMMA = 0.212
r = math.log(2) / 1.04

def V_prime_with_gamma(V):
  t_for_drug = t % 21
  if t_for_drug > 10*4: # the drug wears off after approximately 4 half lifes
    return a * (math.log(b) - math.log(V)) * V
  else:
    return a * (math.log(b) - math.log(V)) * V  - GAMMA * (math.e**(-r * (t_for_drug))) * V


V = V_values[-1]
  
gamma_t_values = []
gamma_V_values = []

t = 0
dt = 0.1 #Keeep it at 1, or you have to change the ticks
t_final = 42

while t <= t_final:
  gamma_t_values.append(t)
  gamma_V_values.append(V)
  V1 = V_prime_with_gamma(V)
  V2 = V_prime_with_gamma(V + dt * V1/2)
  V3 = V_prime_with_gamma(V + dt * V2/2)
  V4 = V_prime_with_gamma(V + dt * V3)


  V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
  t += dt

print (gamma_V_values[0] - gamma_V_values[10*2] / gamma_V_values[0])

V = 12600
  
full_t_values = []
full_V_values = []

t = 0
dt = 0.1 #Keeep it at 1, or you have to change the ticks
t_final = 55

while t <= t_final:
  full_t_values.append(t)
  full_V_values.append(V)
  V1 = V_prime(V)
  V2 = V_prime(V + dt * V1/2)
  V3 = V_prime(V + dt * V2/2)
  V4 = V_prime(V + dt * V3)


  V += (V1/6 + V2/3 + V3/3 + V4/6) * dt
  t += dt


times = np.concatenate((np.array(t_values), np.array(gamma_t_values) + t_values[-1]))
V_values_combined = np.concatenate((np.array(V_values), np.array(gamma_V_values)))

plt.plot(full_t_values, np.array(full_V_values) / 1000, label = "Gompertz Model without Chemotherapy Component")

plt.plot(times, V_values_combined / 1000, label = "Gompertz Model with Chemotherapy")
plt.ylabel("Predicted Tumor Size (mL)")
plt.xlabel("Time (days)")
leg = plt.legend(loc='lower center')
plt.ylim(ymin=0)
plt.grid()
plt.title("Predicted Gompertz Tumor Growth with and without Doxorubicin Component")

plt.plot(gamma_t_values, np.array(gamma_V_values) / 1000, label = "Gompertz Model with Chemotherapy Predicted Growth")
plt.ylabel("Predicted Tumor Size (mL)")
plt.xlabel("Time (days)")
leg = plt.legend(loc='upper center')
plt.title("")