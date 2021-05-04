import matplotlib.pyplot as plt
import numpy as np

success_rate = []
result = np.load("evaluations.npz")
print('result:', result['results'])

for i in result['results']:
    print(i)
    success_rate.append(i)

success_rate = np.sum(success_rate, 1)
success_rate = success_rate/10
augument_success = []
for s in success_rate:
	for i in range(50):
		augument_success.append(s)

print(len(augument_success))

moving_average = []
num = 100
for i in range(len(augument_success)-num):
	average = sum(augument_success[i:i+num])/num
	moving_average.append(average)




plt.plot(moving_average, label='jumping')
plt.legend()
plt.show()
