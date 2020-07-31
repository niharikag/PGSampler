import numpy as np
import matplotlib.pyplot as plt
'''
# python pg, time comparison
time taken by PG 22.101325035095215
time taken by PG 109.96585321426392
time taken by PG 220.36585760116577
time taken by PG 438.65747356414795 
'''

# data to plot
n_groups = 4
'''
# for 100/100
python_pg = (22.10, 109.96, 220.36, 438.65)
rust_pg = (0.5, 2, 4, 9)
'''

# for 500/500
python_pg = (109, 538, 1079, 2159)
rust_pg = (10, 52, 102, 204)

# create plot
plt.figure(figsize=(10, 4))
#fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, python_pg, bar_width,
alpha=opacity,
color='b',
label='Python')

rects2 = plt.bar(index + bar_width, rust_pg, bar_width,
alpha=opacity,
color='g',
label='Rust')


plt.xlabel('Iterations')
plt.ylabel('Time (in seconds)')
#plt.title('Scores by person')
plt.xticks(index + bar_width, ('1000', '5000', '10000', '20000'))
plt.legend()
plt.tight_layout()
plt.savefig("../plots/time_pg.png")
plt.show()