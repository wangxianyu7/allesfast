import sys, time
sys.path.insert(0, '/Users/wangxianyu/Program/Github')
from allesfast import config
from allesfast.computer import update_params, calculate_lnlike_total, calculate_model

config.init('/Users/wangxianyu/Program/Github/allesfast/examples/K2-140', quiet=True)
theta = config.BASEMENT.theta_0.copy()

for _ in range(5):
    calculate_lnlike_total(update_params(theta))

N = 100
p = update_params(theta)

t0 = time.perf_counter()
for _ in range(N):
    calculate_lnlike_total(update_params(theta))
print('Total lnlike : %.2f ms/call' % ((time.perf_counter() - t0) / N * 1000))

t0 = time.perf_counter()
for _ in range(N):
    calculate_model(p, 'Keck', 'rv')
print('model Keck   : %.3f ms' % ((time.perf_counter() - t0) / N * 1000))
