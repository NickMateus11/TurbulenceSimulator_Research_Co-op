
# import tensorflow as tf
# tf.random.set_seed(0)

from PhaseScreenGenerator import PhaseScreenGenerator

N = 2048
r0 = 0.1
L0 = 100
l0 = 0.01
D = 2
delta = D/N

PSG = PhaseScreenGenerator(r0, N, delta, L0, l0)

PSG.next()
PSG.show()

PSG.next()
PSG.show()

from timeit import default_timer as Timer
start = Timer()
for _ in range(100):
    PSG.next()
print(Timer()-start)
