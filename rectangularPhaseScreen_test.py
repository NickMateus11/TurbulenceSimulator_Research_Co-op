from PhaseScreenGenerator import PhaseScreenGenerator
from matplotlib import pyplot as plt


def test1():
    N = 4096
    M = 1024
    r0 = 0.1
    L0 = 100
    l0 = 0.01
    D = 2
    delta = D/N
    PSG = PhaseScreenGenerator(r0, (N,M), delta, L0, l0)
    rect_phase_screen = PSG.next()

    plt.imshow(rect_phase_screen)
    plt.set_cmap('gray')
    plt.show()


if __name__ == '__main__':
    test1()