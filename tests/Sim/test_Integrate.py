import unittest
import numpy as np
from japl.Sim.Integrate import runge_kutta_4



class TestExample(unittest.TestCase):


    def func(self, t, X, U, dt):
        pos = X[0]
        vel = X[1]
        acc = U[0]
        out = [pos + vel * dt,
               vel + acc * dt]
        return np.array(out)


    def test_rung_kutta_4(self):
        dt = 0.1
        time = 0
        X = np.array([0, 0])
        U = np.array([1])
        T = []
        for _ in range(10):
            X_new, T_new = runge_kutta_4(f=self.func,
                                         t=time,
                                         X=X,
                                         h=dt,
                                         args=(U, dt))
            X = X_new
            T += [T_new]
            time += dt

        truth = np.array([ 0.009999897516607761,
                          0.171827974413516604])
        self.assertTrue((X == truth).all())
        self.assertTrue((T[-1] - 1.0) < 1e-18)


if __name__ == '__main__':
    unittest.main()
