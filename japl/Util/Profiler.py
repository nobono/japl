from time import perf_counter



class Profiler:

    """This class is for profiling functions / classes. The
    Purpose of this class is to measure the execution time of
    a program loop and provide delta-time / frequency information."""

    def __init__(self) -> None:
        self.t = 0
        self.t_total = 0
        self.count = 0
        self.dt_average = 0
        self.continuous_print = False


    def __call__(self) -> None:
        if self.count > 1:
            dt = (perf_counter() - self.t)
            self.t_total += dt
            self.dt_average = self.t_total / self.count
            if self.continuous_print:
                header = "Profiler Info:"
                print("%s dt: %.5f, Hz: %.2f" % (header, dt, (1 / dt)))
        self.t = perf_counter()
        self.count += 1


    def set_continuous_print(self, val: bool|int) -> None:
        """This method enables / disables whether
        or not profiler information is displayed during
        each internal call of the object."""
        self.continuous_print = bool(val)


    def print_info(self) -> None:
        wrap_str = "-" * 50
        header = f"\n{wrap_str}\nProfiler Info:\n{wrap_str}"
        print(header)
        print("ave_dt: %.5f, ave_Hz: %.1f" % (self.dt_average, (1 / max(1e-6, self.dt_average))))
        print(wrap_str, end='\n\n')
