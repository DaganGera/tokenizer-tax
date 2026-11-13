"""NVML energy meter. Uses the hardware cumulative energy counter (mJ); falls back to 50 Hz power sampling."""
import threading, time, torch, pynvml as N

class Meter:
    def __init__(self, idx=0):
        N.nvmlInit(); self.h = N.nvmlDeviceGetHandleByIndex(idx)
        try: N.nvmlDeviceGetTotalEnergyConsumption(self.h); self.mode = "counter"
        except N.NVMLError: self.mode = "sampling"

    def info(self):
        g = lambda f: (lambda x: x.decode() if isinstance(x, bytes) else x)(f())
        return {"gpu": g(lambda: N.nvmlDeviceGetName(self.h)), "driver": g(N.nvmlSystemGetDriverVersion),
                "sm_clock_mhz": N.nvmlDeviceGetClockInfo(self.h, N.NVML_CLOCK_SM),
                "temp_c": N.nvmlDeviceGetTemperature(self.h, N.NVML_TEMPERATURE_GPU), "mode": self.mode}

    def _read(self): return N.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1000.0  # J

    def __enter__(self):
        torch.cuda.synchronize()
        if self.mode == "sampling":
            self._p, self._stop = [], False
            def run():
                while not self._stop: self._p.append((time.perf_counter(), N.nvmlDeviceGetPowerUsage(self.h) / 1000)); time.sleep(0.02)
            self._th = threading.Thread(target=run); self._th.start()
        else: self._e0 = self._read()
        self._t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        torch.cuda.synchronize(); self.seconds = time.perf_counter() - self._t0
        if self.mode == "sampling":
            self._stop = True; self._th.join(); p = self._p
            self.joules = sum((p[i + 1][0] - p[i][0]) * p[i][1] for i in range(len(p) - 1))
        else: self.joules = self._read() - self._e0

    def idle_power(self, seconds=10):
        """Average board power (W) with the GPU idle."""
        torch.cuda.synchronize()
        with self as m: time.sleep(seconds)
        return m.joules / m.seconds
