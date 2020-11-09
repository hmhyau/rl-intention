class Schedule():
    def value(self, step):
        raise NotImplementedError

class ConstantSchedule(Schedule):
    def __init__(self, value):
        self._value = value
    
    def value(self, step):
        return self._value

class LinearSchedule(Schedule):
    def __init__(self, frac, initial, final):
        self._frac = frac
        self._initial = initial
        self._final = final
        self._value = self._initial

    def value(self, step):
        frac = min(float(step/self._frac), 1.)
        self._value = self._initial + frac * (self._final - self._initial)
        return self._value

class ExponentialSchedule(Schedule):
    def __init__(self, frac, initial, final):
        self._frac = frac
        self._initial = initial
        self._final = final
        self._value = self._initial

    def value(self, step):
        self._value = max(self._value*self._frac, self._final)
        return self._value
