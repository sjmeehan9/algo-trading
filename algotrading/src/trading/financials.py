class Financials:
    SET_COMMISION_RATE = 0.00053

    def __init__(self):
        self.commision_factor = 1 - self.SET_COMMISION_RATE