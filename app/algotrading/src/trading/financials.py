class Financials:
    SET_COMMISION_RATE = 0.00053
    SET_MARKET_SPREAD = 0.00006

    def __init__(self):
        self.price_penalty = 1 - self.SET_COMMISION_RATE - self.SET_MARKET_SPREAD