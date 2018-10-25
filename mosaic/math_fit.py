


def fit_third_order(x1, x2, y1, y2, dy1, dy2):
    if x2 != x1:
        a = (x1*dy1 + x1*dy2 - x2*dy1 - x2*dy2 - 2*y1 + 2*y2)/(x1 - x2)**3
        b = (x1**2*(-dy1) - 2*x1**2*dy2 - x1*x2*dy1 + x1*x2*dy2 + 3*x1*y1 - 3*x1*y2 + 2*x2**2*dy1 + x2**2*dy2 + 3*x2*y1 - 3*x2*y2)/(x1 - x2)**3
        c = (x1**3*dy2 + 2*x1**2*x2*dy1 + x1**2*x2*dy2 - x1*x2**2*dy1 - 2*x1*x2**2*dy2 - 6*x1*x2*y1 + 6*x1*x2*y2 - x2**3*dy1)/(x1 - x2)**3
        d = (x1**3*(-x2)*dy2 + x1**3*y2 - x1**2*x2**2*dy1 + x1**2*x2**2*dy2 - 3*x1**2*x2*y2 + x1*x2**3*dy1 + 3*x1*x2**2*y1 - x2**3*y1)/(x1 - x2)**3   
        return a, b, c, d
    return None
