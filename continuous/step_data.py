class StepData:

    def __init__(self, n, grid, val):
        self.n = n
        self.grid = grid
        self.val = val
        

    def __str__(self):
        res = 'StepData <'
        res += f'n={self.n} '
        res += f'grid={self.grid} '
        res += f'val={self.val} '
        return res