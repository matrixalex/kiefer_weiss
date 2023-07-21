class StepData:

    def __init__(self, h, last_step, val=None, accept_at=None, frm=None, length=None):
        self.last_step = last_step
        self.frm = frm
        self.accept_at = accept_at
        self.length = length
        self.val = val
        self.h = h

    def __str__(self):
        res = 'StepData <'
        res += f'h={self.h} '
        res += f'length={self.length} '
        res += f'val={self.val} '
        res += f'last_step={self.last_step} '
        res += f'accept_at={self.accept_at}>'
        return res
