"""Isolated correction to a parity bug. Not injected into running trials/campaigns."""
class SharedDispatchMode:
    def __init__(self,visits=0):self.visits=visits
    def next(self):
        mode='cycle' if self.visits%2==0 else 'cases'
        self.visits+=1
        return mode
