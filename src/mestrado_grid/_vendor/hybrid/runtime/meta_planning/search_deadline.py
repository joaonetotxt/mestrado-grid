"""Cooperative wall-clock termination; never interrupts an AC evaluation."""
import time
import math


class SearchDeadlineReached(Exception):
    pass


class SearchDeadline:
    def __init__(self, seconds, start=None, clock=None):
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('Positive finite wall time required')
        self.clock=clock or time.monotonic
        self.start=self.clock() if start is None else start
        self.seconds=float(seconds)
        self.deadline=self.start+self.seconds

    def check(self):
        if self.clock()>=self.deadline:
            raise SearchDeadlineReached()

    def report(self):
        elapsed=self.clock()-self.start
        return dict(limit_seconds=self.seconds,elapsed_search_seconds=elapsed,
            overrun_seconds=max(0.,elapsed-self.seconds),clock='monotonic',
            stop_policy='Before next evaluation or scheduling visit; current atomic work finishes')
