import unittest
from mestrado_grid.worker import Meter,BudgetEnd
class Contracts(unittest.TestCase):
    def test_no_free_duplicate_or_budget_overrun_and_separate_audit(self):
        class E:
            def evaluate(self,c):return self._case(c)
            def _case(self,c):return c
        meter=Meter(E,2);ev=E()
        self.assertEqual(ev.evaluate(7),7);ev.evaluate(7)
        with self.assertRaises(BudgetEnd):ev.evaluate(7)
        self.assertEqual((meter.search_calls,meter.search_AC),(2,2))
        meter.phase='audit';E().evaluate(7)
        self.assertEqual((meter.audit_calls,meter.audit_AC),(1,1))
    def test_hidden_case_calls_fail(self):
        class E:
            def evaluate(self,c):return self._case(c)
            def _case(self,c):return c
        meter=Meter(E,2)
        with self.assertRaisesRegex(RuntimeError,'Unbudgeted'):E()._case(3)
    def test_lp_meter(self):
        class Module:
            linprog=staticmethod(lambda x:x+1)
        class E:
            def evaluate(self,c):return c
            def _case(self,c):return c
        meter=Meter(E,2);meter.instrument_lp(Module)
        self.assertEqual(Module.linprog(2),3);self.assertEqual(meter.lp_calls,1)
if __name__=='__main__':unittest.main()
