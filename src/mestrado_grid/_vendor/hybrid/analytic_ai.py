"""Reuse the authenticated, tool-free transport with the expanded physical contract."""
import allocation_ai as transport
from analytic_policy import PROMPT, SCHEMA, validate

transport.PROMPT = PROMPT
transport.SCHEMA = SCHEMA
transport.validate = validate


class Analyst(transport.CodexSupervisor):
    def choose(self, *args, **kwargs):
        answer = super().choose(*args, **kwargs)
        prefix = self.path / f"{self.calls - 1:06d}"
        command = transport.read(prefix.with_suffix(".command.json"))
        command["role"] = "Physical analyst: concrete catalogue actions and coupled dispatch experiments"
        transport.save(prefix.with_suffix(".command.json"), command)
        return answer
