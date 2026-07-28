import pytest

from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor


def test_issue_2395():
    executor = LocalPythonExecutor(additional_authorized_imports=[])

    with pytest.raises(InterpreterError, match="Forbidden dunder method definition: __del__"):
        executor(
            """
class TimeBomb:
    def __del__(self):
        pass

bomb = TimeBomb()
"""
        )
