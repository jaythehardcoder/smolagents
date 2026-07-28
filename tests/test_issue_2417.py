import sys
import types

from smolagents.models import VLLMModel


def test_issue_2417(monkeypatch):
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "formatted prompt"

    class FakeLLM:
        def __init__(self, **kwargs):
            self.initialization_kwargs = kwargs

        def generate(self, prompt, sampling_params):
            self.prompt = prompt
            self.sampling_params = sampling_params
            return [
                types.SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[types.SimpleNamespace(text="4", token_ids=[3])],
                )
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm.LLM = FakeLLM
    vllm.SamplingParams = FakeSamplingParams

    tokenizers = types.ModuleType("vllm.tokenizers")
    tokenizers.get_tokenizer = lambda model_id: FakeTokenizer()

    sampling_params = types.ModuleType("vllm.sampling_params")
    sampling_params.StructuredOutputsParams = lambda **kwargs: kwargs

    monkeypatch.setattr("smolagents.models._is_package_available", lambda package: package == "vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)
    monkeypatch.delitem(sys.modules, "vllm.transformers_utils", raising=False)
    monkeypatch.delitem(sys.modules, "vllm.transformers_utils.tokenizer", raising=False)

    model = VLLMModel("test-model", max_tokens=4096)
    output = model.generate([{"role": "user", "content": "What is 2+2?"}])

    assert output.content == "4"
    assert model.model.sampling_params.kwargs["max_tokens"] == 4096
