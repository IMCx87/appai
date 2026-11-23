import json
import random

from appai import workload as w


def test_build_prompt_size_and_nocache():
    p = w.build_prompt("Pergunta?", 400, nocache=False)
    assert abs(w.approx_tokens(p) - 400) < 25
    a = w.build_prompt("Q", 50, True, random.Random(1))
    b = w.build_prompt("Q", 50, True, random.Random(2))
    assert a != b and a.startswith("[ref ")


def test_lognormal_mean_and_bounds():
    rng = random.Random(3)
    vals = [w.sample_size(200, "Lognormal (realista)", 0.6, rng) for _ in range(5000)]
    assert 170 < sum(vals) / len(vals) < 230
    assert min(vals) >= 8 and max(vals) <= 1200
    assert w.sample_size(200, "Fixo", 0.6, rng) == 200


def test_make_requests_dataset():
    ds = [{"prompt": "a b c", "out": 33}, {"prompt": "d e", "out": None}]
    reqs = w.make_requests(20, {"question": "q", "in": 10, "out": 77, "nocache": False,
                                "dataset": ds}, random.Random(0))
    assert {r["max_tokens"] for r in reqs} <= {33, 77}
    assert {r["prompt"] for r in reqs} <= {"a b c", "d e"}


def test_load_dataset_formats():
    jsonl = b'{"prompt": "oi", "max_tokens": 50}\n{"prompt": "tchau"}\nlixo\n'
    assert w.load_dataset(jsonl, "x.jsonl") == [{"prompt": "oi", "out": 50},
                                                {"prompt": "tchau", "out": None}]
    sharegpt = json.dumps([{"conversations": [{"from": "human", "value": "pergunta"},
                                              {"from": "gpt", "value": "um dois tres"}]}])
    d = w.load_dataset(sharegpt.encode(), "sg.json")
    assert d[0]["prompt"] == "pergunta" and d[0]["out"] == 4
    assert len(w.load_dataset(b"a\n\nb\n", "p.txt")) == 2
    assert w.dataset_stats(d)["n"] == 1
