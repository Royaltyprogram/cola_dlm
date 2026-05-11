import json

import torch

from cola_dlm.logging import JSONLMetricsLogger


def test_jsonl_metrics_logger_writes_valid_one_line_records(tmp_path):
    path = tmp_path / "metrics.jsonl"

    with JSONLMetricsLogger(path) as logger:
        logger.log(
            1,
            {
                "loss": torch.tensor(1.25),
                "tokens": 4,
                "values": torch.tensor([1.0, 2.0]),
            },
        )
        logger.log(2, {"loss": 0.75}, metadata={"split": "train"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first == {
        "loss": 1.25,
        "step": 1,
        "tokens": 4,
        "values": [1.0, 2.0],
    }
    assert second == {
        "loss": 0.75,
        "metadata": {"split": "train"},
        "step": 2,
    }


def test_jsonl_metrics_logger_tensorboard_is_optional(tmp_path):
    path = tmp_path / "metrics.jsonl"

    logger = JSONLMetricsLogger(path, tensorboard_dir=tmp_path / "tb")
    logger.log(1, {"loss": torch.tensor(1.0)})
    logger.close()

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "loss": 1.0,
        "step": 1,
    }
