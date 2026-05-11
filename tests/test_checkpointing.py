import torch
import torch.nn.functional as F
from torch import nn

import pytest

from cola_dlm.checkpointing import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)


def test_save_and_load_model_optimizer_scheduler_step_and_config(tmp_path):
    model, optimizer, scheduler = _make_training_objects()
    _run_training_step(model, optimizer, scheduler)
    config = {"model": {"input_dim": 2, "output_dim": 1}, "run": {"seed": 7}}
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        config=config,
        metadata={"note": "tiny"},
    )
    restored_model, restored_optimizer, restored_scheduler = _make_training_objects()
    loaded = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert loaded.step == 3
    assert loaded.config == config
    assert loaded.metadata == {"note": "tiny"}
    assert loaded.format_version == CHECKPOINT_FORMAT_VERSION
    _assert_models_close(model, restored_model)
    _assert_optimizer_states_close(optimizer, restored_optimizer)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_named_extra_model_states_round_trip(tmp_path):
    vae = nn.Linear(2, 2)
    dit = nn.Linear(2, 2)
    path = tmp_path / "stage2.pt"

    save_checkpoint(
        path,
        extra_models={"vae": vae, "dit": dit},
        step=1,
        config={"stage": 2},
    )
    restored_vae = nn.Linear(2, 2)
    restored_dit = nn.Linear(2, 2)
    loaded = load_checkpoint(
        path,
        extra_models={"vae": restored_vae, "dit": restored_dit},
    )

    assert loaded.step == 1
    assert loaded.config == {"stage": 2}
    _assert_models_close(vae, restored_vae)
    _assert_models_close(dit, restored_dit)


def test_resume_restores_optimizer_and_scheduler_for_next_step(tmp_path):
    model, optimizer, scheduler = _make_training_objects()
    _run_training_step(model, optimizer, scheduler)
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        config={"stage": 1},
    )

    restored_model, restored_optimizer, restored_scheduler = _make_training_objects()
    load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    original_loss = _run_training_step(model, optimizer, scheduler)
    restored_loss = _run_training_step(
        restored_model,
        restored_optimizer,
        restored_scheduler,
    )

    torch.testing.assert_close(restored_loss, original_loss)
    _assert_models_close(model, restored_model)
    _assert_optimizer_states_close(optimizer, restored_optimizer)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_missing_checkpoint_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        load_checkpoint(tmp_path / "missing.pt")


def test_malformed_checkpoint_contents_raise_clear_errors(tmp_path):
    path = tmp_path / "malformed.pt"
    torch.save({"model": {}}, path)

    with pytest.raises(CheckpointError, match="missing required keys"):
        load_checkpoint(path)

    list_path = tmp_path / "list.pt"
    torch.save(["not", "a", "mapping"], list_path)
    with pytest.raises(CheckpointError, match="must be a mapping"):
        load_checkpoint(list_path)

    bad_extra_path = tmp_path / "bad_extra.pt"
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "package_version": "0.1.0",
            "model": None,
            "extra_models": {"vae": []},
            "optimizer": None,
            "scheduler": None,
            "step": 0,
            "config": {},
            "metadata": {},
        },
        bad_extra_path,
    )
    with pytest.raises(CheckpointError, match="extra model state 'vae'"):
        load_checkpoint(bad_extra_path)


def _make_training_objects() -> tuple[nn.Module, torch.optim.Optimizer, object]:
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    return model, optimizer, scheduler


def _run_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
) -> torch.Tensor:
    inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([[1.0], [0.0]])
    optimizer.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def _assert_models_close(left: nn.Module, right: nn.Module) -> None:
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(right_parameter, left_parameter)


def _assert_optimizer_states_close(
    left: torch.optim.Optimizer,
    right: torch.optim.Optimizer,
) -> None:
    left_state = left.state_dict()
    right_state = right.state_dict()
    assert right_state["param_groups"] == left_state["param_groups"]
    assert right_state["state"].keys() == left_state["state"].keys()
    for parameter_id, left_values in left_state["state"].items():
        right_values = right_state["state"][parameter_id]
        assert right_values.keys() == left_values.keys()
        for name, left_value in left_values.items():
            right_value = right_values[name]
            if torch.is_tensor(left_value):
                torch.testing.assert_close(right_value, left_value)
            else:
                assert right_value == left_value
