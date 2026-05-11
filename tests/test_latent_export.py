import csv
import importlib
import json

import pytest
import torch

from cola_dlm.latent_export import export_latent_projection
from cola_dlm.vae import DiagonalGaussianPosterior, TextVAEOutput


def test_latent_export_public_exports_are_small():
    import cola_dlm.latent_export as latent_export

    assert latent_export.__all__ == ("export_latent_projection",)


def test_pca_jsonl_export_preserves_pairwise_distances_and_metadata(tmp_path):
    latents = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 1.0]],
        ]
    )
    token_ids = torch.tensor([[10, 11], [12, 13]])
    path = tmp_path / "projection.jsonl"

    records = export_latent_projection(
        latents,
        path,
        token_ids=token_ids,
        max_points=4,
    )

    assert [(record["batch_index"], record["token_position"]) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [record["token_id"] for record in records] == [10, 11, 12, 13]
    projected = torch.tensor([[record["x"], record["y"]] for record in records])
    flat_latents = latents.reshape(-1, latents.shape[-1])
    torch.testing.assert_close(
        torch.cdist(projected, projected),
        torch.cdist(flat_latents, flat_latents),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    for record in records:
        assert record["explained_variance_ratio_x"] >= 0.0
        assert record["explained_variance_ratio_y"] >= 0.0
    assert records == [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_export_accepts_batches_of_vae_outputs(tmp_path):
    first = _make_output(torch.tensor([[[1.0, 0.0], [2.0, 0.0]]]))
    second = _make_output(torch.tensor([[[3.0, 0.0]]]))
    path = tmp_path / "projection.jsonl"

    records = export_latent_projection(
        [first, second],
        path,
        token_ids=[torch.tensor([[20, 21]]), torch.tensor([[22]])],
        max_points=8,
    )

    assert [(record["batch_index"], record["token_position"]) for record in records] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert [record["token_id"] for record in records] == [20, 21, 22]


def test_csv_export_and_max_points_truncate_deterministically(tmp_path):
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    path = tmp_path / "projection.csv"

    records = export_latent_projection(latents, path, max_points=4)

    assert [(record["batch_index"], record["token_position"]) for record in records] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
    ]
    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [(int(row["batch_index"]), int(row["token_position"])) for row in rows] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
    ]
    assert "x" in rows[0]
    assert "y" in rows[0]
    assert "explained_variance_ratio_x" in rows[0]


def test_umap_export_fails_clearly_when_optional_dependency_is_absent(
    tmp_path,
    monkeypatch,
):
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "umap":
            raise ModuleNotFoundError("No module named 'umap'")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="use method='pca'"):
        export_latent_projection(
            torch.zeros(1, 2, 2),
            tmp_path / "projection.jsonl",
            max_points=2,
            method="umap",
        )


def _make_output(latents: torch.Tensor) -> TextVAEOutput:
    posterior = DiagonalGaussianPosterior(
        mu=latents,
        logvar=torch.zeros_like(latents),
    )
    return TextVAEOutput(
        logits=torch.zeros(*latents.shape[:2], 3),
        posterior=posterior,
        latents=latents,
        kl=torch.zeros(latents.shape[:2]),
    )
