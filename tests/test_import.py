import torch


def test_import():
    import controllable_patching_striding  # noqa: F401
    from controllable_patching_striding.models.shared_utils.mlps import MLP

    model = MLP(3)
    model(torch.randn(1, 3))
