import numpy as np
import pytest
import torch

from local_checks.check import (
    FUNCTIONS,
    check_numerics,
    check_rewards,
    official_advantages,
)


def test_upstream_matches_independent_numpy_for_six_settings():
    assert len(check_numerics()) == 6


def test_official_reward_separates_correct_wrong_and_malformed():
    assert len(check_rewards()) == 3


def test_gdpo_can_still_cancel_opposing_reward_signals():
    rewards = torch.tensor(
        [
            [0.0, 3.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [0.0, 3.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    mask = torch.ones(8, 2)
    groups = ["a"] * 4 + ["b"] * 4
    # Opposing identical binary signals also cancel under GDPO in group a.
    # This fixture guards against falsely claiming GDPO can prevent every collapse.
    for method in ("grpo", "gdpo"):
        result = official_advantages(rewards, mask, groups, method)
        assert torch.isfinite(result).all()
        assert torch.allclose(result[:4], torch.zeros(4, 2), atol=1e-5)


def test_singleton_group_matches_upstream_special_case():
    result, _ = FUNCTIONS["compute_grpo_outcome_advantage"](
        torch.tensor([[0.0, 2.0]]), torch.ones(1, 2), ["singleton"]
    )
    np.testing.assert_allclose(result.numpy(), [[2 / (1 + 1e-6)] * 2], rtol=1e-6)


def test_whitening_rejects_only_one_valid_token():
    with pytest.raises(ValueError, match="sum of the mask is one"):
        FUNCTIONS["masked_whiten"](torch.ones(1, 1), torch.ones(1, 1))
