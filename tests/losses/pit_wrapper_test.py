import pytest
import torch
from torch.testing import assert_allclose

from asteroid.losses import PITLossWrapper
from asteroid.losses import nosrc_mse, pairwise_mse
from asteroid.losses import nosrc_neg_sisdr, pairwise_neg_sisdr


def bad_loss_func_ndim0(y_true, y_pred):
    return torch.randn(1).mean()


def bad_loss_func_ndim1(y_true, y_pred):
    return torch.randn(1)


def good_batch_loss_func(y_true, y_pred):
    batch, *_ = y_true.shape
    return torch.randn(batch)


def good_pairwise_loss_func(y_true, y_pred):
    batch, n_src, *_ = y_true.shape
    return torch.randn(batch, n_src, n_src)


@pytest.fixture()
def targets():
    return torch.randn(10, 2, 32000)


@pytest.fixture()
def est_targets():
    return torch.randn(10, 2, 32000)


def test_wrapper(targets, est_targets):
    for bad_loss_func in [bad_loss_func_ndim0, bad_loss_func_ndim1]:
        loss = PITLossWrapper(bad_loss_func)
        with pytest.raises(AssertionError):
            loss(targets, est_targets)
    # wo_src loss function / With and without return estimates
    loss = PITLossWrapper(good_batch_loss_func, mode='wo_src')
    loss_value = loss(targets, est_targets)
    loss_value, reordered_est = loss(targets, est_targets, return_est=True)
    assert reordered_est.shape == est_targets.shape

    # pairwise loss function / With and without return estimates
    loss = PITLossWrapper(good_pairwise_loss_func, mode='pairwise')
    loss_value = loss(targets, est_targets)
    loss_value, reordered_est = loss(targets, est_targets, return_est=True)
    assert reordered_est.shape == est_targets.shape


def test_mse(targets, est_targets):
    pw_wrapper = PITLossWrapper(pairwise_mse, mode='pairwise')
    batch_wrapper = PITLossWrapper(nosrc_mse, mode='wo_src')
    assert_allclose(pw_wrapper(targets, est_targets),
                    batch_wrapper(targets, est_targets))


def test_sisdr():
    targets = torch.randn(10, 2, 32000)
    est_targets = torch.randn(10, 2, 32000)
    pw_wrapper = PITLossWrapper(pairwise_neg_sisdr, mode='pairwise')
    batch_wrapper = PITLossWrapper(nosrc_neg_sisdr, mode='wo_src')

    pw = pw_wrapper(targets, est_targets)
    ba = batch_wrapper(targets, est_targets)

    assert_allclose(pw_wrapper(targets, est_targets),
                    batch_wrapper(targets, est_targets))
