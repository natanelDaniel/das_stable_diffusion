import torch
from src.models.cvae import CVAE


def test_cvae_forward_shapes():
    model = CVAE(latent_dim=64, patch_channels=32, patch_time=256,
                 enc_channels=(32, 64, 128), n_classes=9)
    model.train()
    x = torch.randn(4, 1, 32, 256)
    c = torch.zeros(4, 9); c[:, 2] = 1.0
    x_hat, mu, logvar = model(x, c)
    assert x_hat.shape == (4, 1, 32, 256), x_hat.shape
    assert mu.shape == (4, 64), mu.shape
    assert logvar.shape == (4, 64), logvar.shape


def test_cvae_generate():
    model = CVAE(latent_dim=64, patch_channels=32, patch_time=256,
                 enc_channels=(32, 64, 128), n_classes=9)
    c = torch.zeros(1, 9); c[0, 0] = 1.0
    samples = model.generate(c, n=8, device="cpu")
    assert samples.shape == (8, 1, 32, 256), samples.shape


def test_reparameterisation_is_stochastic_in_train():
    model = CVAE(latent_dim=64, patch_channels=32, patch_time=256)
    model.train()
    mu = torch.zeros(2, 64)
    logvar = torch.zeros(2, 64)
    z1 = model.reparameterise(mu, logvar)
    z2 = model.reparameterise(mu, logvar)
    assert not torch.allclose(z1, z2), "z should differ between draws in train mode"


def test_reparameterisation_is_deterministic_in_eval():
    model = CVAE(latent_dim=64, patch_channels=32, patch_time=256)
    model.eval()
    mu = torch.ones(2, 64) * 0.5
    logvar = torch.zeros(2, 64)
    z1 = model.reparameterise(mu, logvar)
    z2 = model.reparameterise(mu, logvar)
    assert torch.allclose(z1, z2), "z should equal mu in eval mode"
