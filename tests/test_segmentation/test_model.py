"""
Tests for PixelSight segmentation U-Net.
"""

import pytest
import torch

from pixelsight.models.segmentation.unet import (
    INPUT_BANDS,
    NUM_CLASSES,
    UNet,
    count_parameters,
    create_unet,
)


def test_model_creation():
    model = create_unet()

    assert isinstance(model, UNet)
    assert model.in_channels == 4
    assert model.num_classes == 7


def test_parameter_count():
    model = create_unet()

    parameters = count_parameters(model)

    assert parameters > 0
    assert parameters < 20_000_000


def test_forward_shape():
    model = create_unet()
    model.eval()

    x = torch.randn(
        2,
        INPUT_BANDS,
        128,
        128,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        2,
        NUM_CLASSES,
        128,
        128,
    )


def test_single_sample_forward():
    model = create_unet()
    model.eval()

    x = torch.randn(
        1,
        4,
        128,
        128,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 7, 128, 128)


def test_output_is_finite():
    model = create_unet()
    model.eval()

    x = torch.randn(
        1,
        4,
        128,
        128,
    )

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()


def test_wrong_number_of_channels():
    model = create_unet()
    model.eval()

    x = torch.randn(
        1,
        3,
        128,
        128,
    )

    with pytest.raises(ValueError):
        model(x)


def test_wrong_input_dimensions():
    model = create_unet()
    model.eval()

    x = torch.randn(
        4,
        128,
        128,
    )

    with pytest.raises(ValueError):
        model(x)


def test_different_spatial_size():
    model = create_unet()
    model.eval()

    x = torch.randn(
        1,
        4,
        256,
        256,
    )

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1,
        7,
        256,
        256,
    )


def test_custom_model_configuration():
    model = create_unet(
        in_channels=4,
        num_classes=7,
        base_channels=16,
    )

    x = torch.randn(
        1,
        4,
        128,
        128,
    )

    model.eval()

    with torch.no_grad():
        output = model(x)

    assert output.shape == (
        1,
        7,
        128,
        128,
    )