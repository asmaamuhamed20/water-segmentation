from segmentation_models_pytorch import Unet

def get_model():
    model = Unet(
        encoder_name="efficientnet-b0",
        encoder_weights=None,
        in_channels=12,
        classes=1
    )
    return model
