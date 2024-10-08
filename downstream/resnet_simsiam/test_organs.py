from medmnist import INFO
import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch.utils.data as data
import torch
from sklearn.metrics import accuracy_score
from args_parser import Arguments
from downloader import Downloader
from downstream.resnet.resnet_transferlm import ResNetTransferLM
from downstream.resnet.train import initialise_new_network
from utils import (
    NUM_WORKERS,
    RESNET_TRANSFER_CHECKPOINT_PATH,
    SEED,
    SplitType,
    get_accelerator_info,
    get_auroc_metric,
)


if __name__ == "__main__":
    (
        DATA_FLAG,
        MODEL_NAME,
    ) = Arguments.parse_args_test()

    # Seed
    pl.seed_everything(SEED)
    np.random.seed(SEED)

    # Load data
    downloader = Downloader()
    test_data = downloader.load(DATA_FLAG, SplitType.TEST)

    pretrained_path = os.path.join(
        RESNET_TRANSFER_CHECKPOINT_PATH,
        MODEL_NAME,
    )

    # Load model
    # Without this, getting errors due to missing backbone parameter
    resnet_base = initialise_new_network()

    resnet_base.fc = nn.Sequential(
        resnet_base.fc,
        nn.ReLU(inplace=True),
    )

    # Automatically load model with saved hyperparameters
    model = ResNetTransferLM.load_from_checkpoint(
        pretrained_path,
        backbone=resnet_base,
    )
    print("Model loaded")

    # Test

    test_loader = data.DataLoader(
        test_data,
        batch_size=min(128, len(test_data)),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    # Trainer
    accelerator, num_threads = get_accelerator_info()

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_threads,
    )

    # Do not require optional logging
    trainer.logger._default_hp_metric = None

    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    # Calculate per-class accuracy
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    per_class_acc = {}
    for class_index in range(len(INFO[DATA_FLAG]["label"])):
        class_mask = (all_labels == class_index)
        class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
        per_class_acc[INFO[DATA_FLAG]["label"][class_index]] = class_acc

    result = {
        "top-1 acc": test_result[0]["test_acc"],
        "auroc": get_auroc_metric(
            model,
            test_loader,
            num_classes=len(INFO[DATA_FLAG]["label"])
        ),
        "per_class_acc": per_class_acc
    }

    print(f"Model name: {MODEL_NAME}")
    print(f"Dataset: {DATA_FLAG}")
    print(result)
    
