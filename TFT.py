import os
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.tuner import Tuner

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MultiHorizonMetric
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer, EncoderNormalizer

# Fix PyTorch 2.6 default weights_only=True unpickling error during Lightning checkpoint restoration
_original_torch_load = torch.load
def _custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _custom_torch_load

# Set seed for reproducibility
pl.seed_everything(42)


# ==========================================
# CUSTOM CALLBACK TO LOG LOSS EVERY EPOCH
# ==========================================
class EpochLossLogger(Callback):
    """
    Custom callback to capture and print Train and Validation losses 
    to standard output at the end of every epoch.
    """
    def __init__(self):
        super().__init__()
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip initial sanity check run before training starts
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics

        # Retrieve logged loss values from trainer metrics
        val_loss = metrics.get("val_loss")
        train_loss = metrics.get("train_loss") or metrics.get("train_loss_epoch")

        val_str = f"{val_loss.item():.6f}" if val_loss is not None else "N/A"
        train_str = f"{train_loss.item():.6f}" if train_loss is not None else "N/A"

        # Record to internal history list
        self.history.append({
            "epoch": epoch,
            "train_loss": train_loss.item() if train_loss is not None else None,
            "val_loss": val_loss.item() if val_loss is not None else None
        })

        # Print structured log directly to terminal output
        print(f"--> [Epoch {epoch:02d}/{trainer.max_epochs}] | Train Loss: {train_str} | Val Loss: {val_str}")


# ==========================================
# 1. LOAD DATASET & PREPARE TYPES
# ==========================================
def load_and_prep_data(filepath="nifty100_tft_engineered_dataset.csv"):
    print(f"Loading dataset from '{filepath}'...")
    df = pd.read_csv(filepath)

    categorical_cols = [
        "symbol", "sector", "day_of_week", "month", 
        "quarter", "is_fno_expiry", "is_month_end", "is_union_budget_month"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    df = df.sort_values(by=["symbol", "time_idx"]).reset_index(drop=True)
    return df


# ==========================================
# 2. MAIN TRAINING PIPELINE
# ==========================================
def main():
    df = load_and_prep_data()

    max_encoder_length = 60    # 60 trading days lookback
    max_prediction_length = 5   # Forecast 5 trading days ahead

    max_time_idx = df["time_idx"].max()
    val_cutoff = max_time_idx - 20
    training_cutoff = val_cutoff - max_prediction_length

    print(f"Total time steps: {max_time_idx}")
    print(f"Training time range: 0 to {training_cutoff}")
    print(f"Validation time range: {training_cutoff + 1} to {max_time_idx}\n")

    # ----------------------------------------------------
    # Build TimeSeriesDataSets
    # ----------------------------------------------------
    training_data = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="log_return",
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,
        
        static_categoricals=["symbol", "sector"],
        time_varying_known_categoricals=[
            "day_of_week", "month", "quarter", 
            "is_fno_expiry", "is_month_end", "is_union_budget_month"
        ],
        time_varying_known_reals=["time_idx", "day_of_month"],
        time_varying_unknown_reals=[
            "log_return", "log_ret_5d", "log_ret_21d",
            "volatility_10d", "volatility_20d", "vol_ratio_10_20",
            "excess_return_1d", "excess_return_5d", "beta_60d",
            "rsi_14", "rsi_change_3d", "volume_ratio", "volume_surge",
            "nifty_return"
        ],
        
        target_normalizer=GroupNormalizer(groups=["symbol"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_data = TimeSeriesDataSet.from_dataset(
        training_data, 
        df, 
        predict=True, 
        stop_randomization=True
    )

    batch_size = 128
    train_dataloader = training_data.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0, pin_memory=False
    )
    val_dataloader = validation_data.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0, pin_memory=False
    )

    # ----------------------------------------------------
    # Initialize TFT Model
    # ----------------------------------------------------
    tft = TemporalFusionTransformer.from_dataset(
        training_data,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=16,
        loss=QuantileLoss([0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=4,
        log_interval=10,  # Ensure frequent step logging
    )

    # ----------------------------------------------------
    # Setup Callbacks & Loggers
    # ----------------------------------------------------
    epoch_logger = EpochLossLogger()
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=1e-4, 
        patience=10, 
        verbose=True, 
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="tft_checkpoints",
        filename="tft-nifty-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min"
    )
    
    lr_logger = LearningRateMonitor()

    # Dual Logging: TensorBoard for charts, CSVLogger for recorded loss history file
    tb_logger = TensorBoardLogger("tb_logs", name="nifty_tft_model")
    csv_logger = CSVLogger("csv_logs", name="nifty_tft_model")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[
            early_stop_callback, 
            checkpoint_callback, 
            lr_logger, 
            epoch_logger  # Direct console print callback
        ],
        logger=[tb_logger, csv_logger]
    )

    # ----------------------------------------------------
    # Find Optimal Learning Rate & Train
    # ----------------------------------------------------
    print("--- Finding Optimal Learning Rate ---")
    tuner = Tuner(trainer)
    res = tuner.lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )
    new_lr = res.suggestion()
    print(f"Optimal Learning Rate: {new_lr:.6f}\n")
    tft.hparams.learning_rate = new_lr

    print("================ STARTING TRAINING ================")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # Save epoch loss log to a CSV file directly
    history_df = pd.DataFrame(epoch_logger.history)
    history_df.to_csv("epoch_loss_history.csv", index=False)

    print("\n==================================================")
    print(" TRAINING COMPLETE!")
    print(f" Best Checkpoint: {checkpoint_callback.best_model_path}")
    print(f" Recorded Epoch Loss History saved to 'epoch_loss_history.csv'")
    print("==================================================\n")

if __name__ == "__main__":
    main()