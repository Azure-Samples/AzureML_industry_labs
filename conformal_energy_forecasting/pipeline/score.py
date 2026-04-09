import os
import json
import glob
import torch
import pandas as pd


def init():
    global model, conformal_quantile, norm_stats, _header_written
    _header_written = False

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.quantile_forecaster import QuantileForecaster

    # Azure ML batch endpoint injects the model path via this env var
    model_path = os.environ.get("AZUREML_MODEL_DIR")

    # Load conformal config
    config_files = glob.glob(os.path.join(model_path, "**", "conformal_config.json"), recursive=True)
    if not config_files:
        raise FileNotFoundError(f"No conformal_config.json found in {model_path}")
    with open(config_files[0]) as f:
        config = json.load(f)
    conformal_quantile = config["conformal_quantile"]
    print(f"  Conformal quantile Q: {conformal_quantile:.6f}")

    # Load normalisation stats
    stats_files = glob.glob(os.path.join(model_path, "**", "norm_stats.json"), recursive=True)
    if not stats_files:
        raise FileNotFoundError(f"No norm_stats.json found in {model_path}")
    with open(stats_files[0]) as f:
        norm_stats = json.load(f)

    # Load model weights
    pt_files = glob.glob(os.path.join(model_path, "**", "best_model.pt"), recursive=True)
    if not pt_files:
        raise FileNotFoundError(f"No best_model.pt found in {model_path}")

    n_features = len(norm_stats["feat_mean"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = QuantileForecaster(n_features=n_features).to(device)
    m.load_state_dict(torch.load(pt_files[0], map_location=device, weights_only=True))
    m.eval()
    model = m
    print(f"✅ Model loaded from {pt_files[0]}")


def run(mini_batch):
    global _header_written
    results = []
    device = next(model.parameters()).device
    tgt_mean = norm_stats["tgt_mean"]
    tgt_std  = norm_stats["tgt_std"]

    # Emit CSV header on the first mini-batch
    if not _header_written:
        results.append({
            "timestamp":        "timestamp",
            "point_forecast":   "point_forecast",
            "raw_lower":        "raw_lower",
            "raw_upper":        "raw_upper",
            "conformal_lower":  "conformal_lower",
            "conformal_upper":  "conformal_upper",
            "interval_width":   "interval_width",
            "actual":           "actual",
        })
        _header_written = True

    for pt_path in mini_batch:
        if not str(pt_path).endswith(".pt"):
            continue
        try:
            data = torch.load(pt_path, weights_only=True)
            features  = data["features"].unsqueeze(0).to(device)
            timestamp = data.get("timestamp", os.path.basename(pt_path))

            with torch.no_grad():
                pred = model(features).cpu().numpy()[0]

            # Denormalise predictions back to MWh
            q_lower_raw = pred[0] * tgt_std + tgt_mean
            q_median    = pred[1] * tgt_std + tgt_mean
            q_upper_raw = pred[2] * tgt_std + tgt_mean

            # Apply conformal adjustment (in normalised space, then denormalise)
            conf_lower = (pred[0] - conformal_quantile) * tgt_std + tgt_mean
            conf_upper = (pred[2] + conformal_quantile) * tgt_std + tgt_mean

            # Denormalise actual target if present
            actual = None
            if "target" in data:
                actual = float(data["target"].item()) * tgt_std + tgt_mean

            results.append({
                "timestamp":        timestamp,
                "point_forecast":   round(float(q_median), 2),
                "raw_lower":        round(float(q_lower_raw), 2),
                "raw_upper":        round(float(q_upper_raw), 2),
                "conformal_lower":  round(float(conf_lower), 2),
                "conformal_upper":  round(float(conf_upper), 2),
                "interval_width":   round(float(conf_upper - conf_lower), 2),
                "actual":           round(actual, 2) if actual is not None else None,
            })
        except Exception as e:
            results.append({
                "timestamp":        os.path.basename(pt_path),
                "point_forecast":   None,
                "raw_lower":        None,
                "raw_upper":        None,
                "conformal_lower":  None,
                "conformal_upper":  None,
                "interval_width":   None,
                "actual":           None,
            })
            print(f"[WARN] Failed on {pt_path}: {e}")

    return pd.DataFrame(results)
