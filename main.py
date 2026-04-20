import json
import os
import torch
import numpy as np
import pandas as pd
import optuna
try:
    from text.src.evaluation import evaluate_classification
    from text.src.models import BERTClassifier
    from text.src.training import get_bert_dataloaders
    from text.src.training import train_bert_classifier
    from text.src.constants import MODEL_NAME, classes_dict
except ImportError:
    from src.evaluation import evaluate_classification
    from src.models import BERTClassifier
    from src.training import get_bert_dataloaders
    from src.training import train_bert_classifier
    from src.constants import MODEL_NAME, classes_dict
np.random.seed(42)
torch.manual_seed(42)


reversed_classes_dict = {v: k for k, v in classes_dict.items()}
"""
use: nohup uv run main.py > output.log 2>&1 &
"""
if __name__ == "__main__":
    json_path = os.path.join("data", "labeled_text_data.json")
    # json_path = "data/final_dataset.json" #here we need to have labeled data for training
    #json_path = "data/final_dataset.json" #here we need to have labeled data for training
    # BERT-specific parameters
    batch_size = 48
    epochs = 30
    learning_rate = 2e-5
    num_classes = 3
    freeze_until_layer = 7
    train_mode = True
    run_hpo = False
    hpo_trials = 4
    
    custom_bert_path = "./finetuned_bert_mlm" # our path to ssl pretrained model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Używane urządzenie: {device}")

    def objective(trial):
        
        print(f"Trial {trial.number + 1}/{hpo_trials}")
        
        trial_epochs = trial.suggest_int("epochs", 8, 15)
        trial_learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4)
        trial_dropout = trial.suggest_float("dropout_prob", 0.01, 0.2)
        trial_freeze_until = trial.suggest_int("freeze_until_layer", 4, 8)

        dataloader_train, dataloader_test = get_bert_dataloaders(
            json_path,
            batch_size=batch_size,
            model_path=custom_bert_path,
        )

        model = BERTClassifier(
            num_classes=num_classes,
            dropout_prob=trial_dropout,
            pretrain_path=custom_bert_path,
        )

        trained_model = train_bert_classifier(
            model,
            dataloader_train,
            dataloader_test,
            epochs=trial_epochs,
            learning_rate=trial_learning_rate,
            freeze_until_layer=trial_freeze_until,
        )

        trained_model.eval()
        metrics = evaluate_classification(dataloader_test, trained_model)
        return metrics["f1"]

    if run_hpo:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=hpo_trials)

        print("\nNajlepsze parametry z Optuna:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")

        print(f"Najlepszy wynik f1: {study.best_value:.4f}")

        best_params = study.best_params
        epochs = best_params["epochs"]
        learning_rate = best_params["learning_rate"]
        freeze_until_layer = best_params["freeze_until_layer"]
        best_dropout = best_params["dropout_prob"]
    else:
        best_dropout = 0.2

    dataloader_train, dataloader_test = get_bert_dataloaders(
        json_path,
        batch_size=batch_size,
        model_path=custom_bert_path,
    )

    model_name = f"{MODEL_NAME}_{epochs}_epochs.pth"
    
    if train_mode:
        model = BERTClassifier(
            num_classes=num_classes,
            dropout_prob=best_dropout,
            pretrain_path=custom_bert_path,
        )

        trained_model = train_bert_classifier(
            model, 
            dataloader_train, 
            dataloader_test,
            epochs=epochs, 
            learning_rate=learning_rate,
            freeze_until_layer=freeze_until_layer
        )
        
        trained_model_cpu = trained_model.to(torch.device("cpu"))
        os.makedirs("models", exist_ok=True)
        torch.save(
            trained_model_cpu.state_dict(),
            os.path.join("models", model_name),
        )
        trained_model = trained_model.to(device)
    else:
        
        trained_model = BERTClassifier(num_classes=num_classes, dropout_prob=0.3)
        state = torch.load(
            os.path.join("models", model_name), map_location=device
        )
        trained_model.load_state_dict(state)
        trained_model = trained_model.to(device)

    trained_model.eval()
    metrics = evaluate_classification(dataloader_test, trained_model)

    print("\nMetryki końcowe (BERT):")
    for metric, value in metrics.items():
        if metric == "confusion_matrix":
            print(f"{metric}: \n{value}")
        else:
            print(f"{metric}: {value:.4f}")