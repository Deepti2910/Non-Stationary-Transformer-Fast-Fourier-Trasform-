import optuna
from run import main as train_main
import argparse
import copy

def objective(trial):
    # Default arguments
    args = argparse.Namespace(
        is_training=1,
        model_id='optuna_run',
        model='ns_Transformer',
        data='custom',
        root_path='./dataset/illness/',
        data_path='national_illness.csv',
        features='MS',
        target='% WEIGHTED ILI',
        freq='w',
        checkpoints='./checkpoints/',
        seq_len=trial.suggest_int('seq_len', 24, 96, step=24),
        label_len=trial.suggest_int('label_len', 24, 96, step=24),
        pred_len=trial.suggest_categorical('pred_len', [12, 24, 48]),
        enc_in=7, dec_in=7, c_out=7,
        d_model=trial.suggest_categorical('d_model', [128, 256, 512]),
        n_heads=trial.suggest_categorical('n_heads', [2, 4, 8]),
        e_layers=trial.suggest_int('e_layers', 1, 3),
        d_layers=trial.suggest_int('d_layers', 1, 2),
        d_ff=trial.suggest_categorical('d_ff', [512, 1024, 2048]),
        moving_avg=25,
        factor=trial.suggest_int('factor', 1, 5),
        distil=True,
        dropout=trial.suggest_float('dropout', 0.01, 0.3),
        embed='timeF',
        activation='gelu',
        output_attention=False,
        do_predict=False,
        num_workers=4,
        itr=1,
        train_epochs=10,
        batch_size=trial.suggest_categorical('batch_size', [4, 8, 16]),
        patience=5,
        learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        des='tune',
        loss='mse',
        lradj='type1',
        use_amp=False,
        use_gpu=False,
        gpu=0,
        use_multi_gpu=False,
        devices='0',
        seed=2021,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        phase_mode=True,
        period_k=trial.suggest_categorical('period_k', [None, 26, 52])
    )

    try:
        # Redirect output so we can read performance
        from exp.exp_main import Exp_Main
        exp = Exp_Main(args)
        setting = f"{args.model_id}_{trial.number}"
        mse, mae = exp.train(setting)
        return mse  # or mae depending on your goal
    except Exception as e:
        print(f"Trial failed: {e}")
        return float("inf")
if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print(f"  MSE: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
