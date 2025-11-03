import argparse
import torch
import numpy as np
import copy

import random

from exp.exp_main import Exp_Main
import os


def main():
    parser = argparse.ArgumentParser()

    # ----- Basic Config -----
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='test_run')
    parser.add_argument('--model', type=str, default='ns_Transformer')

    # ----- Data Loader -----
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/illness/')
    parser.add_argument('--data_path', type=str, default='national_illness.csv')
    parser.add_argument('--features', type=str, default='MS')  # M: multivariate, S: univariate
    parser.add_argument('--target', type=str, default='% WEIGHTED ILI')
    parser.add_argument('--freq', type=str, default='w')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # ----- Forecasting Task -----
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=24)

    # ----- Model Config -----
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)#512
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.2629577329390419)#0.05
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', type=bool, default=False)
    parser.add_argument('--do_predict', type=bool, default=False)

    # ----- Optimization -----
    parser.add_argument('--num_workers', type=int, default=10)  # SAFEST for Windows
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.00009)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', type=bool, default=False)

    # ----- GPU -----
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    # ----- Random Seed -----
    parser.add_argument('--seed', type=int, default=2021)

    # ----- Non-Stationary Transformer Special Config -----
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128])
    parser.add_argument('--p_hidden_layers', type=int, default=2)

    parser.add_argument('--phase_mode', action='store_true',default=True)
    parser.add_argument('--period_k', type=int, default=52)
    args = parser.parse_args()

    # ----- Fix Seed -----
    fix_seed = args.seed
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    print('Args in experiment:')
    print(args)

    # ----- Start Experiment -----
    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.model_id}_{args.model}_{args.data}_ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.distil}_{args.des}_{ii}'

            # exp = Exp_Main(args)  # build model
            #
            # print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            # exp.train(setting)
# new
            k = args.period_k if args.period_k is not None else 52  # or the k you computed
            for phase in range(k):
                # 1) build a phase‑filtered dataset
                args_copy = copy.deepcopy(args)
                args_copy.model_id = f'{args.model_id}_phase{phase}'
                args_copy.phase_index = phase  # new flag you’ll read in Dataset_Custom

                exp = Exp_Main(args_copy)
                print(f'>>>>> start phase {phase}/{k - 1} : {args_copy.model_id}')
                exp.train(args_copy.model_id)
# new
            print(f'>>>>>>>testing : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.test(setting)

            if args.do_predict:
                print(f'>>>>>>>predicting : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                exp.predict(setting)

            torch.cuda.empty_cache()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)  # Critical for Windows safety
    main()
