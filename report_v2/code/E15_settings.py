"""E15. Settings of the prediction models (Appendix E).

Writes the settings that models.py fixes, with the XGBoost settings that the validation split chose
for seed 0 in E8 and the GRU epochs chosen by early stopping, so that Appendix E can be written from
one file.
"""
import common as C
import models as M

SEEDS = {}


def main():
    e8, e9 = C.read_json('E08'), C.read_json('E09')
    out = {'settings': M.settings(), 'xgboost_chosen_seed0': e8['xgboost_chosen_seed0'],
           'gru_epochs': e8['gru_epochs'], 'surface_krr': e9['krr_params'], 'gamma_bootstrap': 200,
           'folds': {'E8': 'leave one alpha out (3 folds)', 'E10': 'leave one cell out (48 cells)'}}
    C.write_json('E15', out, SEEDS)
    ep = sorted(e8['gru_epochs'].values())
    print('gru epochs', ep)


if __name__ == '__main__':
    main()
