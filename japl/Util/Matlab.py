import pickle
from scipy.io import loadmat


"""This is a temporary script for loading aeromodel data from .mat files
and saving them to a .pickle"""



__aero_data_path = ""

matfile = loadmat(__aero_data_path)

aeromodel_keys = [
        'Model',
        'Type',
        'Software',
        'Version',
        'Note_1',
        'Note_2',
        'Note_3',
        'Note_4',
        'Note_5',
        'Alpha',
        'Phi',
        'Mach',
        'Iota',
        'Iota_Prime',
        'Alt',
        'Alt_Units',
        'Lref_Units',
        'Lref',
        'Sref',
        'MRC',
        'CA_inv',
        'CA_Basic',
        'CA_0_Boost',
        'CA_0_Coast',
        'CNB',
        'CNB_IT',
        'CLMB',
        'CLMB_IT',
        'Fin2_CN',
        'Fin2_BM',
        'Fin2_HM',
        'Fin4_CN',
        'Fin4_BM',
        'Fin4_HM',
        ]

notes = [matfile[i][0] for i in [
        'Note_1',
        'Note_2',
        'Note_3',
        'Note_4',
        'Note_5',
    ]]

aeromodel = {
        "notes": notes,
        "increments": {
            "alpha": matfile["Alpha"].flatten(),
            "phi": matfile["Phi"].flatten(),
            "mach": matfile["Mach"].flatten(),
            "alt": matfile["Alt"].flatten(),
            "iota": matfile["Iota"].flatten(),
            "iota_prime": matfile["Iota_Prime"].flatten(),
            },
        "units": {
            "alt": matfile["Alt_Units"][0],
            "lref": matfile["Lref_Units"][0],
            },
        "lref": matfile["Lref"].flatten()[0],
        "sref": matfile["Sref"].flatten()[0],
        "mrc": matfile["MRC"].flatten()[0],
        "CA_inv": matfile["CA_inv"][0][0][0],
        "CA_Basic": matfile["CA_Basic"][0][0][0],
        "CA_0_Boost": matfile["CA_0_Boost"][0][0][0],
        "CA_0_Coast": matfile["CA_0_Coast"][0][0][0],
        "CNB": matfile["CNB"][0][0][0],
        "CNB_IT": matfile["CNB_IT"][0][0][0],
        "CLMB": matfile["CLMB"][0][0][0],
        "CLMB_IT": matfile["CLMB_IT"][0][0][0],
        "Fin2_CN": matfile["Fin2_CN"][0][0][0],
        "Fin2_BM": matfile["Fin2_BM"][0][0][0],
        "Fin2_HM": matfile["Fin2_HM"][0][0][0],
        "Fin4_CN": matfile["Fin4_CN"][0][0][0],
        "Fin4_BM": matfile["Fin4_BM"][0][0][0],
        "Fin4_HM": matfile["Fin4_HM"][0][0][0],
        }


if __name__ == "__main__":

    with open("aeromodel.pickle", "ab") as f:
        pickle.dump(aeromodel, f)


