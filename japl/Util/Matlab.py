import pickle
from scipy.io import loadmat
from astropy import units as u
import numpy as np



"""This is a temporary script for loading aeromodel data from .mat files
and saving them to a .pickle"""



__ft2m = (1.0 * u.imperial.foot).to_value(u.m) #type:ignore
__inch2m = (1.0 * u.imperial.inch).to_value(u.m) #type:ignore
__inch_sq2m_sq = (1.0 * u.imperial.inch**2).to_value(u.m**2) #type:ignore
__deg2rad = (np.pi / 180.0)
__lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2) #type:ignore


__aero_data_path = "/home/david/work_projects/control/aeromodel/aeromodel_bs.mat"
__aero_data_path_psb = "/home/david/work_projects/control/aeromodel/aeromodel_psb.mat"

matfile = loadmat(__aero_data_path)
matfile_psb = loadmat(__aero_data_path_psb)

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
        # 'Note_2',
        'Note_3',
        'Note_4',
        'Note_5',
    ]]
notes_psb = [matfile_psb[i][0] for i in [
        'Note_1',
        'Note_2',
        'Note_3',
        'Note_4',
        'Note_5',
    ]]

aeromodel = {
        "increments": {
            "alpha": matfile["Alpha"].flatten() * __deg2rad,    # (deg)
            "phi": matfile["Phi"].flatten() * __deg2rad,    # (deg)
            "mach": matfile["Mach"].flatten(),
            "alt": matfile["Alt"].flatten() * __ft2m,    # (ft.)
            "iota": matfile_psb["Iota"].flatten() * __deg2rad,  # (deg)
            "iota_prime": matfile_psb["Iota_Prime"].flatten() * __deg2rad,
            },
        "units": {
            "alt": matfile["Alt_Units"][0],     # (ft.)
            "lref": matfile["Lref_Units"][0],   # (in)
            },
        "lref": matfile["Lref"].flatten()[0] * __inch2m, # (in)
        "sref": matfile["Sref"].flatten()[0] * __inch_sq2m_sq, # (in^2)
        "mrc": matfile["MRC"].flatten()[0],
        "bs": {
            "notes": notes,
            "type": matfile["Type"][0],
            "CA_inv": matfile["CA_inv"][0][0][0],
            "CA_Basic": matfile["CA_Basic"][0][0][0],
            "CA_0_Boost": matfile["CA_0_Boost"][0][0][0],
            "CA_0_Coast": matfile["CA_0_Coast"][0][0][0],
            "CYB": matfile["CYB"][0][0][0],
            "CNB": matfile["CNB"][0][0][0],
            "CLLB": matfile["CLLB"][0][0][0],
            "CLMB": matfile["CLMB"][0][0][0],
            "CLNB": matfile["CLMB"][0][0][0],
            },
        "psb": {
            "CA_inv": matfile_psb["CA_inv"][0][0][0],           # (alpha, phi, mach)
            "CA_Basic": matfile_psb["CA_Basic"][0][0][0],       # (alpha, phi, mach)
            "CA_0_Boost": matfile_psb["CA_0_Boost"][0][0][0],   # (phi, mach, alt)
            "CA_0_Coast": matfile_psb["CA_0_Coast"][0][0][0],   # (phi, mach, alt)
            "CA_IT": matfile_psb["CA_IT"][0][0][0],             # (alpha, phi, mach, iota)
            "CYB_Basic": matfile_psb["CYB_Basic"][0][0][0],     # (alpha, phi, mach)
            "CYB_IT": matfile_psb["CYB_IT"][0][0][0],           # (alpha, phi, mach, iota)
            "CNB_Basic": matfile_psb["CNB_Basic"][0][0][0],     # (alpha, phi, mach)
            "CNB_IT": matfile_psb["CNB_IT"][0][0][0],           # (alpha, phi, mach, iota)
            "CLLB_Basic": matfile_psb["CLLB_Basic"][0][0][0],   # (alpha, phi, mach)
            "CLLB_IT": matfile_psb["CLLB_IT"][0][0][0],         # (alpha, phi, mach, iota)
            "CLMB_Basic": matfile_psb["CLMB_Basic"][0][0][0],   # (alpha, phi, mach)
            "CLMB_IT": matfile_psb["CLMB_IT"][0][0][0],         # (alpha, phi, mach, iota)
            "CLNB_Basic": matfile_psb["CLNB_Basic"][0][0][0],   # (alpha, phi, mach)
            "CLNB_IT": matfile_psb["CLNB_IT"][0][0][0],         # (alpha, phi, mach, iota)
            "Fin2_CN": matfile_psb["Fin2_CN"][0][0][0],         # (alpha, phi, mach, iota)
            "Fin2_CBM": matfile_psb["Fin2_CBM"][0][0][0],       # (alpha, phi, mach, iota)
            "Fin2_CHM": matfile_psb["Fin2_CHM"][0][0][0],       # (alpha, phi, mach, iota)
            "Fin4_CN": matfile_psb["Fin4_CN"][0][0][0],         # (alpha, phi, mach, iota)
            "Fin4_CBM": matfile_psb["Fin4_CBM"][0][0][0],       # (alpha, phi, mach, iota)
            "Fin4_CHM": matfile_psb["Fin4_CHM"][0][0][0],       # (alpha, phi, mach, iota)
            },
        }



if __name__ == "__main__":

    with open("aeromodel/aeromodel.pickle", "ab") as f:
        pickle.dump(aeromodel, f)


