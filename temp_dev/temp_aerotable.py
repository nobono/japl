import os
import numpy as np
from japl import AeroTable
from japl.Aero.AeroTable import DataTable
DIR = os.path.dirname(__file__)


units = "si"

# aerotable = AeroTable(f"{DIR}/../aeromodel/aeromodel_psb.mat")
aerotable = AeroTable(f"{DIR}/../aeromodel/cms_sr_stage1aero.mat",
                      from_template="CMS",
                      units=units)

pi = np.pi
cos = np.cos
sin = np.sin
deg2rad = np.deg2rad(1)


def cosd(deg):
    return np.cos(np.radians(deg))


def sind(deg):
    return np.sin(np.radians(deg))


####################################
alphaTotal = 0.021285852300486 # * np.radians(1)
alphaMax = 80 # * np.radians(1)
alt = 0.097541062161326
am_c = 4.848711297583447
cg = np.nan
mass = 3.040822554083198e+02
mach = 0.020890665777000
qBar = 30.953815676156566
thrust = 50_000.0
Sref = 0.05

CT = 0.236450041229858
CA = 0.400000000000000
CN_alpha = 0.140346623943120 #* np.radians(1)
CL = 0.224556527015915
CD = 0.406796003141811
CA_alpha = 0.0
CL_alpha = 0.133185708253379 #* np.radians(1)
CD_alpha = 0.008056236784475 #* np.radians(1)

#######
alpha = 1.689147711404596 #* np.radians(1)

alpha_tol = .01 #* np.radians(1)

ca = CA
cn = CT
ca_alpha = CA_alpha
cn_alpha = CN_alpha
cl = CL
cd = CD
cl_alpha = CL_alpha
cd_alpha = CD_alpha

if units == "si":
    alphaTotal *= np.radians(1)
    alphaMax *= np.radians(1)
    CT *= np.radians(1)
    CA *= np.radians(1)
    CN_alpha *= np.radians(1)
    CL *= np.radians(1)
    CD *= np.radians(1)
    CA_alpha *= np.radians(1)
    CL_alpha *= np.radians(1)
    CD_alpha *= np.radians(1)
    alpha *= np.radians(1)
    alpha_tol *= np.radians(1)

# ca = aerotable.get_CA_Boost(alpha=alpha, mach=mach, alt=alt)
# cn = aerotable.get_CNB(alpha=alpha, mach=mach)
# ca_alpha = aerotable.get_CA_Boost_alpha(alpha=alpha, mach=mach, alt=alt)
# cn_alpha = CN_alpha

# cosa = cos(alpha)
# sina = sin(alpha)
# cl = (cn * cosa) - (ca * sina)
# cd = (cn * sina) + (ca * cosa)
# cl_alpha = ((cn_alpha - (ca * deg2rad)) * cosa) - ((ca_alpha + (cn * deg2rad)) * sina)
# cd_alpha = ((ca_alpha + (cn * deg2rad)) * cosa) + ((cn_alpha - (ca * deg2rad)) * sina)

# print(ca - CA)
# print(cn - CT)
# print(ca_alpha - CA_alpha)
# print(cn_alpha - CN_alpha)
# print(cl_alpha - CL_alpha)
# print(cd_alpha - CD_alpha)
# quit()
####################################

# Alpha tolerance (deg)
# alpha_tol = 1e-16
# Initialize angle of attack
alpha_0 = alphaTotal
# Last alpha
alpha_last = -1000
# Interation counter
cnt = 0
# Gradient search - fixed number of steps
while ((abs(alpha_0 - alpha_last) > alpha_tol) and (cnt < 10)):
    # Update iteration counter
    cnt = cnt + 1
    # Update last alpha
    alpha_last = alpha_0
    # Get derivative of cn wrt alpha
    # (CT,
    #  CA,
    #  CN_alphl,
    #  CA_alpha,
    #  CL,
    #  CD,
    #  CL_alpha,
    #  CD_alpha) = getAeroCoeffs(AeroData, mach, alpha_0, cg, alt, thrust)

    # Get derivative of missile acceleration normal to flight path wrt alpha
    d_alpha = alpha_0 - alphaTotal

    if units == "si":
        am_alpha = cl_alpha * qBar * Sref / mass + thrust * cos(alpha_0) / mass
        am_0 = cl * qBar * Sref / mass + thrust * sin(alpha_0) / mass
    else:
        am_alpha = cl_alpha * qBar * Sref / mass + thrust * cosd(alpha_0) * pi / 180 / mass
        am_0 = cl * qBar * Sref / mass + thrust * sind(alpha_0) / mass

    # Update angle of attack
    alpha_0 = alpha_0 + (am_c - am_0) / am_alpha
    # Bound angle of attack
    alpha_0 = max(0, min(alpha_0, alphaMax))

# Set output
aoa = alpha_0  # * np.degrees(1)
out = 1.6893924177952654997

if units == "si":
    aoa *= np.degrees(1)

print()
print("-" * 50)
print(f"aoa: {aoa}")
print(f"tru: {out}")
print(f"count: {cnt}")
print(f"alpha error: {abs(alpha_0 - alpha_last)}")
print(f"diff: {out - aoa}")
print("-" * 50)
print()
# assert (aoa == out)
# print("pass")
print("-" * 50)

quit()

alpha_id = 2
phi_id = 0
mach_id = 0
iota_id = 3
neg_iota_id = 5

alpha = aerotable.increments.alpha[alpha_id]
phi = aerotable.increments.phi[phi_id]
mach = aerotable.increments.mach[mach_id]
iota = aerotable.increments.iota[iota_id]

cait = aerotable._get_CNB_IT(alpha, phi, mach, iota)
neg_cait = aerotable._get_CNB_IT(-alpha, phi, mach, iota)
print(cait, neg_cait)

cait = aerotable._CNB_IT[alpha_id, phi_id, mach_id, iota_id]
neg_cait = -aerotable._CNB_IT[alpha_id, phi_id, mach_id, neg_iota_id]
print(cait, neg_cait)

alpha_mirr = aerotable.create_mirrored_array(aerotable.increments.alpha)
# CNB_IT = np.concatenate([-aerotable._CNB_IT[::-1][:-1, :, :], aerotable._CNB_IT], axis=0)
# # cait = CNB_IT[alpha_id, phi_id, mach_id, iota_id]
CNB_IT = aerotable.create_mirrored_table(aerotable._CNB_IT, axis=0)
p = interpn((alpha_mirr,
             aerotable.increments.phi,
             aerotable.increments.mach,
             aerotable.increments.iota),
            CNB_IT,
            [alpha, phi, mach, iota],
            method="linear")[0]

neg_p = interpn((alpha_mirr,
                 aerotable.increments.phi,
                 aerotable.increments.mach,
                 aerotable.increments.iota),
                CNB_IT,
                [-alpha, phi, mach, -iota],
                method="linear")[0]

# cait = p(alpha, phi, mach, iota)
# neg_cait = p(-alpha, phi, mach, iota)
# neg_cait = -CNB_IT[alpha_id, phi_id, mach_id, neg_iota_id]
print(p, neg_p)
