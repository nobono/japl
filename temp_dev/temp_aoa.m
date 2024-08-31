format long
digits(20)

AeroData.Sref = 0.05;
alphaTotal = 0.021285852300486;
alphaMax = 80;
alt = 0.097541062161326;
am_c = 4.848711297583447;
cg = NaN;
mass = 3.040822554083198e+02;
mach = 0.020890665777000;
qBar = 30.953815676156566;
thrust = 50000;

CT = 0.236450041229858;
CA = 0.400000000000000;
CN_alpha = 0.140346623943120;
CL = 0.224556527015915;
CL_alpha = 0.133185708253379;
CD_alpha = 0.008056236784475;

% CA_alpha = 0;
% CD = 0.406796003141811;

aoa = getAoA(AeroData, alphaTotal, alphaMax, alt, am_c, cg,...
    mass, mach, qBar, thrust,...
    CT, CA, CN_alpha, CL, CL_alpha, CD_alpha);

out = 1.6893924177952654997;
assert(aoa == out);
disp("pass")