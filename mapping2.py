import torch

# LIVE 2 param
#brisuqe
A1 = 10.0
B1 = 0.0
C1 = 57.4975833323062
S1 = 24.142607403486405

#cornia
A2 = 10.0
B2 = 0.0
C2 = 58.370096903220310
S2 = 23.807663946360070

#LFC
A3 = 10.0
B3 = 0.0
C3 = 2.6943240
S3 = 0.9155527

#UNIQUE
A4 = 10.0
B4 = 0.0
C4 = -0.4135264
S4 = 1.6893845
#C4 = 2.5863140
#S4 = 1.6897490




def logistic_mapping(x, model):
    if model == 0:
        A = A1
        B = B1
        C = C1
        S = S1
    elif model == 1:
        A = A2
        B = B2
        C = C2
        S = S2
    elif model == 2:
        A = A2
        B = B2
        C = C2
        S = S2
    elif model == 3:
        A = A3
        B = B3
        C = C3
        S = S3

    z = (x - C) / S
    return (A - B) / (1 + torch.exp(-z)) + B