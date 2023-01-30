import numpy as np

def IntLagrange(x,xx):
    Lw = np.ones(x.shape[0])
    for j in range(x.shape[0]):
        for m in range(x.shape[0]):
            if j != m:
                Lw[j] *= (xx - x[m])# / (x[j] - x[m])
    return Lw

def Lagrange(x,xx):
    L = np.ones(x.shape[0])
    for j in range(x.shape[0]):
        for m in range(x.shape[0]):
            if j != m:
                L[j] *= (xx - x[m]) / (x[j] - x[m])
    return L

def dLagrange(x,xx):
    Lp = np.zeros(x.shape[0])
    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if i != j:
                prod = 1
                for l in range(x.shape[0]):
                    if l != i and l != j:
                        prod *= (xx - x[l]) / (x[j] - x[l])
                Lp[j] += prod/(x[j] - x[i])
    return Lp

def d2Lagrange(x,xx):
    Lpp = np.zeros(x.shape[0])
    for j in range(x.shape[0]):
        for i in range(x.shape[0]):
            if i != j:
                summ = 0
                for m in range(x.shape[0]):
                    if m != i and m != j:
                        prod = 1
                        for l in range(x.shape[0]):
                            if l != i and l != j and l != m:
                                prod *= (xx - x[l]) / (x[j] - x[l])
                        summ += prod/(x[j] - x[m])
                Lpp[j] += summ/(x[j] - x[i])
    return Lpp

def d3Lagrange(x,xx):
    Lppp = np.zeros(x.shape[0])
    for k in range(x.shape[0]):
        for j in range(x.shape[0]):
            for i in range(x.shape[0]):
                if i != j:
                    summ = 0
                    for m in range(x.shape[0]):
                        if m != i and m != j:
                            prod = 1
                            for l in range(x.shape[0]):
                                if l != i and l != j and l != m:
                                    prod *= (xx - x[l]) / (x[j] - x[l])
                            summ += prod/(x[j] - x[m])
                    Lppp[j] += summ/(x[j] - x[i])
    return Lppp

def d4Lagrange(x,xx):
    Lpppp = np.zeros(x.shape[0])
    for h in range(x.shape[0]):
        for k in range(x.shape[0]):
            for j in range(x.shape[0]):
                for i in range(x.shape[0]):
                    if i != j:
                        summ = 0
                        for m in range(x.shape[0]):
                            if m != i and m != j:
                                prod = 1
                                for l in range(x.shape[0]):
                                    if l != i and l != j and l != m:
                                        prod *= (xx - x[l]) / (x[j] - x[l])
                                summ += prod/(x[j] - x[m])
                        Lpppp[j] += summ/(x[j] - x[i])
    return Lpppp
