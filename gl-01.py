import numpy as np
import matplotlib.pyplot as plt
from decimal import *
from scipy import special
import math
from numba import jit, prange



D = Decimal

alpha = D(0.355028053887817239)
beta = D(0.258819403792806798)
natancnost = D(10)**(-10)

stx = 10000
x_range = np.linspace(-20., 15., stx)
Ai_mcl = np.zeros(stx)
Bi_mcl = np.zeros(stx)
Ai_asi = np.zeros(stx)
Bi_asi = np.zeros(stx)
Ai_ref = np.zeros(stx)
Bi_ref = np.zeros(stx)
Ai_odv = np.zeros(stx)
Bi_odv = np.zeros(stx)
Ai_kom = np.zeros(stx)
Bi_kom = np.zeros(stx)


def mclaurin_Ai_Bi(x, natancnost):
    x = D(x)
    tocen = False
    Ai = D(0)
    Bi = D(0)
    aij = D(0)
    bij = D(0)
    j = D(1)
    f = 1
    g = x

    while not tocen:
        aij = alpha * f - beta * g
        bij = D(3).sqrt() * (alpha * f + beta * g)

        if np.abs(aij).compare(natancnost) < 1 and np.abs(bij).compare(natancnost) < 1:
            tocen = True

        f *= x**D(3) * (D(1) - D(2)/D(3)/j) / ((D(3)*j - D(1)) * (D(3)*j - D(2)))
        g *= x**D(3) * (D(1) - D(1)/D(3)/j) / ((D(3)*j - D(1)) * (D(3)*j + D(1)))

        Ai += aij
        Bi += bij

        j += D(1)

    return Ai, Bi


i = 0
for x in x_range:
    mcl = mclaurin_Ai_Bi(x, natancnost)
    Ai_mcl[i] = mcl[0]
    Bi_mcl[i] = mcl[1]
    i += 1


#plotanje Maclaurinova vrsta
'''
plt.plot(x_range, Ai_mcl, label = '$y = $Ai$(x)$')
plt.plot(x_range, Bi_mcl, label = '$y = $Bi$(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.ylim(-0.75, 0.75)

plt.title('Ai in Bi z Maclaurinovo vrsto')
plt.grid()
plt.legend()
plt.show()'''



def asimptotska_Ai_Bi(x, natancnost):
    x = D(x)
    ksi = (D(2)/D(3)) * (np.abs(D(x))**(D(3)/D(2)))
    j = D(1)
    #Ai_star = D(float('inf'))
    #Bi_star = D(float('inf'))
    Ai = D(0)
    Bi = D(0)

    if x.compare(D(0)) > -1:
        faktor_Ai = np.exp(-ksi) / (D(2) * D(np.pi)**D(0.5) * x**D(0.25))
        faktor_Bi = np.exp(ksi) / (D(np.pi)**D(0.5) * x**D(0.25))
        L = D(1)
        aij = faktor_Ai
        bij = faktor_Bi

        rai = faktor_Ai
        rbi = faktor_Bi

        while np.abs(rai).compare(D(1)) == -1 or np.abs(rbi).compare(D(1)) == -1:
            L *= (D(3)*j - D(1/2)) * (D(3)*j - D(3/2)) * (D(3)*j - D(5/2)) / (D(54) * j * (j-D(1/2)) * ksi)
            
            Ai += aij
            Bi += bij
            
            rai = faktor_Ai * D(-1)**j * L / aij
            rbi = faktor_Bi * L / bij

            aij = faktor_Ai * D(-1)**j * L
            bij = faktor_Bi * L

            j += D(1)

    
    if x.compare(D(0)) == -1:
        faktor_neg = D(1) / (D(math.pi)**D(0.5) * (-x)**D(0.25))
        P = D(1)
        Q = (D(2) + D(0.5)) * (D(1) + D(0.5)) / (D(54) * ksi)
        aineg = faktor_neg * (D(math.sin(ksi - D(math.pi/4))) * Q + D(math.cos(ksi - D(math.pi/4))) * P)
        bineg = faktor_neg * (-D(math.sin(ksi - D(math.pi/4))) * P + D(math.cos(ksi - D(math.pi/4))) * Q)

        razmerjeai = aineg
        razmerjebi = bineg

        while np.abs(razmerjeai).compare(D(1)) == -1 or np.abs(razmerjebi).compare(D(1)) == -1:
            P *= D(-1) * (D(6) * j - D(11/2)) * (D(6) * j - D(9/2)) * (D(6) * j - D(7/2)) * \
                (D(6) * j - D(5/2)) * (D(6) * j - D(3/2)) * (D(6) * j - D(1/2)) / \
                (D(54)**D(2) * D(2)*j * (D(2)*j - D(1)) * (D(2) * j - D(1/2)) * (D(2) * j - D(3/2)) * ksi**D(2))
            Q *= D(-1) * (D(6) * j + D(5/2)) * (D(6) * j + D(3/2)) * (D(6) * j + D(1/2)) * \
                (D(6) * j-+ D(1/2)) * (D(6) * j - D(3/2)) * (D(6) * j - D(5/2)) / \
                (D(54)**D(2) * D(2)*j * (D(2)*j + D(1)) * (D(2) * j + D(1/2)) * (D(2) * j - D(1/2)) * ksi**D(2))

            Ai += aineg
            Bi += bineg

            razmerjeai = faktor_neg * (D(math.sin(ksi - D(math.pi/4))) * Q + D(math.cos(ksi - D(math.pi/4))) * P) / aineg
            razmerjebi = faktor_neg * (-D(math.sin(ksi - D(math.pi/4))) * P + D(math.cos(ksi - D(math.pi/4))) * Q) / bineg
            
            aineg = faktor_neg * (D(math.sin(ksi - D(math.pi/4))) * Q + D(math.cos(ksi - D(math.pi/4))) * P)
            bineg = faktor_neg * (-D(math.sin(ksi - D(math.pi/4))) * P + D(math.cos(ksi - D(math.pi/4))) * Q)

            j += D(1)

    return Ai, Bi


#Racunanje asimptotske vrste
#x_range = np.linspace(-20., 5, 10000)

i = 0
for x in x_range:
    asi = asimptotska_Ai_Bi(x, natancnost)
    Ai_asi[i] = asi[0]
    Bi_asi[i] = asi[1]
    i += 1

'''
#Plotanje asimptotske vrste
plt.plot(x_range, Ai_asi, label = '$y = $Ai$(x)$')
plt.plot(x_range, Bi_asi, label = '$y = $Bi$(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.ylim(-0.75, 0.75)

plt.title('Ai in Bi z Asimptotsko vrsto')
plt.grid()
plt.legend()
plt.show()'''

#Računanje kombinirane funkcije
'''def Ai(x):
    if -6.66666666667 <= x < 5.20202020202:
        return Ai_mcl
    else:
        return Ai_asi

def Bi(x):
    if -6.66666666667 <= x < 5.20202020202:
        return Bi_mcl
    else:
        return Bi_asi'''


j = 0
for x in x_range:
    if -6.6 <= x < 5.2:
        Ai_kom[j] = Ai_mcl[j]
    else:
        Ai_kom[j] = Ai_asi[j]

    if -6.6 <= x < 8.4:
        Bi_kom[j] = Bi_mcl[j]
    else:
        Bi_kom[j] = Bi_asi[j] 

    j += 1




#Racunanje referencnih funkcij
Ai_ref, Ai_odv, Bi_ref, Bi_odv = special.airy(x_range)

'''
#Plotanje abs napak pa tega
plt.plot(x_range, np.abs(Ai_ref - Ai_kom), marker = '.', markersize = 1, linestyle = '', label = '$y = \delta$Ai$(x)$')
plt.plot(x_range, np.abs(Bi_ref - Bi_kom), marker = '.', markersize = 1, linestyle = '', label = '$y = \delta$Bi$(x)$')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.yscale('log')
#plt.ylim(-26,-6)

plt.title('Absolutna napaka kombinacije vrst v logaritemski skali')
plt.grid()
plt.legend()
plt.show()

'''
#Plotanje rel napak pa tega
plt.plot(x_range, np.abs((Ai_ref - Ai_kom)/Ai_ref), marker = '.', markersize = 1, linestyle = '', label = '$y = \delta$Ai$(x)/$Ai')
plt.plot(x_range, np.abs((Bi_ref - Bi_kom)/Bi_ref), marker = '.', markersize = 1, linestyle = '', label = '$y = \delta$Bi$(x)/$Bi')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.yscale('log')
#plt.ylim(top=3)

plt.title('Relativna napaka kombinacije vrst v logaritemski skali')
plt.grid()
plt.legend()
plt.show()