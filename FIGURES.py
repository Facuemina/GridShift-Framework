# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 12:02:54 2026

@author: Facundo
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-30,30,1000)
s = 5
d = 5
g0 = np.exp(-(x/s)**2*.5)
gr = np.exp(-((x-d)/s)**2*.5)
gl = np.exp(-((x+d)/s)**2*.5)

Gsum0R = np.exp(-((x-d/4 + d/2)/(1.5*s))**2*.5)
GsumR = np.exp(-((x-d/1.5+ d/2)/(1.5*s))**2*.5)
Gsum0L = np.exp(-((x-d/4)/(1.5*s))**2*.5)
GsumL = np.exp(-((x-d/1.5)/(1.5*s))**2*.5)

ar_R = 1
al_R = .3

ar_L = al_R + 0
al_L = ar_R + 0

gain_r = 1.6

I0 = 1

I = .15

plt.subplot(221)
plt.plot(x,ar_R * gr,'--b')
plt.plot(x, al_R * gl, '--r')
plt.plot(x,gain_r * ar_R * gr - I, '-b')
plt.plot(x, al_R * gl - I, '-r')
plt.plot([-.3,2],'--',color='gray')
plt.xticks([0])
plt.yticks([0])
plt.xlabel('y')
plt.ylim([-.3,1.7])
plt.xlim([-29,29])
plt.ylabel('Inputs (a.u.)')

plt.subplot(222)
plt.plot(x,ar_L * gr,'--b')
plt.plot(x, al_L * gl, '--r')
plt.plot(x,gain_r * ar_L * gr - I, '-b')
plt.plot(x, al_L * gl - I, '-r')
plt.plot([-.3,2],'--',color='gray')
plt.xticks([0])
plt.yticks([0])
plt.xlabel('y')
plt.ylim([-.3,1.7])
plt.xlim([-29,29])
plt.ylabel('Inputs (a.u.)')

plt.subplot(224)
plt.plot(x,np.maximum(Gsum0R-I,0),'--',color='gray')
plt.plot(x,np.maximum(GsumR-I,0), '-',color='gray')
plt.plot([-2,2],'--',color='k')
plt.xlabel('y')
plt.ylim([-.1,1])
plt.xlim([-29,29])
plt.xticks([0])
plt.yticks([0])
plt.ylabel('Firing rate (a.u.)')

plt.subplot(223)
plt.plot(x,np.maximum(Gsum0L-I,0),'--',color='gray')
plt.plot(x,np.maximum(GsumL-I,0), '-',color='gray')
plt.plot([-2,2],'--',color='k')
plt.xlabel('y')
plt.ylim([-.1,1])
plt.xticks([0])
plt.yticks([0])
plt.xlim([-29,29])
plt.ylabel('Firing rate (a.u.)')

plt.tight_layout()

plt.savefig('gaussian_scheme.svg')