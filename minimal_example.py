#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:58:31 2023

@author: eibarragp
"""
# Regular python packages
import numpy as np
from scipy.ndimage import gaussian_filter as gsf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Structural complexity module 
import Structural_complexity_opensource as SC

####################### Example of symmetries #################################
def symmetry_example():
    Lsize = 10
    example_sym = np.zeros((Lsize,Lsize))
    example_sym[3,2] = 1
    example_sym = example_sym.flatten()
    
    fig,axs = plt.subplots(2,4,figsize=(8,4),sharex=True,sharey=True)
    plt.suptitle("Example of symmetries",fontsize=14,y=0.97)
    
    for j in range(0,4):
        axs[0,j].imshow(SC.square_symmetry(example_sym,"rot",j,Lsize),origin="lower")
        axs[0,j].set_title("rot k=%s"%j)
        axs[1,j].imshow(SC.square_symmetry(example_sym,"ref",j,Lsize),origin="lower")
        axs[1,j].set_title("ref k=%s"%j)
    
    for i in range(0,2):
        for j in range(0,4):
            axs[i,j].axhline(4.5)
            axs[i,j].axvline(4.5)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
            
    return None

####################### Example of extract region #############################
def extraction_example():
    Lwindow = 20
    center = 50
    
    def gauss_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    
    x = np.linspace(0,100,100)
    y = np.linspace(0,100,100)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    z = gauss_2d(x, y,center,center,10,10)
    
    # Select region to work with
    locator = center-Lwindow//2
    z2 = SC.extract_region(z, Lwindow, locator)
    
    fig,axs = plt.subplots(1,2,figsize=(8,4))
    plt.suptitle("Example of extraction",fontsize=14,y=0.97)
    
    axs[0].imshow(z,origin="lower")
    axs[0].set_title("Original image")
    
    axs[1].imshow(z2,origin="lower")
    axs[1].set_title("Extracted region")
    
    for i in range(0,1):
        rect = patches.Rectangle((locator,locator),Lwindow,Lwindow, linewidth=2, edgecolor='black', facecolor='none')
        axs[i].add_patch(rect)
    plt.show()
    
    return None

####################### Structural complexity example #########################
def SC_example(case):
    """
    case 1 : random +-1 spins
    case 2 : perfect AFM ordering (+-1 spins)
    case 3 : a maze-looking image
    """
    Lsize = 20
    Lcoarse=2
    numsamples = 300
    
    kinit = 0
    detailed = True
    example = True
    
    def random_spin():
        spin = np.random.randint(2,size=Lsize**2)
        spin = 2*spin - 1
        return spin
    
    def AFM():
        spin = np.zeros((Lsize,Lsize)) + 1
        for x in range(Lsize):
            for y in range(Lsize):
                spin[x,y] = (-1)**(x+y)*spin[x,y]
        return spin
    
    # Generate some test data
    if case == 1:
        test_data = np.vstack([random_spin() for j in range(0,numsamples)])
    elif case ==2:
        test_data = np.vstack([AFM().flatten() for j in range(0,numsamples)])
    elif case ==3:
        test_data = np.vstack([gsf(2*np.eye(Lsize,Lsize),1).flatten() for j in range(0,numsamples)])
    
    # Compute structural complexity
    kMax,SC0,SC1,Dks,ks = SC.analyzer(test_data,Lsize,Lcoarse,kinit,detailed,example)
    
    print("")
    print("SC0 = %1.6f"%SC0)
    print("SC1 = %1.6f"%SC1)
    
    plt.figure()
    plt.title("Example of structural complexity",fontsize=14)
    plt.xlabel("k")
    plt.ylabel("Dk")
    plt.errorbar(ks,Dks,fmt="o",color="red",ls="-")

    return None

################### Uncomment to run examples #################################

# plt.close("all")
# symmetry_example()
# extraction_example() 
# for case in range(1,4):
#     print("")
#     print("Case #%s"%case)
#     SC_example(case)
