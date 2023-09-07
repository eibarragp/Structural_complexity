"""
Created on Wed Aug 23 15:02:41 2023
@author: eibarragp

Open source structural complexity code for manuscript.
"""

import numpy as np
import matplotlib.pyplot as plt

###################### AUXILIARY FUNCTIONS ################3###################

def square_symmetry(data1d,sym,kval,Lsize):
    """
    Parameters
    ----------
    data1d: Single snapshot flattened array
    sym: Rotation or reflection symemtry to be applied
    kval: Type of rotation/reflection to be applied
    Lsize: Linear size of the array Lsize x Lsize

    Returns
    -------
    Lsize x Lsize array after one of the square lattice 
    point symmetries has been applied
    """
    
    # First reshape as square 
    matrix = data1d.reshape(Lsize,Lsize)
    
    def C4rotation(matrix,kval):
        # Apply a rotation: the identity and the allowed number of 90 deg rotations
        rot_matrix = np.rot90(matrix, k=kval, axes=(0, 1))
        return rot_matrix
    
    def reflection(matrix,kval):
        # Reflect horizontally, vertically, y=-x, or y=x
        if kval == 0:
            ref_matrix = np.flip(matrix, 0) # Horizonally
        elif kval == 1:
            ref_matrix = np.flip(matrix, 1) # Vertically 
        elif kval == 2:
            ref_matrix = np.flip(np.transpose(matrix)) # y=-x
        elif kval == 3:
            ref_matrix = np.flip(np.rot90(matrix, k=3, axes=(0, 1)),1) # y=x
        return ref_matrix
    
    if sym == "rot":
        return C4rotation(matrix,kval)
    elif sym == 'ref':
        return reflection(matrix,kval)
    
def extract_region(data,Lwindow,locator):
    """
    Parameters
    ----------
    data : Square numpy array
    Lwindow : Linear size of the window
    locator : Location where the window should be centered

    Returns
    ----------
    extracted : Square numpy array of size Lwindow x Lwindow
    """
    
    extracted = data[locator:locator+Lwindow,locator:locator+Lwindow]
    return extracted
    
def optimize_kMax(Lsize,sym_num_samples):
    """
    Parameters
    ----------
    Lsize : Linear size of the smaller array Lsize x Lsize
    sym_num_samples : Number of samples available (generally obtained after 
                     using the square lattice point symmetries)

    Returns
    ---------
    kMax : Maximum number of coarse graining steps
    Lbig : Linear size of the bigger teselated image
    
    Notes
    ---------
    Finds the combination to maximize the number of coarse graining steps
    
    x = Lsize
    y = sym_num_samples
    f(x,y) = output to maximize 
    """
    
    Lbig = int(np.sqrt(sym_num_samples))

    def f(x,y):
        n = 0 
        while x*y % 2**n == 0:
            n+=1
        return n-1
    
    Lbigs,array = [],[]
    for y in range(2,Lbig+1):
        Lbigs.append(y)
        array.append(f(Lsize,y))
    
    max_val = np.max(array)
    locations = np.array(np.where(array >= max_val)).flatten()
    
    array = [array[i] for i in locations]
    Lbigs = [Lbigs[i] for i in locations] 
    
    max_idx = np.argmax(Lbigs)
    kMax = array[max_idx]
    Lbig = Lbigs[max_idx]
    return kMax,Lbig
    

def big_matrices_generator(data,Lsize):
    """
    Parameters
    ----------
    data : Numpy array with flattened snapshots taken at a fixed U,t,mu,T.
    Lsize : Linear size of coarse graining window.

    Returns
    ---------
    kMax
    moments_output
    dens_output
    
    Notes
    ---------
    (a) Performs the teselation procedure to make a bigger image.
    (b) For single-spin resolved snapshots in the spin-balanced
    case, this array can contain both spin up and down due to SU(2)
    symmetry.
    """
    numpoints = data.shape[0]

    # The number of symmetries of the square group is 8
    num_samples = numpoints
    sym_num_samples = 8*num_samples
    kMax,Lbig = optimize_kMax(Lsize,sym_num_samples)
    
    print("Lbig = %s"%Lbig)
    print("nx = %s"%int(Lbig*Lsize))
    print("kMax = %s"%kMax)

    # The size of the original square matrices
    Lmatrix = int(np.sqrt(len(data[0])))

    output = []

    # Apply symmetries to that data set
    data_sym_configs= []

    for j in range(0,num_samples):
        for kval in range(0,4):
            data_sym_configs.append(square_symmetry(data[j],"rot",kval,Lsize))
            data_sym_configs.append(square_symmetry(data[j],"ref",kval,Lsize))
    data_sym_configs = np.array(data_sym_configs).reshape(sym_num_samples,Lsize*Lsize)
    
    # Shuffle the data
    np.random.shuffle(data_sym_configs)
    
    # Extract the number of samples that produce the maximum square for teselation
    data_sym_configs = data_sym_configs[:Lbig**2,:]
    
    # Generate new big matrix of local moments
    data_matrix_list = []
    random_order_list1 = np.random.permutation(range(0,Lbig))
    for row in random_order_list1:
        random_order_list2 = np.random.permutation(range(0,Lbig))
        data_matrix_list.append([data_sym_configs[row*Lbig + m].reshape(Lmatrix,Lmatrix) for m in random_order_list2])

    data_Big_matrix = np.bmat(data_matrix_list)
    output.append(data_Big_matrix)
    output = np.array(output)[0]
    
    return kMax,output

#################### STRUCTURAL COMPLEXITY PROCEDURE ##########################

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    # https://stackoverflow.com/questions/16856788/slice-2d-array-into-smaller-2d-arrays
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))

def s_ijk(matrix,L=2):
    # LxL block for coarse graining
    block_matrix = blockshaped(matrix,L,L)
    block_n = int(np.sqrt(block_matrix.shape[0]))
    coarsed_matrix = []
    for i in range(0,block_matrix.shape[0]):
        coarsed_matrix.append(np.mean(block_matrix[i]))
    return np.array(coarsed_matrix).reshape((block_n,block_n))

def coarse_graining(matrix,kMax,L=2):
    # Do the coarse graining procedure
    output_dict = {}
    for k in range(0,kMax+1):
        if k ==0:
            output_dict[0] = matrix
        else:
            output_dict[k] = s_ijk(output_dict[k-1],L)
    return output_dict

def overlap(coarsed_data,L=2):
    # Compute overlap between the coarsed grained data
    def blow_up(x,Lbig):
        # To generate arrays of same size
        Lsmall = len(x)
        Lx = int(np.sqrt(Lsmall))
        x_big = []
        ratio = int(Lbig/Lsmall)
        reshape_ratio = int(np.sqrt(Lbig/Lsmall))

        idx_element = 0
        for i in range(0,Lx):
            row = []
            for j in range(0,Lx):
                cx = np.array(ratio*[x[idx_element]]).reshape(reshape_ratio,reshape_ratio)
                row.append(cx)
                idx_element += 1
            x_big.append(row)
        x_big = np.bmat(x_big)
        x_big = np.array(x_big).flatten()
        return x_big
    
    # Get list of ks
    ks = list(coarsed_data.keys())
    ks.sort(reverse=True)
    overlap_dict = {}
    
    # Get pixel size and size of largest array
    Lbig = len(coarsed_data[0].flatten())

    # Loop over (k1,k2) to get the overlaps
    for k1 in ks:
        v1 = coarsed_data[k1].flatten()
        if k1==0:
            overlap_dict[(k1,k1)] = np.dot(v1,v1)/Lbig
        else:
            for k2 in range(k1,k1-2,-1):
                v2 = coarsed_data[k2].flatten()
                if k1 == k2:  
                    v1_big = blow_up(v1,Lbig)
                    overlap_dict[(k1,k2)] = np.dot(v1_big,v1_big)/Lbig
                else:
                    v1_big = blow_up(v1,Lbig)
                    v2_big = blow_up(v2,Lbig)
                    overlap_dict[(k1,k2)] = np.dot(v1_big,v2_big)/Lbig
    return overlap_dict

def structural_complexity(coarsed_data,overlap_data,kinit=0,details=False):
    # Actually compute the structural complexity, return C0,C1,Dks,ks
    ks = list(coarsed_data.keys())
    C0 = 0
    C1 = 0
    Dks = []
    counter = 0
    for k in range(kinit,ks[-1]):
        counter += 1
        Dk = np.abs(overlap_data[(k+1,k)] - 0.5*(overlap_data[(k,k)]+overlap_data[(k+1,k+1)]))
        Dks.append(Dk)
        C0 += Dk
        if k>0:
            C1 += Dk
        if details == True:
            print("k = %s, Dk = %1.3f, C = %1.3f"%(k,Dk,C0))
    return C0,C1,Dks,ks[:-1]


################### STRUCTURAL COMPLEXITY MEASUREMENT ########################
def analyzer(data,Lsize,Lcoarse=2,kinit=0,detailed=False,example=False):
    """
    Reads the data and performs the procedure.
    Lsize: Linear size of original arrays.
    """
    # Teselate big image
    kMax,output_big = big_matrices_generator(data,Lsize)
    new_nx,new_ny = output_big.shape[0],output_big.shape[1]
    output_matrix = output_big.reshape(new_nx,new_ny)
    
    # Do the coarse graining procude as many times as possible 
    output_coarsed_data = coarse_graining(output_matrix,kMax,Lcoarse) 
    
    # Compute the overlaps
    c_overlap_data = overlap(output_coarsed_data,Lcoarse)
    
    # Compute dissimilarities and structural complexity
    cc,cc1,dks,ks = structural_complexity(output_coarsed_data,c_overlap_data,kinit,detailed)
    
    if example == True:
        # Original
        plt.figure()
        plt.imshow(data[0].reshape(Lsize,Lsize),origin="lower")
        plt.title("Example of original image")
        plt.colorbar()
        
        # Teselated
        plt.figure()
        plt.imshow(output_matrix,origin="lower")
        plt.title("Example of teselated image")
        plt.colorbar()
    
        # Coarse graining steps
        fig,axs = plt.subplots(1,kMax+1,figsize=(14,4),gridspec_kw={'hspace':0.6,'wspace':0.2})
        plt.suptitle("Example of coarse-graining steps",fontsize=14,y=0.8)
        for i in range(0,kMax+1):
            axs[i].imshow(output_coarsed_data[i],vmin=0,vmax=1,origin="lower")
            axs[i].set_title("k=%s"%i)
            if i== kMax:
                im = axs[i].imshow(output_coarsed_data[i],vmin=0,vmax=1,origin="lower")
        fig.colorbar(im,fraction=0.008, pad=0.01, ax=axs.ravel().tolist())

    return kMax,cc,cc1,dks,ks