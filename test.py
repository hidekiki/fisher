print("\n \n")

print("######### COMPUTING... ######### \n")

#importing useful packages
import sys
import csv

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy import *
from numpy import matrix
from numpy import linalg

import scipy
from scipy import misc
from scipy import interpolate
from scipy.stats import chi2

from pylab import *

import os.path

import fishfun

##############################################
#    Parameters for the most general model   # (will be used for all model combinations)
##############################################
# most general model, set all fid /prior values : no need to edit this in general even when changin model.
allparamleg = ['f_{NL}','b_{10}','b_{20}','b_{01}','b_{11}','b_{02}','\chi_1','\omega_{10}','\sigma','R'] #has to be the same in the same order as in fishfun
#allfiducial = [0.,1.51,0.00871,0.,0.,0.,0.,0.,5.76,1.6] # all zero except b01 b02
#allfiducial = [0.,0.454,-0.861,1.87,1.155,3.037,-2.024,-0.4821,5.76,0.8]# 10^12 msun
allfiducial = [0.,1.51,0.00871,9.38,16.62,74.17,-15.17,-13.09,5.76,1.6]# 10^13 msun fiducial values of parameters ORDER MATTERS
#allfiducial = [0.,5.36,2.3,48.3,281.6,1864,-150.1,-403.8,5.76,3.6]#10^14 msun
allpriors = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] # priors

[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid]=allfiducial


# all the models and combinations of data that we want to compute. for shape and data it's possible to specify more than 1 element : all combinations will be computed
#models=[[["local",],["P","B","P+B"],[1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],1.,0.16,"full"],
#       [["equilateral",],["P","B","P+B"],[1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],1.,0.16,"full"]
#        ] #shape, data, parameters , n, kmax

# all the models and combinations of data that we want to compute. for shape and data it's possible to specify more than 1 element : all combinations will be computed
models=[[["local",],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16],
        [["equilateral",],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16]
        ] #shape, data, parameters , n, kmax

#models=[[["orthogonal"],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16,"simple"]]

#########################
#    Loop over models   #
#########################
for m in models :
    shapenames= m[0]# what shapes to consider, possible values : none, orthogonal, local, equilateral
    data = m[1]# what data to use, possible values: B, P, P+B
    active = m[2] # choose which parameters to include in the model
    nn=m[3] #set n
    kkhigh = m[4] #set kmax
    fishfun.initialize(active, allfiducial, allpriors,nn,kkhigh); # initialization
    fishfun.model_output()
    param = fishfun.param # for convienience
    fnlindex = param.index("fnl") #for convienience
    
    ################################
    #    create relevant folders   #
    ################################
    
    modelname = "model_"+"_".join(fishfun.param) #create model name
    
    fishfun.modelhere = modelname
    
    for shapeiter in shapenames: #  creating folders for results
        for dataiter in data :
            if not os.path.exists(modelname+"/"+dataiter+"/"+shapeiter):
                os.makedirs(modelname+"/"+dataiter+"/"+shapeiter)
    if not os.path.exists(modelname+"/temp/"):
        os.makedirs(modelname+"/temp/")

    #######################
    # plotting parameters #
    #######################
    paramleg = fishfun.set_active(active,allparamleg) # list of legends
    label_size = 12.; # font size for plots legends
    title_size = 10.  ; # font size for plots titles
    tick_size = 9.
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['ytick.labelsize'] = tick_size
    
    alpha1 = np.sqrt(chi2.ppf(0.6827, 2, loc=0, scale=1)); # which ellipses to draw amount to multiply the eigenvalues
    alpha2 = np.sqrt(chi2.ppf(0.9545, 2, loc=0, scale=1));
    alpha3 = np.sqrt(chi2.ppf(0.9973, 2, loc=0, scale=1));
    
    ###############################
    # LOOP over shapes and fisher #
    ###############################
    
    for chosenshape in shapenames :
        
        fishfun.shapehere = chosenshape # setting the shape in the other file
        
        print("\n")
        print("################  "+fishfun.shapehere+"  ################ \n") #here we use the print to be sure it has been correctly set
        
        for chosendata in data :
            
            fishfun.datahere = chosendata # setting the data used  in the other file
            
            f = open(modelname+'/'+chosendata+'_'+chosenshape+'.dat', 'w+') # opens a file where we print all output
            
            print "###### data used : "+fishfun.datahere+" ###### \n"

            #fishfun.compute_pfid()
#fishfun.dpkpar(0.1,"b10")
#fishfun.DP_integrand(0.1,0.1,0.1,allfiducial,"b10")
#       fishfun.compute_bfid() #compute ptotlsit for this shape
          
          # print fishfun.B_integrand(fishfun.trianglelist[1],0.1,0.1,[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid])
          #fishfun.bkfid([0.1,0.1,0.1])

            fishfun.compute_dpfid()
            quit()
#            print fishfun.compute_fisher()
        
#        F =fishfun.F_BB()
#        
#        print F
#
#        print linalg.inv(F); #inverse

        #tri = [x for x in fishfun.trianglelist if x[0]==x[1] ]
        
        #tri = [fishfun.trianglelist[i] for i in range(10) ]
        
#        tri =  [fishfun.trianglelist[i] for i in range(2) ]
#
#        for i in range(len(tri)): #
#            print "triangle %i" % i
#            print tri[i]
#            print fishfun.DB(tri[i],'fnl')
#        
# quit()


#k = 0.07

#for i in range(10):
#    for j in range (10):
#        print Fel_PP(0.3,0.2,0.21,i,j)

#def f(y):
#    return DP2(k,y[0],y[1],"b01")
#
## print f([0.1,0.1])
#
#integ = vegas.Integrator([[qmin, qmax], [-1, 1]])
#
#result = integ(f, nitn=6, neval=4000)
#print(result.summary())
#print('result = %s    Q = %.2f ' % (result, result.Q))