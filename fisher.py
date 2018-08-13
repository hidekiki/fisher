print("\n \n")

print("######### COMPUTING... ######### \n")

#importing useful packages
#import sys
#import csv

#import matplotlib
#from matplotlib import pyplot as plt
#import matplotlib.cm as cm
#from matplotlib.patches import Ellipse
#from matplotlib import rc
#from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from numpy import *
from numpy import matrix
from numpy import linalg

import scipy
from scipy import misc
from scipy import interpolate
from scipy.stats import chi2

#from pylab import *
import datetime

import os.path

import fishfun

#from extras import Survey

# euclid = Survey(1.60354*10**10,0.000399415,0.001,0.16,1,1)
#euclidlike = Survey(1.60354*10**10,0.000399415,0,0.16,1.,1.)
# old = Survey(10**10,0.005,0,0.17,1,1)


##############################################
#    Parameters for the most general model   # (will be used for all model combinations)
##############################################
# most general model, set all fid /prior values : no need to edit this in general even when changin model.
allparamleg = ['f_{NL}','b_{10}','b_{20}','b_{01}','b_{11}','b_{02}','\chi_1','\omega_{10}','\sigma','R'] #has to be the same in the same order as in fishfun
allpriors = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] # priors

#allfiducial = [0.,0.454,-0.861,1.87,1.155,3.037,-2.024,-0.4821,5.76,0.8]# 10^12 msun fiducial values of parameters
#allfiducial = [0.,0.454,-0.361,1.87,1.94634,3.037,-2.024,-0.4821,5.76,0.8]# 10^12 msun constrained shift b20-b11
#allfiducial = [0.,0.454,-0.361,1.87,1.155,3.037,-1.23266,-0.4821,5.76,0.8]# 10^12 msun constrained shift b20-chi1

allfiducial = [0.,1.51,0.00871,9.38,16.62,74.17,-15.17,-13.09,5.76,1.6]# 10^13 msun fiducial values of parameters
#allfiducial = [0.,1.51,0.50871,9.38,15.30,74.17,-15.17,-13.09,5.76,1.6]# 10^13 constrained shift b20-b11
#allfiducial = [0.,1.51,0.50871,9.38,16.62,74.17,-16.49,-13.09,5.76,1.6]# 10^13 constrained shift b20-chi1
#allfiducial = [0.,1.51,0.00871,0.,0.,0.,0.,0.,5.76,1.6] # all zero except b01 b02

#allfiducial = [0.,5.36,2.3,48.3,281.6,1864,-150.1,-403.8,5.76,3.6]#10^14 msun fiducial values of parameters
#allfiducial = [0.,5.36,2.8,48.3,193.984,1864,-150.1,-403.8,5.76,3.6]#10^14 msun constrained shift b20-b11
#allfiducial = [0.,5.36,2.8,48.3,281.6,1864,-237.716,-403.8,5.76,3.6]#10^14 msun constrained shift b20-chi1

# all the models and combinations of data that we want to compute. for shape and data it's possible to specify more than 1 element : all combinations will be computed
#models=[[["local",],["P","B","P+B"],[1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],1.,0.16,"full"],
#       [["equilateral",],["P","B","P+B"],[1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],1.,0.16,"full"]
#        ] #shape, data, parameters , n, kmax

# VALUE OF BNG FOR SQUEEZED POWER SPECTRUM

#(bng,s0,s1,s2) = (-0.908122,1.027,0.852,1.219) # bng for simple model 10^12 mass
#(bng,s0,s1,s2) = (0.910357,1.027,0.852,1.219) # bng for full model 10^12 mass
#(bng,s0,s1,s2) = (0.00445277, 0.715,0.311, 0.219) # bng for simple model 10^13 mass
moments = (bng,s0,s1,s2) = (2.58659, 0.715,0.311, 0.219) # bng for full model 10^13 mass
#(bng,s0,s1,s2) = (0.461619, 0.448,0.0990,0.0332) # bng for simple model 10^14 mass
#(bng,s0,s1,s2) = (4.20369, 0.448,0.0990,0.0332) # bng for full model 10^14 mass

# for simple model
models=[[["equilateral",],["P","B","P+B"],[ 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],1.,0.16,moments]
        #,[["equilateral",],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16,bng]
        ] #shape, data, parameters , n, kmax

# for full model
# all the models and combinations of data that we want to compute. for shape and data it's possible to specify more than 1 element : all combinations will be computed
#models=[[["local","equilateral","orthogonal"],["P","B","P+B"],[ 1.,1.,1.,1.,1.,1.,1.,1.,1.,1.],1.,0.16,bng]
#        #,[["equilateral",],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16,bng]
#        ] #shape, data, parameters , n, kmax

#models=[[["orthogonal"],["P","B","P+B"],[ 1.,1., 1.,0.,0.,0.,0.,0., 1., 1.],1.,0.16,bng]]

#########################
#    Loop over models   #
#########################
for m in models :
    shapenames= m[0]# what shapes to consider, possible values : none, orthogonal, local, equilateral
    data = m[1]# what data to use, possible values: B, P, P+B
    active = m[2] # choose which parameters to include in the model
    nn=m[3] #set n
    kkhigh = m[4] #set kmax
    mmoments = m[5]
    fishfun.initialize(active, allfiducial, allpriors,nn,kkhigh,mmoments); # initialization
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
    #plt.rcParams['xtick.labelsize'] = tick_size
    #plt.rcParams['ytick.labelsize'] = tick_size

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

        fsyst = open(modelname+'/syst_'+chosenshape+'.dat', 'w+') # opens a file where we print ouput of systematic shifts for all data
        fchi2 = open(modelname+'/chi2'+'_'+chosenshape+'.dat', 'w+') # opens a file where we print ouput of systematic shifts for all data

        for chosendata in data :

            fishfun.datahere = chosendata # setting the data used  in the other file

            f = open(modelname+'/'+chosendata+'_'+chosenshape+'.dat', 'w+') # opens a file where we print all output

            print "###### data used : "+fishfun.datahere+" ###### \n"

            F = fishfun.fisher();
            Finv = linalg.inv(F); #inverse

            # eigen vectors and values of the inverse
            #w,v = linalg.eig(Finv);
            #print("eigen values of 1/F",w)
            #print("eigen vectors of 1/F",v)

            ######################################
            #  non marginalized errors on param  #
            ######################################
            print("######### non marg errors #########")
            f.write("######### non marg errors ######### \n")

            for i in range(0, len(param)):
                print "1-sigma error on %s, non marg : %f" % (param[i],1./np.sqrt(F[i,i]))
                f.write("1-sigma error on %s, non marg : %f \n" % (param[i],1./np.sqrt(F[i,i])))

            # fixing all parameter, it's a 1 param model, so factor is 1. for 1 sigma etc...

            ######################################
            #    marginalized errors on param    #
            ######################################
            print("######### marg errors #########")
            f.write("######### marg errors ######### \n")

            for i in range(0, len(param)):
                print "1-sigma error on %s, marg over all other parameters : %f" % (param[i],1.*np.sqrt(Finv[i,i]))
                f.write("1-sigma error on %s, marg over all other parameters : %f \n" % (param[i],1.*np.sqrt(Finv[i,i])))
            #for 1 parameter of interest, the factor is 1 for 1 sigma, 2 for 2 sigma etc...

            ################################################
            #   marg error on fnl with sigma and R fixed   #
            ################################################
            indhere3 = [ x for x in range(len(param)) if x not in (param.index("sig"),param.index("R")) ] #indices which are not sigam and R

            Fhere3 = np.zeros([len(param)-2,len(param)-2])
            # redefine fisher matrix without sigma and R
            for i in range(len(param)-2):
                for j in range(len(param)-2):
                    Fhere3[i,j]=F[indhere3[i],indhere3[j]]

            Finvhere3 = linalg.inv(Fhere3)

            print "1-sigma error on fnl, marg over all param at fixed R and sigma: %f.2" % np.sqrt(Finvhere3[fnlindex,fnlindex])
            f.write("1-sigma error on fnl, marg over all param at fixed R and sigma: %f.2\n" % np.sqrt(Finvhere3[fnlindex,fnlindex]) )


            ##################################################
            #   marg error on fnl with sigma R b10 b20 fixed #
            ##################################################

            list4 = (param.index("sig"), param.index("R"), param.index("b10"), param.index("b20"))

            indhere4 = [ x for x in range(len(param)) if x not in list4 ] #indices which are not sigam R b10 b20

            Fhere4 = np.zeros([len(param)-len(list4),len(param)-len(list4)])
            # redefine fisher matrix without sigma and R
            for i in range(len(param)-len(list4)):
                for j in range(len(param)-len(list4)):
                    Fhere4[i,j]=F[indhere4[i],indhere4[j]]

            Finvhere4 = linalg.inv(Fhere4)

            print "1-sigma error on fnl, marg over all param at fixed b10, b20, R, sigma: %f.2" % np.sqrt(Finvhere4[fnlindex,fnlindex])
            f.write("1-sigma error on fnl, marg over all param at fixed b10, b20, R, sigma: %f.2\n" % np.sqrt(Finvhere4[fnlindex,fnlindex]))


            ############################
            #   ind1 and ind2 planes   #
            ############################

            for ind1 in range(0,len(param)): # first index is always smaller than second, so ind1 is always first column, then ind2 is always second column
                for ind2 in range(ind1+1,len(param)):
                    #print "ind1= %i, ind2= %i" % (ind1,ind2)
                    #ind1 = 0; # index of the parameters that we consider here
                    #ind2 = 1;
                    parfid1 = fishfun.fiducial[ind1]; # fiducial values of the corresponding parameters
                    parfid2 = fishfun.fiducial[ind2];

                    allindices = range(0,len(param))

                    def delete__by_values(lst, values):
                        values_as_set = set(values)
                        return [ x for x in lst if x not in values_as_set ]

                    otherind = delete__by_values( allindices, [ind1,ind2] )

                    #print(otherind)

                    # generates a string of names of the other parameters
                    def namelist(fulllist, indiceshere):
                        outstring='';
                        for i in indiceshere :
                            outstring+= ' '+fulllist[i]+','
                        return outstring[:-1]

                    otherparam=namelist(param,otherind);
                    otherparamleg=namelist(paramleg,otherind);

                    #print(namelist(param,otherind))

                    #correlation coefficient for this pair of param. Finv of the full matrix, is it marg over the 3rd directino or not?

                    print "######### correlation coefficient %s - %s : %f \n" % (param[ind1],param[ind2],Finv[ind1,ind2]/np.sqrt(Finv[ind1,ind1]*Finv[ind2,ind2]))
                    f.write(" ######### correlation coefficient %s - %s : %f \n" % (param[ind1],param[ind2],Finv[ind1,ind2]/np.sqrt(Finv[ind1,ind1]*Finv[ind2,ind2])))

                    #########################
                    #    FIXED OTHER PARAM  #
                    #########################
                    print("######### %s - %s plane at FIXED%s #########" % (param[ind1],param[ind2],otherparam))
                    f.write("######### %s - %s plane at FIXED%s ######### \n" % (param[ind1],param[ind2],otherparam))

                    # redefine a fisher matrix with the two parameters that we look at, the others being at their fid values
                    Fhere = [[F[ind1,ind1],F[ind1, ind2]],[F[ind2, ind1],F[ind2, ind2]]];

                    #print("Fhere ",Fhere )
                    Finvhere  = linalg.inv(Fhere); #inverse F

                    #print("Finvhere ",Finvhere )

                    print "1-sigma error on %s, marg over %s at fixed%s: %f" % (param[ind1],param[ind2],otherparam,1.*np.sqrt(Finvhere[0,0]))
                    f.write("1-sigma error on %s, marg over %s at fixed%s: %f \n" % (param[ind1],param[ind2],otherparam,1.*np.sqrt(Finvhere[0,0])))
                    print "1-sigma error on %s, marg over %s at fixed%s: %f" % (param[ind2],param[ind1],otherparam,1.*np.sqrt(Finvhere[1,1]))
                    f.write("1-sigma error on %s, marg over %s at fixed%s: %f \n" % (param[ind2],param[ind1],otherparam,1.*np.sqrt(Finvhere[1,1])))

#                    # eigen vectors and eigen values
#                    where,vhere = linalg.eig(Finvhere);
#
#                    b4 = np.sqrt(where[0])
#                    a4 = np.sqrt(where[1])
#                    theta4a = 360/(2.*np.pi)*np.arctan2(vhere[1,0],vhere[0,0]); #angle of the ellipse: here we use the "four quadrant arctan" which gives the angle between the Ox axis and the line (0,0)-(x,y) where the arguements of the functions are (y,x)
#
#                    #########################
#                    # plotting the ellipses #
#                    #########################
#
#                    wc=4*np.sqrt(Finvhere[0,0]); # adjusting x and y axes of the plot
#                    hc=4*np.sqrt(Finvhere[1,1]);
#
#                    pp = PdfPages(modelname+'/'+chosendata+'/'+chosenshape+'/fig_'+chosendata+'_'+chosenshape+'_'+param[ind1]+'_'+param[ind2]+'.pdf') #opens a new pdf file with names of the param
#                    fig = plt.figure()
#                    #fig.subplots_adjust(wspace=1.)
#                    ax = fig.add_subplot(221)
#
#                    ellipse1 = Ellipse(xy=(parfid1,parfid2), width=2.*b4*alpha1, height=2.*a4*alpha1, angle = theta4a, edgecolor='r', fc='None', lw=1)
#                    ellipse2 = Ellipse(xy=(parfid1,parfid2), width=2.*b4*alpha2, height=2.*a4*alpha2, angle = theta4a, edgecolor='b', fc='None', lw=1)
#                    ellipse3 = Ellipse(xy=(parfid1,parfid2), width=2.*b4*alpha3, height=2.*a4*alpha3, angle = theta4a, edgecolor='g', fc='None', lw=1)
#                    ax.add_patch(ellipse1)
#                    ax.add_patch(ellipse2)
#                    ax.add_patch(ellipse3)
#
#                    ax.set_title(r'$1-, \, 2-  \, and \, 3-\sigma \,{\rm joint \, confidence  \, ellipses \, for \, fixed  \,} %s $' % otherparamleg,fontsize =title_size)
#                    ax.set_xlabel(r'$%s$' % paramleg[ind1],fontsize =label_size);
#                    ax.set_ylabel(r'$%s$' % paramleg[ind2],fontsize =label_size)
#                    ax = plt.gca() #remove crappy tick labels
#                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
#                    ax.get_yaxis().get_major_formatter().set_useOffset(False)
#
#                    plt.draw()
#                    plt.axis((parfid1-wc,parfid1+wc,parfid2-hc,parfid2+hc)) #axis limits
#                    plt.scatter(parfid1,parfid2) # center dot
#                    #plt.show()
#                    #pp.savefig(fig) #prints the fig to the pdf
#
#                    ################################
#                    #    MARGINALIZED OTHER PARAM  #
#                    ################################
#                    print("######### %s - %s plane MARGINALIZED over%s #########" % (param[ind1],param[ind2],otherparam))
#                    f.write("######### %s - %s plane MARGINALIZED over%s ######### \n" % (param[ind1],param[ind2],otherparam))
#
#                    #select the columns and lines corresponding to the parameters of interest
#                    Finvhere2 = np.array([[Finv[ind1,ind1],Finv[ind1, ind2]],[Finv[ind2, ind1],Finv[ind2, ind2]]]);
#                    Fhere2 = linalg.inv(Finvhere2)
#
#                    print "1-sigma error on %s, marg over%s at fixed %s: %f" % (param[ind1],otherparam,param[ind2],1./np.sqrt(Fhere2[0,0]))
#                    f.write("1-sigma error on %s, marg over%s at fixed %s: %f \n" % (param[ind1],otherparam,param[ind2],1./np.sqrt(Fhere2[0,0])))
#                    print "1-sigma error on %s, marg over%s at fixed %s: %f" % (param[ind2],otherparam,param[ind1],1./np.sqrt(Fhere2[1,1]))
#                    f.write("1-sigma error on %s, marg over%s at fixed %s: %f \n" % (param[ind2],otherparam,param[ind1],1./np.sqrt(Fhere2[1,1])))
#
#                    # eigen vectors and eigen values
#                    where2,vhere2 = linalg.eig(Finvhere2);
#
#                    b5 = np.sqrt(where2[0])
#                    a5 = np.sqrt(where2[1])
#
#                    theta5a = 360/(2.*np.pi)*np.arctan2(vhere2[1,0],vhere2[0,0]);
#
#                    # plotting the ellipses
#                    wc2=4*np.sqrt(Finvhere2[0,0]); # adjusting x and y axes of the plot
#                    hc2=4*np.sqrt(Finvhere2[1,1]);
#
#                    #fig2 = plt.figure()
#                    ax2 = fig.add_subplot(222)
#
#                    ellipse4 = Ellipse(xy=(parfid1,parfid2), width=2.*b5*alpha1, height=2.*a5*alpha1, angle = theta5a, edgecolor='r', fc='None', lw=1)
#                    ellipse5 = Ellipse(xy=(parfid1,parfid2), width=2.*b5*alpha2, height=2.*a5*alpha2, angle = theta5a, edgecolor='b', fc='None', lw=1)
#                    ellipse6 = Ellipse(xy=(parfid1,parfid2), width=2.*b5*alpha3, height=2.*a5*alpha3, angle = theta5a, edgecolor='g', fc='None', lw=1)
#                    ax2.add_patch(ellipse4)
#                    ax2.add_patch(ellipse5)
#                    ax2.add_patch(ellipse6)
#
#                    ax2.set_title(r'$1-, \, 2-  \,  and \, 3-\sigma \, {\rm joint \, confidence  \, ellipses \, marginalized \, over  \,} %s $' % otherparamleg,fontsize =title_size)
#                    ax2.set_xlabel(r'$%s$' % paramleg[ind1],fontsize =label_size);
#                    ax2.set_ylabel(r'$%s$' % paramleg[ind2],fontsize =label_size)
#                    ax2 = plt.gca() #remove crappy tick labels
#                    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
#                    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
#
#                    plt.draw()
#                    plt.axis((parfid1-wc2,parfid1+wc2,parfid2-hc2,parfid2+hc2)) #axis limits
#                    plt.scatter(parfid1,parfid2) # center dot
#                    #pp.savefig(fig2) #prints the fig to the pdf
#
#                    ############################################################
#                    ########## combined plot of marg and non marg ##############
#                    ############################################################
#                    #fig3 = plt.figure()
#                    ax3 = fig.add_subplot(223)
#
#                    ellipse7 = Ellipse(xy=(parfid1,parfid2), width=2.*b4*alpha2, height=2.*a4*alpha2, angle = theta4a, edgecolor='b', fc='None', lw=1)
#                    ellipse8 = Ellipse(xy=(parfid1,parfid2), width=2.*b5*alpha2, height=2.*a5*alpha2, angle = theta5a, edgecolor='g', fc='None', lw=1)
#                    ax3.add_patch(ellipse7)
#                    ax3.add_patch(ellipse8)
#
#                    ax3.set_title(r'$2- \sigma \, {\rm joint \, confidence  \, ellipses \, marginalized \,and \, non \, marginalized \, over  \,} %s $' % otherparamleg,fontsize =title_size)
#                    ax3.set_xlabel(r'$%s$' % paramleg[ind1],fontsize =label_size);
#                    ax3.set_ylabel(r'$%s$' % paramleg[ind2],fontsize =label_size)
#                    ax3 = plt.gca() #remove crappy tick labels
#                    ax3.get_xaxis().get_major_formatter().set_useOffset(False)
#                    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
#
#                    plt.draw()
#                    plt.axis((parfid1-wc2,parfid1+wc2,parfid2-hc2,parfid2+hc2)) #axis limits
#                    plt.scatter(parfid1,parfid2) # center dot
#                    plt.tight_layout() #pad=1.08, h_pad=None, w_pad=None, rect=None
#                    pp.savefig(fig) #prints the fig to the pdf
#                    pp.close() #closes pdf doc
#                    plt.close(fig)

###########################################
# squeezed power spectrum and bispectrum  #
###########################################


            if chosenshape == "local" :

                fsq = open(modelname+'/'+chosendata+'_'+chosenshape+'_squeezed.dat', 'w+') # opens a file where we print all output

                print "\n"
                print "###### data used : "+fishfun.datahere+" SQUEEZED ###### \n"

                Fsq = fishfun.fisher_squeezed();
                Fsqinv = linalg.inv(Fsq); #inverse

                ######################################
                #  non marginalized errors on param  #
                ######################################
                print("######### non marg errors #########")
                fsq.write("######### non marg errors ######### \n")

                for i in range(0, len(param)):
                    print "1-sigma error on %s, non marg : %f" % (param[i],1./np.sqrt(Fsq[i,i]))
                    fsq.write("1-sigma error on %s, non marg : %f \n" % (param[i],1./np.sqrt(Fsq[i,i])))

                # fixing all parameter, it's a 1 param model, so factor is 1. for 1 sigma etc...

                ######################################
                #    marginalized errors on param    #
                ######################################
                print("######### marg errors #########")
                fsq.write("######### marg errors ######### \n")

                for i in range(0, len(param)):
                    print "1-sigma error on %s, marg over all other parameters : %f" % (param[i],1.*np.sqrt(Fsqinv[i,i]))
                    fsq.write("1-sigma error on %s, marg over all other parameters : %f \n" % (param[i],1.*np.sqrt(Fsqinv[i,i])))
                    #for 1 parameter of interest, the factor is 1 for 1 sigma, 2 for 2 sigma etc...
                fsq.close()
            print("\n")

########################################
# for systematic shift in parameters   #
########################################

            fishfun.shift(fsyst)
            fsyst.write("\n")
            fishfun.double_shift(fsyst)
            fishfun.bigshift(fchi2)
            
#            print "\n"
#            print "############# one sigma systematic shift ###############"
#            f.write("############# one sigma systematic shift ###############\n")
#
#            for shiftedparam in  [x for x in param if x != "fnl"] :# vary each parameter which is not fnl
#
#                print("################  varying %s  ################" % shiftedparam )
#
#                fnlindex = param.index("fnl") #for convienience
#
#                shiftedfid = fishfun.fiducial[param.index(shiftedparam)]#fid value of the shifted param
#
#                #factorlist = fishfun.compute_fnlshift_list(shiftedparam) #factors of fnl
#
#                deltafnllist = fishfun.compute_fnlshift_list(shiftedparam)
#
#                #deltafnllist = [Finv[param.index(shiftedparam),fnlindex] * x for x in factorlist];  #  list of shifted values of fnl when shifting the values of "shiftedparam" to the values in interp_list(fiducial[param.index(shiftedparam)])
#
#                parametervalues = fishfun.interp_list(shiftedparam) # jsut for plotting
#
#                #print("#############  systematic shifts ###############")
#                #print("shift in b01 from %f to %f produces a shift in fnl %s of %f while fnl marg is %f " % (fiducial[b01index], b01shift, chosenshape , shiftedfnlfactor, np.sqrt(Finv[fnlindex,fnlindex])))
#                #f.write("shift in b01 from %f to %f produces a shift in fnl %s of %f while fnl marg is %f " % (fiducial[b01index], b01shift, chosenshape ,shiftedfnlfactor, np.sqrt(Finv[fnlindex,fnlindex])))
#
#                # interpolated the list of delta fnl as function of shift
#                deltafnl = interpolate.splrep(parametervalues, deltafnllist) #cubic spline interpolation of scipy
#
#    #            xnew = np.arange(0,2*shiftedfid,0.01) # new finer k list values where to evaluate for the figure
#    #            ynewcub = interpolate.splev(xnew, deltafnl) # evaluation of the interpolated p(k) at these points
#    #            plt.figure()
#    #            plt.plot(parametervalues, deltafnllist, 'x', xnew, ynewcub)
#    #            plt.legend(['Data', 'Cubic Spline'])
#    #            plt.axis([min(xnew), max(xnew), min(ynewcub), max(ynewcub)])
#    #            plt.title('delta fnl of %s' % shiftedparam)
#    #            plt.show()
#
#                #find where deltafnl is equal to fnl marg
#                onesigmafnl = np.sqrt(Finv[param.index("fnl"),fnlindex])
#
#                def zerofnl(shift) :
#                    return  interpolate.splev(shift, deltafnl)-onesigmafnl
#
##                xnew = np.arange(min(parametervalues),max(parametervalues),(max(parametervalues)-min(parametervalues))/100) # new finer k list values where to evaluate for the figure
##                ynewcub = interpolate.splev(xnew, deltafnl) # evaluation of the interpolated p(k) at these points
##                plt.figure()
##                plt.plot(parametervalues, deltafnllist, 'x', xnew, ynewcub,'-')
##                plt.legend(['Data', 'Cubic Spline'])
##                plt.axis([min(xnew), max(xnew), min(ynewcub), max(ynewcub)])
##                plt.title('delta fnl of %s' % shiftedparam)
##                plt.show()
#
#
#                onesigmashifted = scipy.optimize.newton(zerofnl, shiftedfid,maxiter=200)-shiftedfid
#                # difference between fid value of shiftedparam and the values at which it produces a shift of 1 sigma on fnl
#                print "shift in %s leading to 1-sigma shift in fnl (%f)  : %f " % (shiftedparam,onesigmafnl,onesigmashifted)
#                f.write("shift in %s leading to 1-sigma shift in fnl (%f)  : %f \n" % (shiftedparam,onesigmafnl,onesigmashifted) )
#            #            # what if we put the parameter to zero
#            #            print "######### fnl syst error due to b01=0 #########"
#            #            f.write("######### fnl syst error due to b01=0 ######### \n")
#            #            print  "syst shift in fnl %s : %f" % (chosenshape, interpolate.splev(0., deltafnl))
#            #            f.write("syst shift in fnl %s : %f" % (chosenshape, interpolate.splev(0., deltafnl)))
#            #            ratio = interpolate.splev(0., deltafnl) / np.sqrt(Finv[3,3]) ;
#            #            print "ratio syst fnl %s error / stat fnl error : %f" % (chosenshape, ratio)
#            #            f.write("ratio syst fnl %s error / stat fnl error : %f" % (chosenshape, ratio))
#
#                print("\n")

            f.close()
        print("\n")
        fsyst.close()
        fchi2.close()
    print("\n")
print datetime.datetime.now()
print("############ DONE ############ \n")
