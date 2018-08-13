# in this module we put all the definitions of functions used in the other file
#maybe it would be better to use functions that pass all needed parameters everytime, without use of global variables but whatevers

#############
#  imports  #
#############
import numpy as np
from numpy import linalg, pi, sin
from scipy import interpolate

#import multiprocessing
import multiprocess # BETTER THAN MULTIPROCESSING
import datetime
from tempfile import TemporaryFile
import os.path
#import matplotlib
#from matplotlib import pyplot as plt
import vegas

from extras import set_active, cartesian, interp_list, set_shift

#import psutil # to see how many cores are allocated
#ncores =  len(psutil.Process().cpu_affinity()) # for cluster
ncores =  multiprocess.cpu_count() # for local

############################
#    numerical parameters  #
############################
E = np.exp(1.) # no longer needed?
qmin = 0.0001
qmax = 10. #default
n=1.; # default consider every n*kf for the bispectrum . for the power specutrm it computes every kf.
ni = 2; #number iterations
ne = 5000; #number of evaluations #3500 normal, 5000 for high prec
chunksize = 500 # divide the triangle list in chunks of size "chunksize". carefull not to change the chunksize between runs!

#############################
#  cosmological parameters  #
#############################
#(one could make the code read it from the class output directly)
#initial conditions and cosmological parameters used in CLASS in particular # As s.t. sigma8=0.9
As =  2.1265*10**(-9) ; # amplitude of the dimensionless power spectrum Delta_s of the curvature perturbation at the initial conditions (inflation) as we use the transfer funciton of CLASS linking the power spectrum at inflation to the linear delta of matter at the desired redshit.
# note that for the loops, as here we do Eulerian SPT we directly use the power spectrum output of CLASS at the desired z.
Azeta = 2.*np.pi*np.pi*As; #amplitude entering the shapes definition
zhere = 1.3; # this is just to make sure we are aware of the redshift that we consider
Om = 0.32; # Omega matter, for the beta factor for RSD
ns = 0.96; # fixed for now, appears in the shapes
deltac = 1.68

##########################
#  Pk and Tk from CLASS  #
##########################
#importing linear matter power spectrum at z=0 computed from class with plank 2015 param.
pklin = np.loadtxt('./pklin_pk.dat'); # creates an array with arrays of [k, p(k)]
kvalues = pklin[:,0] # only the k list
pvalues = pklin[:,1] # only the p(k) list
pkint = interpolate.splrep(kvalues, pvalues) #cubic spline interpolation of scipy

def P(k) : # rename for convienience (this is the dimension full power spectrum)
    return interpolate.splev(k, pkint);

# importing and interpolating transfer function of matter : the dimension full power spectrum of delta is then P_delta(k,z) = Tk(z)^2 P_zeta(k) where <zeta(k) zeta(k') > = (2PI)^3 delta(k+k') P_zeta(z_inflation) to P_delta(k,z) which is what we need here as the templates are defined for <zeta zeta zeta >. No further normalization.
tklin = np.loadtxt('./pklin_tk.dat',skiprows=1)
tkint = interpolate.splrep(tklin[:,0], tklin[:,5])

def T(k) :   # rename for convienience. T relates the initial curvature power spectrum (dimension full 2 PI^2 As (k/k_*)^(ns-1) /k^3 T(k)^2 = P_prim T(k)^2 =  P(k)
    return interpolate.splev(k, tkint) #never zero, good

##########################
#  primodial ng shapes   #
##########################
# shapes with ns dependance UNCOMMENT THE SHAPE YOU USE

#def Fshape(k1,k2,k3): # orthogonal
#    return  np.asarray(- T(k1) * T(k2) * T(k3)  * (18./5.) *  Azeta**2  * ( - (3./(k1**(4.-ns) * k2**(4.-ns))) - (3./(k1**(4.-ns)  * k3**(4.-ns))) - (3./(k2**(4.-ns)  * k3**(4.-ns))) - (2./(k1**(2.*(4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k3**(2.*(4.-ns)/3.))) + (3./(k1**((4.-ns)/3) *  k2**(2.*(4.-ns)/3.)  * k3**(4.-ns) ))  + (3./(k2**((4.-ns)/3.) *  k3**(2.*(4.-ns)/3.) * k1**(4.-ns))) + (3./(k3**((4.-ns)/3.) *  k1**(2.*(4.-ns)/3)  * k2**(4.-ns) ) ) + (3./(k3**((4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k1**(4.-ns)) )+ (3./(k2**((4.-ns)/3.)  * k1**(2*(4.-ns)/3.)  * k3**(4.-ns) )) + (3./(k1**((4.-ns)/3.)  * k3**(2.*(4.-ns)/3.)  * k2**(4.-ns))) ))
#
def Fshape(k1,k2,k3): #"equilateral":
    return  np.asarray(- T(k1) * T(k2) * T(k3)  * (18./5.) *  Azeta**2  * ( - (1./(k1**(4.-ns) * k2**(4.-ns))) - (1./(k1**(4.-ns)  * k3**(4.-ns))) - (1./(k2**(4.-ns)  * k3**(4.-ns))) - (2./(k1**(2.*(4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k3**(2.*(4.-ns)/3.))) + (1./(k1**((4.-ns)/3) *  k2**(2.*(4.-ns)/3.)  * k3**(4.-ns) ))  + (1./(k2**((4.-ns)/3.) *  k3**(2.*(4.-ns)/3.) * k1**(4.-ns))) + (1./(k3**((4.-ns)/3.) *  k1**(2.*(4.-ns)/3)  * k2**(4.-ns) ) ) + (1./(k3**((4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k1**(4.-ns)) )+ (1./(k2**((4.-ns)/3.)  * k1**(2*(4.-ns)/3.)  * k3**(4.-ns) )) + (1./(k1**((4.-ns)/3.)  * k3**(2.*(4.-ns)/3.)  * k2**(4.-ns))) ))

#def Fshape(k1,k2,k3): #local":
#    return np.asarray( - T(k1) * T(k2) * T(k3)  * (6./5.) *  Azeta**2  * ( (1./(k1**(4.-ns) * k2**(4.-ns))) + (1./(k1**(4.-ns)  * k3**(4.-ns))) + (1./(k2**(4.-ns)  * k3**(4.-ns))) ))

#######################
#  survey parameters  #
#######################
#old survey
#V = 10**10 # volume of the survey in (Mpc/h)^3 1.60354*10**10
#ng = 0.005 # galaxy density for shot noise in  (h/Mpc)^3 0.000399415
#
## triangles configurations over which we will sum over // sum over all ordered combinations
#kf = ((2.*np.pi)/(V**(1./3.))); # for a 10Gpc^3 survey, took kf = 2PI/cubic root (V) in 1/(Mpc/h)
#klow =  kf; #0.001
#khigh = 0.17; # as in sefussati and komatsu 0.2

#new survey
V = 1.60354*10**10 # volume of the survey in (Mpc/h)^3
ng = 0.000399415  # galaxy density for shot noise in  (h/Mpc)^3

# triangles configurations over which we will sum over // sum over all ordered combinations
kf = ((2.*np.pi)/(V**(1./3.))); # for a 10Gpc^3 survey, took kf = 2PI/cubic root (V) in 1/(Mpc/h)
klow =  kf; #
khigh = 0.2; # default

############################################
#   plotting points, cubic interpolation   #
############################################
#xnew = np.arange(0,1,0.01) # new finer k list values where to evaluate for the figure
#ynewcub = interpolate.splev(xnew, tkint) # evaluation of the interpolated p(k) at these points
#plt.figure()
#plt.plot(tklin[:,0], tklin[:,5], 'x', xnew, ynewcub)
#plt.legend(['Data', 'Cubic Spline'])
#plt.axis([min(tklin[:,0]), max(tklin[:,0]), min(tklin[:,5]), max(tklin[:,5])])
#plt.title('Checking interpolation')
#plt.show()

################
#  initialize  #
################
shapehere = "none" # for when running the loops
datahere = "none"
currentparameter = "fnl"; # ugly, but current parameters that we are varrying to use the multipool because doesn't support multiple arguments. set to fnl so it doesnt create error on loading

pfid = [] # stores and save values such that we dont compute twice the same quantity (pfid)
pshift = []
bfid = []
bshift = []
dpfid = []
dpfid_sq = []
dbfid = []
klist = []
trianglelist =[]

recpfid = "no" # whether to recomputed pfid, fp, fb or load it from file if it already exists
recpfid_sq = "no"
recpshift = "no"
recbfid = "no"
recbshift = "no"
recdpfid = "no"    # turn off for db fid
recdbfid = "no"
recfp = "no"
recfb = "no"
recfnlb = "no"
recfnlp = "no"

# most general model for which definitions below make sense
allparam=["fnl","b10","b20","b01","b11","b02","chi1","w10","sig","R"]; #ORDER MATTERS!
param = []  # active parameters in the model (all param are in this version)
fiducial = []
priors = []
modelhere = "" # for file naming

fnlfid = 0. # need to put the correct values from main file
b10fid = 0.
b20fid = 0.
b01fid = 0.
b11fid = 0.
b02fid = 0.
w10fid = 0.
Rfid = 0.
sigfid = 0.
chi1fid = 0.
bng = 0.
sigma0 = 0.
sigma1 = 0.
sigma2 = 0. # these are the sigma 0, 1, 2...

# for the defintion for syst shifts fucntions, set by "set_seshift" in extras.py
fnlshift = 0. # only the active ones will be set to their values.
b10shift = 0.
b20shift = 0.
b01shift = 0.
b11shift = 0.
b02shift = 0.
w10shift = 0.
Rshift = 0.
sigshift = 0.
chi1shift = 0.

# all switches are active in this version
snl = 0.;
s10 = 0.;
s20 = 0.;
s01 = 0.;
s11 = 0.;
s02 = 0.;
sw10 = 0.;
sR = 0.;
ssig = 0.;
schi1 = 0.;

#counts progress in dbfid
stri = 0;

#sets dedicated names for the fidvalues
def initialize(act,allfid,allpri,nn,kkhigh,mmoments):
    global qmax
    global khigh
    global n
    global fnlfid
    global b10fid
    global b20fid
    global b01fid
    global b11fid
    global b02fid
    global w10fid
    global Rfid
    global sigfid
    global chi1fid
    global bng
    global sigma0
    global sigma1
    global sigma2
    global param
    global priors
    global fiducial
    global snl
    global s10
    global s20
    global s01
    global s11
    global s02
    global sw10
    global sR
    global ssig
    global schi1
    #global modelname
    #modelname = modelnamee # assign which model we want
    #ORDER MATTERS!
    [fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid] = allfid #change here
    [snl ,s10, s20, s01, s11, s02,schi1, sw10, ssig ,sR ] = act #set the switches for each parameter
    param = set_active(act,allparam) # set values of active parameters
    priors = set_active(act,allpri)
    fiducial = set_active(act,allfid)
    qmax = 5./Rfid
    khigh = kkhigh
    (bng,sigma0,sigma1,sigma2) = mmoments
    n = nn
    compute_list()

def model_output() :
    print("#############################")
    print("#         NEW MODEL         #")
    print("#############################")
    print "the parameters in the model are : "+', '.join(param)
    print "     with fiducial values       :",fiducial
    print "redshift z = %.3f" % zhere
    print "qmax = %.1f, kmax = %.3f, n = %.1f" % (qmax, khigh, n)
    print "there are %i k's for the power specutrm" % len(klist)
    print "there are %i triangles for the bispecutrm" % len(trianglelist)
    print "the chunksize for dbfid is %i" % chunksize
    print "number of cpu used %i" % ncores


#######################
# triangle in the sum #
#######################

#POWER SPECUTRM EVERY KF

# generating the list of triangles from k min to kmax with spacing kf

def compute_list():
    global klist
    global trianglelist
    if klist == [] :
        klist = np.arange(klow,khigh,kf).tolist() # will be used in the sum for the PP fisher

# generating the list of triangles from k min to kmax with spacing kf
    if trianglelist ==  [] :
        pointlistB = np.arange(klow,khigh,n*kf).tolist() # will be used in the sum for the PP fisher
        listraw = cartesian((pointlistB, pointlistB, pointlistB))
        trianglelist = [s for s in listraw if ((2*max(s[0],s[1],s[2]) <= s[0]+s[1]+s[2]) and (s[0]<=s[1]<=s[2])   )] # keep only triangle inequality and with sides ordered by increasing length

####################################
# import defintion for given model #
####################################
# importing relevant definitions
from fullmodel import P_integrand, DP_integrand, B_integrand, DB_integrand, DP_sq_integrand, a_integrand, b_integrand, c_integrand, a0_integrand, a1_integrand, a2_integrand, a3_integrand, double_shift_abc, double_shift_a0123
#from simplemodel import P_integrand, DP_integrand, B_integrand, DB_integrand, DP_sq_integrand, a_integrand, b_integrand, c_integrand, a0_integrand, a1_integrand, a2_integrand, a3_integrand
from chi2 import chi2_delta_p, chi2_delta_b, chi2_coeff_p, chi2_coeff_b

###################
# variance and pt #
###################
# variance of reduced bispectrum function in redshift space// here
def var_b(k1,k2,k3): #only the factor to be able to take the b1 derivativ in the full fisher case
    if k1 != k2 != k3: trianglevar = 1.;
    elif k1 == k2 != k3 or k1 == k3 != k2 or k2 == k3 != k1 : trianglevar = 2.;
    else : trianglevar = 6. ;
    return (2*np.pi)**3 * trianglevar*(pfid[klist.index(k1)]*pfid[klist.index(k2)]*pfid[klist.index(k3)]) / (8. * np.pi**2 * n**3 * k1* k2*k3);

###########################
# one loop Power spectrum #
###########################

# integrates and returns power spectrum for a given k general values of parameters (for use in "shift")
def pk(k,(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid)):
    # print "k = %.5f" % k
    def f(y):
        return P_integrand(k,y[0],y[1],[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid])
    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
    result = integ(f, nitn=ni, neval=4*ne)
    #print result.summary()
    return result.mean

def pfidk(k): # computes p(k) for fiducial values of parameters (for use in parrallel)
    return pk(k,(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid))

def compute_pfid(): # computes power spectrum fiducial over pointlist and saves it
    global pfid

    if os.path.isfile(modelhere+'/temp/pfid_'+shapehere+'.npz') and recpfid == "no":
        data = np.load(modelhere+'/temp/pfid_'+shapehere+'.npz')
        pfid = data['pfid'].tolist()
        data.close()
        print "pfid loaded"
    else :
        print datetime.datetime.now()
        print "computing pfid"
        poolP = multiprocess.Pool(processes=ncores); # compute ptot (pointlist) if not already computed
        ptemp = poolP.map(pfidk, klist)
        pfid = [x+ (1./ng) for x in ptemp] # shot noise
        poolP.close()

        print datetime.datetime.now()
        print "pfid done"

        np.savez(modelhere+'/temp/pfid_'+shapehere+'.npz',pfid=np.asarray(pfid))
    #print pfid

###### derivatives of spectrum ######
# integrates and returns partial derivative of power spectrum wrt a parameter for a given k and fid val of param
def dpkpar(k,par):
    def f(y):
        return  DP_integrand(k,y[0],y[1],[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid],par)
    integ = vegas.Integrator([[qmin, qmax], [-1, 1]])
    result = integ(f, nitn=ni, neval = 4*ne) # slightly more evaluations and iterations for the derivative to be sure to have about few percent accuracy
    #print result.summary()
    return result.mean

def dpk(k): # return an array of the derivatives of P wrt to each param for given k
    print datetime.datetime.now()
    print " k = %.3f" % k
    return [ dpkpar(k,x) for x in param ]

def compute_dpfid(): #computes the derivatives of p wrt each param over the list of k pointlist
    global dpfid

    if os.path.isfile(modelhere+'/temp/dpfid_'+shapehere+'.npz') and recdpfid == "no":
        data = np.load(modelhere+'/temp/dpfid_'+shapehere+'.npz')
        dpfid = data['dpfid'].tolist()
        data.close()
        print "dpfid loaded"
        #print dpfid
    else :
        print datetime.datetime.now()
        print "computing dpfid (dP/dlambda): there are %i k's" % len(klist)

        poolP = multiprocess.Pool(processes=ncores); # start a multiprocess
        dpfid = poolP.map(dpk, klist)
        poolP.close()

        #print dpfid
        print datetime.datetime.now()
        print "dpfid done"
        np.savez(modelhere+'/temp/dpfid_'+shapehere+'.npz',dpfid=np.asarray(dpfid))

######## fisher power spectrum ########

def var_p(k):
    return (pfid[klist.index(k)])**2 * kf**2 / (2. * np.pi * n * k**2)

def Fel_PP(k): #element of the sum of the fisher matrix for given k, reads from computed quantities
    #print datetime.datetime.now()
    #print "computing k = %.3f" % k

    Ftemp = np.zeros([len(param), len(param)],dtype=float)

    for i in range(len(param)):                 #builds the fisher matrix from derivatives and the variance
        for j in range(len(param)):
            Ftemp[i,j] = dpfid[klist.index(k)][i]*dpfid[klist.index(k)][j]/var_p(k)

    return Ftemp

def F_PP() : #computes the fisher with power spectrum data : about 35min (integrals)

    if os.path.isfile(modelhere+'/temp/fp_'+shapehere+'.npz') and recfp == "no": # loads the fisher if already computed
        data = np.load(modelhere+'/temp/fp_'+shapehere+'.npz')
        Ftemp = data['fp']
        data.close()
        print "fp loaded"
        # print Ftemp
    else :
        print datetime.datetime.now()
        print "computing fp"
        compute_pfid() # computes the ps
        compute_dpfid() # computes the derivatives of ps
        print len(klist)
        print len(param)
        print len(pfid)
        print len(dpfid[1])

        Ftemp = np.zeros([len(param), len(param)],dtype=float)

        poolP = multiprocess.Pool(processes=ncores); # start a multiprocess
        Ftemp = np.sum(poolP.map(Fel_PP, klist ),axis=0); # parallel evaluate to speed up PP does not dep on shape so we take it out of the loop. for the defs to be shared, the B-pool needs to be after chosenshape has been set (in the loop)
        poolP.close()

        print datetime.datetime.now()
        print "F_P done"
        # print Ftemp
        np.savez(modelhere+'/temp/fp_'+shapehere+'.npz',fp=Ftemp)
    return Ftemp

######################
#  Fisher Bispectrum #
######################

# integrates and returns partial derivative of power spectrum wrt a parameter for a given k and fid val of param
def dbkpar(k,par):
    print datetime.datetime.now()

    global stri
    stri += 1;
    print "progress %.2f percent" % (100.*(stri/((float(len(param))/ncores)*len(trianglelist)))) # put len of dbfid instead ! ###################

    def f(y):
        return  DB_integrand(k,y[0],y[1],[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid],par)
    integ = vegas.Integrator([[qmin, qmax], [-1, 1]])
    result = integ(f, nitn=ni, neval = ne) # 1 more than the rest
    #print result.summary()
    return result.mean



def dbk(k): # return an array of the derivatives of P wrt to each param for given k

    return [ dbkpar(k,x) for x in param ]


def compute_dbfid(): #computes the derivatives of p wrt each param over the list of k
    global dbfid

    if os.path.isfile(modelhere+'/temp/dbfid_'+shapehere+'.npz') and recdbfid == "no":
        data = np.load(modelhere+'/temp/dbfid_'+shapehere+'.npz')
        dbfid = data['dbfid'].tolist()
        data.close()
        print datetime.datetime.now()

        if len(dbfid) == len(trianglelist):
            print "dbfid loaded"

        #        print " len dbfid "
        #        print len(dbfid)

        elif len(dbfid) > 0:
            print "continuing dbfid computation"
            tmin = len(dbfid) #index of first triangle to be computed
            print dbfid
            print "tmin = %.1f" % tmin
            global stri
            print "stri before = %i " % stri
            stri = (float(len(param))/ncores)*len(dbfid);
            print "stri = %i " % stri
        else:
            print "empty file, computing dbfid (dB/dlambda)"
            dbfid = [] # accumulates the result
            tmin = 0 #index of the first triangle that needs to be included
    else:
        print "no previous file, computing dbfid (dB/dlambda)"
        dbfid = [] # accumulates the result
        tmin = 0

    if len(dbfid) != len(trianglelist):
        poolB = multiprocess.Pool(processes=ncores); # start a multiprocess

    #    print "length of dbfid = %i " % len(dbfid)
    #    print "len dbfid / chunksize = %.1f" % (len(dbfid)/chunksize)

        for i in range(int((len(dbfid)/chunksize)//1),1+int((len(trianglelist)/chunksize)//1)): #does not include the upper bound

        #        print "length of dbfid = %i " % len(dbfid)
        #        print "length of dbfid/chunksize = %i " % (len(dbfid)/chunksize)

            print "chunk %i out of %i " % (i+1,1+int((len(trianglelist)/chunksize)//1))

            if (i+1)*chunksize > len(trianglelist): #check length of list
                tmax = len(trianglelist)-1 #index of last triangle that has to be included
            else :
                tmax = (i+1)*chunksize-1 #index of last triangle in the list

            print "tmin = %i" % tmin
            print "tmax = %i" % tmax

            listhere = trianglelist[tmin:tmax+1] # returns the (tmin+1)th element of the list up to tmax'th element, tmax excluded

            print "list here length %i" % len(listhere)

            dbfidhere = poolB.map(dbk,listhere)

            #        print "dbfid here"
            #        print dbfidhere

            dbfid += dbfidhere

            #        print "dbfid after appending"
            #        print dbfid

            np.savez(modelhere+'/temp/dbfid_'+shapehere+'.npz',dbfid=np.asarray(dbfid))

            tmin = (i+1)*chunksize # set the minimum for the next chunk

            print "chunk %i done" % (i+1)

        poolB.close()

        #print dbfid
        print datetime.datetime.now()
        print "dbfid done"

def F_BB() : #computes the fisher with bispectrum data

    if os.path.isfile(modelhere+'/temp/fb_'+shapehere+'.npz') and recfb == "no":
        data = np.load(modelhere+'/temp/fb_'+shapehere+'.npz')
        Ftemp = data['fb']
        data.close()
        print "fb loaded"
        # print Ftemp
    else :
        print datetime.datetime.now()
        print "computing fisher B"
        print "there are %i triangles" % len(trianglelist)
        print datetime.datetime.now()

        compute_pfid() # computes the powerspectrum if not yet done
        compute_dbfid() # same for the derivatives of b

        Ftemp = np.zeros([len(param), len(param)],dtype=float)

        for t in range(len(trianglelist)): # build fisher matrix from derivatives easier sequentially here
            for j in range(len(param)):
                for i in range(len(param)):
                    Ftemp[i,j] += (dbfid[t][i] * dbfid[t][j]) / var_b(trianglelist[t][0],trianglelist[t][1],trianglelist[t][2])

        print datetime.datetime.now()
        print "F_B done"
        print Ftemp
        np.savez(modelhere+'/temp/fb_'+shapehere+'.npz',fb=Ftemp)
    return Ftemp

def fisher(): # computes the fisher for shaperhere shape and datahere data
    #    load_model(modelname) # loads all definitions needed for the fisher // this furiously looks like a class instention
    if datahere == "P":
        Ftemp = F_PP()
    elif datahere == "B" :
        Ftemp = F_BB()
    elif datahere == "P+B" :
        Ftemp = F_BB()+F_PP()
    else :
        print "wrong data name"
    for i in range(0,len(param)): # add priors on param
        Ftemp[i,i]+=priors[i];
    return Ftemp


####################
#  squeezed only   #
####################
###### derivatives of squeezd spectrum ######
# integrates and returns partial derivative of power spectrum wrt a parameter for a given k and fid val of param
def dpkpar_sq(k,par):
    def f(y):
        return  DP_sq_integrand(k,y[0],y[1],[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid],bng,par)
    integ = vegas.Integrator([[qmin, qmax], [-1, 1]])
    result = integ(f, nitn=ni, neval = 4*ne) # slightly more evaluations and iterations for the derivative to be sure to have about few percent accuracy
    #print result.summary()
    return result.mean

def dpk_sq(k): # return an array of the derivatives of P wrt to each param for given k
    print datetime.datetime.now()
    print " k = %.3f" % k
    return [ dpkpar_sq(k,x) for x in param ]

def compute_dpfid_squeezed(): #computes the derivatives of p wrt each param over the list of k pointlist
    global dpfid_sq

    if os.path.isfile(modelhere+'/temp/dpfid_'+shapehere+'_squeezed.npz') and recdpfid_sq == "no":
        data = np.load(modelhere+'/temp/dpfid_'+shapehere+'_squeezed.npz')
        dpfid_sq = data['dpfid_sq'].tolist()
        data.close()
        print "dpfid squeezed loaded"
    #print dpfid_sq
    else :
        print datetime.datetime.now()
        print "computing dpfid (dP/dlambda) squeezed: there are %i k's" % len(klist)

        poolP = multiprocess.Pool(processes=ncores); # start a multiprocess
        dpfid_sq = poolP.map(dpk_sq, klist)
        poolP.close()

        #print dpfid sq
        print datetime.datetime.now()
        print "dpfid squeezed done"
        np.savez(modelhere+'/temp/dpfid_'+shapehere+'_squeezed.npz',dpfid_sq=np.asarray(dpfid_sq))

def Fel_PP_squeezed(k): #element of the sum of the fisher matrix for given k, reads from computed quantities
    #print datetime.datetime.now()
    #print "computing k = %.3f" % k

    Ftemp = np.zeros([len(param), len(param)],dtype=float)

    for i in range(len(param)):                 #builds the fisher matrix from derivatives and the variance
        for j in range(len(param)):
            Ftemp[i,j] = dpfid_sq[klist.index(k)][i]*dpfid_sq[klist.index(k)][j]/var_p(k)

    return Ftemp

def F_PP_squeezed() : #computes the fisher with power spectrum data : about 35min (integrals)

    if os.path.isfile(modelhere+'/temp/fp_'+shapehere+'_squeezed.npz') and recfp == "no": # loads the fisher if already computed
        data = np.load(modelhere+'/temp/fp_'+shapehere+'_squeezed.npz')
        Ftemp = data['fp']
        data.close()
        print "fp squeezed loaded"
    # print Ftemp
    else :
        print datetime.datetime.now()
        print "computing fp squeezed"
        compute_pfid() # computes the ps
        compute_dpfid_squeezed() # computes the derivatives of ps
        print len(klist)
        print len(param)
        print len(pfid)
        print len(dpfid_sq[1])

        Ftemp = np.zeros([len(param), len(param)],dtype=float)

        poolP = multiprocess.Pool(processes=ncores); # start a multiprocess
        Ftemp = np.sum(poolP.map(Fel_PP_squeezed, klist ),axis=0); # parallel evaluate to speed up PP does not dep on shape so we take it out of the loop. for the defs to be shared, the B-pool needs to be after chosenshape has been set (in the loop)
        poolP.close()

        print datetime.datetime.now()
        print "F_P squeezed done"
        # print Ftemp
        np.savez(modelhere+'/temp/fp_'+shapehere+'_squeezed.npz',fp=Ftemp)
    return Ftemp

def F_BB_squeezed() : #computes the fisher with bispectrum data

    if os.path.isfile(modelhere+'/temp/fb_'+shapehere+'_squeezed.npz') and recfb == "no":
        data = np.load(modelhere+'/temp/fb_'+shapehere+'_squeezed.npz')
        Ftemp = data['fb']
        data.close()
        print "fb squeezed loaded"
    # print Ftemp
    else :
        print datetime.datetime.now()
        print "computing fisher B"
        print "there are %i triangles" % len(trianglelist)
        print datetime.datetime.now()

        compute_pfid() # computes the powerspectrum if not yet done
        compute_dbfid() # same for the derivatives of b

        Ftemp = np.zeros([len(param), len(param)],dtype=float)

        #computing indices of squeezed triangles with factor 10

        indices_squeezed = ()

        for t in range(len(trianglelist)):
            if min(trianglelist[t][2],trianglelist[t][1]) >= 10*trianglelist[t][0] :
                indices_squeezed = np.append(indices_squeezed, t)

        #print indices_squeezed.astype(int)

        for t in indices_squeezed.astype(int): # build fisher matrix summing only squeezd triangles
            for j in range(len(param)):
                for i in range(len(param)):
                    Ftemp[i,j] += (dbfid[t][i] * dbfid[t][j]) / var_b(trianglelist[t][0],trianglelist[t][1],trianglelist[t][2])

        print datetime.datetime.now()
        print "F_B squeezed done"
        #print Ftemp

        np.savez(modelhere+'/temp/fb_'+shapehere+'_squeezed.npz',fb=Ftemp)
    return Ftemp




def fisher_squeezed(): # computes the fisher for shaperhere shape and datahere data
    #    load_model(modelname) # loads all definitions needed for the fisher // this furiously looks like a class instention
    if datahere == "P":
        Ftemp = F_PP_squeezed()
    elif datahere == "B" :
        Ftemp = F_BB_squeezed()
    elif datahere == "P+B" :
        Ftemp = F_BB_squeezed()+F_PP_squeezed()
    else :
        print "wrong data name"
    for i in range(0,len(param)): # add priors on param
        Ftemp[i,i]+=priors[i];
    return Ftemp


#######################
##  Systematic shifts #
#######################


shift_list=[]
shift_indices=[]

# create a list with only the parameters which we can solve for
def compute_shift_list():
    global shift_list
    global shift_indices
    shift_list = [ x for x in param if (x not in ['R','sig','fnl'] )]
    shift_indices = range(len(shift_list))
    print "shift_list"
    print shift_list
    print "shift_indices"
    print shift_indices

def integrate_func_index(name_of_function,index,k,nev): # integrates the corresponding coefficient over q and x for given index (param) in shift list
    print datetime.datetime.now()
    def f(y):
        return name_of_function(k,y[0],y[1],(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid),index)
    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
    result = integ(f, nitn=ni, neval=nev)
    print result.summary()
    #print datetime.datetime.now()

    return result.mean

def coeff_array(name_of_function,k,index_list): # return an array of the value of the coeff for all parameters in shift_list for a given k
    if type(k) == type(float()):
        print("k = %.5f" % k)
        nev = 4*ne # more evaluations for the power spectrum, only few k values
    else :
        print("triangle : (%.3f, %.3f %.3f)" % (k[0],k[1],k[2]))
        nev = ne # ne for the bispectrum, 24k triangles
    return [ integrate_func_index(name_of_function,ind,k,nev) for ind in index_list ]

def make_mappable(name_of_function,indices_here) : # makes a function of a single variable k for intergation in parallel
    def func_map(k):
        return coeff_array(name_of_function,k,indices_here)
    return func_map

# maps the function "name_of_function" to all "k" in "klist" and save the array to a file and returns the array
def map_to_list(name_of_function,list,index_list):

    if os.path.isfile(modelhere+'/temp/syst_'+shapehere+'_'+name_of_function.__name__+'.npz') : # if file exists, load it
        data = np.load(modelhere+'/temp/syst_'+shapehere+'_'+name_of_function.__name__+'.npz')
        res = data['res'].tolist()
        data.close()
        print datetime.datetime.now()
        print "coeff "+name_of_function.__name__+" loaded"
        #print res

        if len(res) == len(list):
            print name_of_function.__name__+" loaded"
        #        print " len list "
        #        print len(res)

        elif len(res) > 0:
            print "continuing computation of "+name_of_function.__name__
            tmin = len(res) #index of first triangle to be computed
            #print res
            print "tmin = %.1f" % tmin
            #global stri
            #print "stri before = %i " % stri
            #stri = (float(len(param))/ncores)*len(res);
            #print "stri = %i " % stri
        else:
            print "empty file, computing "+name_of_function.__name__
            res = [] # accumulates the result
            tmin = 0 #index of the first triangle/k that needs to be included

    else : # else, compute it
        print datetime.datetime.now()
        print "no previous file, computing "+name_of_function.__name__+" : there are %i k/triangles's" % len(list)
        res = [] # accumulates the result
        tmin = 0

    if len(res) != len(list): # compute it
        yo = make_mappable(name_of_function,index_list)
        # print "test"
        # print yo(0.1)
        poolres = multiprocess.Pool(processes=ncores); # start a multiprocess
################# loop over chuncks and save
        for i in range(int((len(res)/chunksize)//1),1+int((len(list)/chunksize)//1)): #does not include the upper bound

            #        print "length of dbfid = %i " % len(dbfid)
            #        print "length of dbfid/chunksize = %i " % (len(dbfid)/chunksize)

            print "chunk %i out of %i " % (i+1,1+int((len(list)/chunksize)//1))

            if (i+1)*chunksize > len(list): #check length of list
                tmax = len(list)-1 #index of last triangle that has to be included
            else :
                tmax = (i+1)*chunksize-1 #index of last triangle in the list

            print "tmin = %i" % tmin
            print "tmax = %i" % tmax

            listhere = list[tmin:tmax+1] # returns the (tmin+1)th element of the list up to tmax'th element, tmax excluded

            # print "list here length %i" % len(listhere)
            # print "list here"
            #print listhere
            reshere = poolres.map(yo,listhere)

            # print "res here"
            # print reshere

            res += reshere

            #        print "res after appending"
            #        print res

            np.savez(modelhere+'/temp/syst_'+shapehere+'_'+name_of_function.__name__+'.npz',res=np.asarray(res))

            tmin = (i+1)*chunksize # set the minimum for the next chunk

            print "chunk %i done" % (i+1)
################ all chunks done
        poolres.close()
        print datetime.datetime.now()
        print name_of_function.__name__+" done"

    return res

def coefficients_ps_par(par,a,b,c): # computes the coefficients of the quadratic equation for a given shifted parameter. no RHS of eqn yet.

    #par = considered parameter
    # a, b, c = tables of the integrated a, b, c coefficients for each k in klist
    #print "hola"
    #parindex = shift_list.index(par) # index of current parameter
    #print "par index %i" % parindex

    A = 0. # will collect the sum over the k's
    B = 0.
    C = 0.

    # print "length a b c %i" % len(a)
    # print "length klist %i" % len(klist)
    # print "length dpfid %i" % len(dpfid)

    for i in range(len(klist)):
        print i
        A += a[i][shift_list.index(par)]*dpfid[i][param.index(par)]/var_p(klist[i])
        B += b[i][shift_list.index(par)]*dpfid[i][param.index(par)]/var_p(klist[i])
        C += c[i][shift_list.index(par)]*dpfid[i][param.index(par)]/var_p(klist[i])

#print [A,B,C]
    return [A,B,C]

def coefficients_bis_par(par,a0,a1,a2,a3): # computes the coefficients of the quadratic equation for a given shifted parameter: arguments:

    #par = considered parameter
    # a0, ... = tables of the integrated a0, ...  coefficients for each tri in tringlelist
    # deltafnl = targeted shift in fnl
    # Finv = inverse fisher matrix for the data and shape in question

    A0 = 0. # will collect the sum over the k's
    A1 = 0.
    A2 = 0.
    A3 = 0.

    #print "length a0 %i" % len(a0)

    #print "length dbfid %i" % len(dbfid)

    for i in range(len(trianglelist)):
        A0 += a0[i][shift_list.index(par)]*dbfid[i][param.index(par)]/var_b(trianglelist[i][0],trianglelist[i][1],trianglelist[i][2])
        A1 += a1[i][shift_list.index(par)]*dbfid[i][param.index(par)]/var_b(trianglelist[i][0],trianglelist[i][1],trianglelist[i][2])
        A2 += a2[i][shift_list.index(par)]*dbfid[i][param.index(par)]/var_b(trianglelist[i][0],trianglelist[i][1],trianglelist[i][2])
        A3 += a3[i][shift_list.index(par)]*dbfid[i][param.index(par)]/var_b(trianglelist[i][0],trianglelist[i][1],trianglelist[i][2])

#print [A0,A1,A2,A3]

    return [A0,A1,A2,A3]

# compute a b c only once... so give then as an input?
# or do all parameters systematically and that's it.

def shift(file): # computes the shift in a given parameter "par" leading to a systematic shift in fnl of deltafnl for a given data combination
    # file is the file to which we write the results

    #print "data here %s, shape here %s" % (datahere,shapehere)
    #file.write(print "data here %s, shape here %s" % (datahere,shapehere))

    #print "compute shift list"
    compute_shift_list()
    #print "compute k and tri list"
    compute_list()

    F = fisher() # fisher for the current data
    Finv = linalg.inv(F);

    deltafnl = np.sqrt(Finv[param.index('fnl'),param.index('fnl')]) # syst shift in fnl that we want the shift in the parameter to generate, here equal to fnl marginalized

    compute_pfid() # be sure they are assigned


    if datahere == "P" :
        print("###### systematic shifts: P ######")
        file.write("----- data used: P ----- \n")

        compute_dpfid()

        a = map_to_list(a_integrand,klist,shift_indices) # loads if already computed, esle computes it.
        b = map_to_list(b_integrand,klist,shift_indices)
        c = map_to_list(c_integrand,klist,shift_indices)

        for par in shift_list:

            coe = coefficients_ps_par(par,a,b,c)
            sol = np.roots([coe[0],coe[1],coe[2]-deltafnl/Finv[param.index("fnl"),param.index(par)]])
            #  res = min(sol-parfid value)

            print "solution(s) for shift %s :" % par
            print sol-fiducial[param.index(par)]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s : \n" % par )
            (sol-fiducial[param.index(par)]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")

    if datahere == "B" :
        print("###### systematic shifts: B ######")
        file.write("----- data used: B ----- \n")

        compute_dbfid()

        a0 = map_to_list(a0_integrand,trianglelist,shift_indices)
        a1 = map_to_list(a1_integrand,trianglelist,shift_indices)
        a2 = map_to_list(a2_integrand,trianglelist,shift_indices)
        a3 = map_to_list(a3_integrand,trianglelist,shift_indices)

        for par in shift_list:

            coe = coefficients_bis_par(par,a0,a1,a2,a3)
            sol = np.roots([coe[3],coe[2],coe[1],coe[0]-deltafnl/Finv[param.index("fnl"),param.index(par)]])

            print "solution(s) for shift %s :" % par
            print sol-fiducial[param.index(par)]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s : \n" % par )
            (sol-fiducial[param.index(par)]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")

    if datahere == "P+B":
        print("###### systematic shifts: P+B ######")
        file.write("----- data used: P+B ----- \n")

        compute_dbfid()
        compute_dpfid()

        a = map_to_list(a_integrand,klist,shift_indices) # loads if already computed, esle computes it.
        b = map_to_list(b_integrand,klist,shift_indices)
        c = map_to_list(c_integrand,klist,shift_indices)

        a0 = map_to_list(a0_integrand,trianglelist,shift_indices)
        a1 = map_to_list(a1_integrand,trianglelist,shift_indices)
        a2 = map_to_list(a2_integrand,trianglelist,shift_indices)
        a3 = map_to_list(a3_integrand,trianglelist,shift_indices)

        for par in shift_list:

            coe_p = coefficients_ps_par(par,a,b,c)
            coe_b = coefficients_bis_par(par,a0,a1,a2,a3)
            sol = np.roots([coe_b[3],coe_b[2]+coe_p[0],coe_b[1]+coe_p[1],coe_b[0]+coe_p[2]-deltafnl/Finv[param.index("fnl"),param.index(par)]])

            print "solution(s) for shift %s :" % par
            print sol-fiducial[param.index(par)]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s : \n" % par )
            (sol-fiducial[param.index(par)]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")

# double shifts with bng constrained

def double_shift(file): # computes the shift in a given parameter "par" leading to a systematic shift in fnl of deltafnl for a given data combination
    # file is the file to which we write the results

    #print "data here %s, shape here %s" % (datahere,shapehere)
    #file.write(print "data here %s, shape here %s" % (datahere,shapehere))

    #print "compute shift list"
    compute_shift_list() # needs to be allocated

    #print "compute shift list"
    double_shift_list = (("b20","b11"),("b20","b02"),("b20","chi1"),("b20","w10"),("b11","b02"),("b11","chi1"),("b11","w10"),("b02","chi1"),("b02","w10"), ("chi1","w10"))  # combination of 2 parmaetners appearing in "bng" that we will solve for

    #print "compute k and tri list"
    compute_list()

    F = fisher() # fisher for the current data
    Finv = linalg.inv(F);

    deltafnl = np.sqrt(Finv[param.index('fnl'),param.index('fnl')]) # syst shift in fnl that we want the shift in the parameter to generate, here equal to fnl marginalized

    compute_pfid() # be sure they are assigned


    if datahere == "P" :
        print("###### systematic constrained shifts: P ######")
        file.write("----- data used: P ----- \n")

        compute_dpfid()

        a = map_to_list(a_integrand,klist,shift_indices) # loads if already computed, esle computes it.
        b = map_to_list(b_integrand,klist,shift_indices)
        c = map_to_list(c_integrand,klist,shift_indices)

        for pars in double_shift_list: # pars contained the 2 parameter in consideration

            (ax,bx,cx) = coefficients_ps_par(pars[0],a,b,c) # coefficient for 1st param
            (ay,by,cy) = coefficients_ps_par(pars[1],a,b,c) # coefficient for 2nd param
            Fx = 1./Finv[param.index("fnl"),param.index(pars[0])] # fisher factor for 1st and second
            Fy = 1./Finv[param.index("fnl"),param.index(pars[1])]

            (da,db,dc) = double_shift_abc(bng,sigma0,sigma1,sigma2,b20fid, b11fid, b02fid, chi1fid, w10fid,Fx,ax,bx,cx,Fy,ay,by,cy,double_shift_list.index(pars))

            sol = np.roots([da,db,dc-deltafnl])

            print "solution(s) for double shift %s, compensated by %s" % pars
            print sol-fiducial[param.index(pars[0])]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s, compensated by %s \n" % pars )
            (sol-fiducial[param.index(pars[0])]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")

    if datahere == "B" :
        print("###### systematic constrained shifts: B ######")
        file.write("----- data used: B ----- \n")

        compute_dbfid()

        a0 = map_to_list(a0_integrand,trianglelist,shift_indices)
        a1 = map_to_list(a1_integrand,trianglelist,shift_indices)
        a2 = map_to_list(a2_integrand,trianglelist,shift_indices)
        a3 = map_to_list(a3_integrand,trianglelist,shift_indices)

        for pars in double_shift_list: # pars contained the 2 parameter in consideration

            (a0x,a1x,a2x,a3x) = coefficients_bis_par(pars[0],a0,a1,a2,a3)
            (a0y,a1y,a2y,a3y) = coefficients_bis_par(pars[1],a0,a1,a2,a3)
            Fx = 1./Finv[param.index("fnl"),param.index(pars[0])] # fisher factor for 1st and second
            Fy = 1./Finv[param.index("fnl"),param.index(pars[1])]

            (da0,da1,da2,da3) = double_shift_a0123(bng,sigma0,sigma1,sigma2,b20fid,b11fid,b02fid,chi1fid,w10fid,Fx,a0x,a1x,a2x,a3x,Fy,a0y,a1y,a2y,a3y,double_shift_list.index(pars))

            sol = np.roots([da3,da2,da1,da0-deltafnl])

            print "solution(s) for double shift %s, compensated by %s" % pars
            print sol-fiducial[param.index(pars[0])]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s, compensated by %s \n" % pars )
            (sol-fiducial[param.index(pars[0])]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")


    if datahere == "P+B":
        print("###### systematic shifts: P+B ######")
        file.write("----- data used: P+B ----- \n")

        compute_dbfid()
        compute_dpfid()

        a = map_to_list(a_integrand,klist,shift_indices) # loads if already computed, esle computes it.
        b = map_to_list(b_integrand,klist,shift_indices)
        c = map_to_list(c_integrand,klist,shift_indices)

        a0 = map_to_list(a0_integrand,trianglelist,shift_indices)
        a1 = map_to_list(a1_integrand,trianglelist,shift_indices)
        a2 = map_to_list(a2_integrand,trianglelist,shift_indices)
        a3 = map_to_list(a3_integrand,trianglelist,shift_indices)

        for pars in double_shift_list:

            (ax,bx,cx) = coefficients_ps_par(pars[0],a,b,c) # coefficient for 1st param
            (ay,by,cy) = coefficients_ps_par(pars[1],a,b,c) # coefficient for 2nd param

            (a0x,a1x,a2x,a3x) = coefficients_bis_par(pars[0],a0,a1,a2,a3)
            (a0y,a1y,a2y,a3y) = coefficients_bis_par(pars[1],a0,a1,a2,a3)

            Fx = 1./Finv[param.index("fnl"),param.index(pars[0])] # fisher factor for 1st and second
            Fy = 1./Finv[param.index("fnl"),param.index(pars[1])]

            (da,db,dc) = double_shift_abc(bng,sigma0,sigma1,sigma2,b20fid, b11fid, b02fid, chi1fid, w10fid,Fx,ax,bx,cx,Fy,ay,by,cy,double_shift_list.index(pars))
            (da0,da1,da2,da3) = double_shift_a0123(bng,sigma0,sigma1,sigma2,b20fid,b11fid,b02fid,chi1fid,w10fid,Fx,a0x,a1x,a2x,a3x,Fy,a0y,a1y,a2y,a3y,double_shift_list.index(pars))

            sol = np.roots([da3,da2+da,da1+db,da0+dc-deltafnl])

            print "solution(s) for double shift %s, compensated by %s" % pars
            print sol-fiducial[param.index(pars[0])]
            print "delta fnl = %.3f" % deltafnl

            file.write("solution(s) for shift %s, compensated by %s \n" % pars )
            (sol-fiducial[param.index(pars[0])]).tofile(file, sep=", ", format="%s")
            file.write("\n")
            file.write("delta fnl = %.3f" % deltafnl)
            file.write("\n \n")

#################################
##  Chi2 simple ng - full gauss #
#################################

coeff_p_names = ("constant","b20","b20^2","b10","b10b20","fnl","fnlb20","fnlb10","fnlb10b20")
coeff_p_indices = ((0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))

coeff_b_names = ("constant","b20","b10","b10b20","b10^2","b10^2b20","b10^3","fnl","fnlb20","fnlb20^2","fnlb10","fnlb10b20","fnlb10b20^2","fnlb10^2","fnlb10^2b20","fnlb10^3")
coeff_b_indices = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(0,2,0),(0,2,1),(0,3,0),(1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2),(1,2,0),(1,2,1),(1,3,0))

# separating in 8 chunks for parallel evaluation
coeff_b_indices_1 = ((0,0,0),(0,0,1))
coeff_b_indices_2 = ((0,1,0),(0,1,1))
coeff_b_indices_3 = ((0,2,0),(0,2,1))
coeff_b_indices_4 = ((0,3,0),(1,0,0))
coeff_b_indices_5 = ((1,0,1),(1,0,2))
coeff_b_indices_6 = ((1,1,0),(1,1,1))
coeff_b_indices_7 = ((1,1,2),(1,2,0))
coeff_b_indices_8 = ((1,2,1),(1,3,0))
# we can reuse previous function to map chi2_b and chi2_p to triangle_list and k_list
# all what is left to do it combine all this to form the chi square and add them



def bigshift(file): # computes the coefficients of the chi2
    # file is the file to which we write the results

    print "data here %s, shape here %s" % (datahere,shapehere)
    file.write("data here %s, shape here %s" % (datahere,shapehere))

    #print "compute k and tri list"
    compute_list()

    compute_pfid() # be sure they are assigned to have the variance

    if datahere == "P" :
        delta_coeff = map_to_list(chi2_delta_p,klist,coeff_p_indices)

        chi2_prelist = map(chi2_coeff_p,delta_coeff)
        # print len(chi2_prelist)
        # print len(chi2_prelist[0])
        # print len(chi2_prelist[0][0])
        # print len(chi2_prelist[0][0][0])

        chi2_coeff = np.zeros((len(chi2_prelist[0]),len(chi2_prelist[0][0]),len(chi2_prelist[0][0][0])))
        # divide by the variance and sum
        for i in range(len(chi2_prelist)):
            #print chi2_prelist[i]/var_p(klist[i])
            chi2_coeff += chi2_prelist[i]/var_p(klist[i])

        # print "result"
        # print len(chi2_coeff)
        # print len(chi2_coeff[0])
        # print len(chi2_coeff[0][0])

        print "table of coefficients for chi2 with p"
        print chi2_coeff

        print >> file, "table of coefficients for chi2 with p \n", repr(chi2_coeff)


    if datahere == "B" :
        delta_coeff = map_to_list(chi2_delta_b,trianglelist,coeff_b_indices)

        chi2_prelist = map(chi2_coeff_b,delta_coeff)
        # print len(chi2_prelist)
        # print len(chi2_prelist[0])
        # print len(chi2_prelist[0][0])
        # print len(chi2_prelist[0][0][0])

        chi2_coeff = np.zeros((len(chi2_prelist[0]),len(chi2_prelist[0][0]),len(chi2_prelist[0][0][0])))
        # divide by the variance and sum
        for i in range(len(chi2_prelist)):
            #print chi2_prelist[i]/var_p(klist[i])
            chi2_coeff += chi2_prelist[i]/var_b(trianglelist[i][0],trianglelist[i][1],trianglelist[i][2])

        # print "result"
        # print len(chi2_coeff)
        # print len(chi2_coeff[0])
        # print len(chi2_coeff[0][0])

        print "table of coefficients for chi2 with b"
        print chi2_coeff

        print >> file, "table of coefficients for chi2 with b \n", repr(chi2_coeff)
