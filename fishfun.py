# in this module we put all the definitions of functions used in the other file

#maybe it would be better to use functions that pass all needed parameters everytime, without use of global variables but whatevers

#############
#  imports  #
#############
import numpy as np
from scipy import interpolate
import multiprocessing
from extras import set_active, cartesian, interp_list, set_shift
import matplotlib
from matplotlib import pyplot as plt
import vegas
import datetime
from tempfile import TemporaryFile
import os.path
from numpy import linalg, pi, sin

E = np.exp(1.)

# for integration loops and parallel
qmin = 0.0001
qmax = 10. #default
ncores = multiprocessing.cpu_count()-1
n=1.; # default consider every n*kf for the bispectrum . for the power specutrm it computes every kf. 
ni = 2; #number iterations
ne = 2000; #number of evaluations

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
dbfid = []
pointlistP = []
trianglelist =[]

recpfid = "no" # whether to recomputed pfid, fp, fb or load it from file if it already exists
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
chunksize = 100 # divide the triangle list in chunks of size "chunksize". carefull not to change the chunksize between runs!

#sets dedicated names for the fidvalues
def initialize(act,allfid,allpri,nn,kkhigh):
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
    n = nn
    compute_list()

def model_output() :
    print("#############################")
    print("#         NEW MODEL         #")
    print("#############################")
    print "the parameters in the model are : "+', '.join(param)
    print "     with fiducial values       :",fiducial
    print "redshift z = %f" % zhere
    print "qmax = %f, kmax = %f, n = %f" % (qmax, khigh, n)
    print "there are %i k's for the power specutrm" % len(pointlistP)
    print "there are %i triangles for the bispecutrm" % len(trianglelist)
    print "the chunksize for dbfid is %i" % chunksize


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
klist = pklin[:,0] # only the k list
plist = pklin[:,1] # only the p(k) list
pkint = interpolate.splrep(klist, plist) #cubic spline interpolation of scipy

def P(k) : # rename for convienience (this is the dimension full power spectrum)
    return interpolate.splev(k, pkint);


# importing and interpolating transfer function of matter : the dimension full power spectrum of delta is then P_delta(k,z) = Tk(z)^2 P_zeta(k) where <zeta(k) zeta(k') > = (2PI)^3 delta(k+k') P_zeta(z_inflation) to P_delta(k,z) which is what we need here as the templates are defined for <zeta zeta zeta >. No further normalization.
tklin = np.loadtxt('./pklin_tk.dat',skiprows=1)
tkint = interpolate.splrep(tklin[:,0], tklin[:,5])

def T(k) :   # rename for convienience. T relates the initial curvature power spectrum (dimension full 2 PI^2 As (k/k_*)^(ns-1) /k^3 T(k)^2 = P_prim T(k)^2 =  P(k)
    return interpolate.splev(k, tkint) #never zero, good

#plotting points, cubic interpolation
#xnew = np.arange(0,1,0.01) # new finer k list values where to evaluate for the figure
#ynewcub = interpolate.splev(xnew, tkint) # evaluation of the interpolated p(k) at these points
#plt.figure()
#plt.plot(tklin[:,0], tklin[:,5], 'x', xnew, ynewcub)
#plt.legend(['Data', 'Cubic Spline'])
#plt.axis([min(tklin[:,0]), max(tklin[:,0]), min(tklin[:,5]), max(tklin[:,5])])
#plt.title('Checking interpolation')
#plt.show()

#######################
# triangle in the sum #
#######################

#POWER SPECUTRM EVERY KF

# generating the list of triangles from k min to kmax with spacing kf

def compute_list():
    global pointlistP
    global trianglelist
    
    if pointlistP == [] :
        pointlistP = np.arange(klow,khigh,kf).tolist() # will be used in the sum for the PP fisher

# generating the list of triangles from k min to kmax with spacing kf
    if trianglelist ==  [] :
        pointlistB = np.arange(klow,khigh,n*kf).tolist() # will be used in the sum for the PP fisher
        listraw = cartesian((pointlistB, pointlistB, pointlistB))
        trianglelist = [s for s in listraw if ((2*max(s[0],s[1],s[2]) <= s[0]+s[1]+s[2]) and (s[0]<=s[1]<=s[2])   )] # keep only triangle inequality and with sides ordered by increasing length

###################
# variance and pt #
###################
# variance of reduced bispectrum function in redshift space// here
def Var2Factor(k1,k2,k3): #only the factor to be able to take the b1 derivativ in the full fisher case
    if k1 != k2 != k3: trianglevar = 1.;
    elif k1 == k2 != k3 or k1 == k3 != k2 or k2 == k3 != k1 : trianglevar = 2.;
    else : trianglevar = 6. ;
    return (2*np.pi)**3 * trianglevar*(pfid[pointlistP.index(k1)]*pfid[pointlistP.index(k2)]*pfid[pointlistP.index(k3)]) / (8. * np.pi**2 * n**3 * k1* k2*k3);

##########################
#  shapes and bispectra  #
##########################
# shapes with ns dependance 
def Fshape(k1,k2,k3):
    if shapehere == "none":
        return 0.
    if shapehere == "orthogonal":
        return  np.asarray(- T(k1) * T(k2) * T(k3)  * (18./5.) *  Azeta**2  * ( - (3./(k1**(4.-ns) * k2**(4.-ns))) - (3./(k1**(4.-ns)  * k3**(4.-ns))) - (3./(k2**(4.-ns)  * k3**(4.-ns))) - (2./(k1**(2.*(4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k3**(2.*(4.-ns)/3.))) + (3./(k1**((4.-ns)/3) *  k2**(2.*(4.-ns)/3.)  * k3**(4.-ns) ))  + (3./(k2**((4.-ns)/3.) *  k3**(2.*(4.-ns)/3.) * k1**(4.-ns))) + (3./(k3**((4.-ns)/3.) *  k1**(2.*(4.-ns)/3)  * k2**(4.-ns) ) ) + (3./(k3**((4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k1**(4.-ns)) )+ (3./(k2**((4.-ns)/3.)  * k1**(2*(4.-ns)/3.)  * k3**(4.-ns) )) + (3./(k1**((4.-ns)/3.)  * k3**(2.*(4.-ns)/3.)  * k2**(4.-ns))) ))
    elif shapehere == "equilateral":
        return  np.asarray(- T(k1) * T(k2) * T(k3)  * (18./5.) *  Azeta**2  * ( - (1./(k1**(4.-ns) * k2**(4.-ns))) - (1./(k1**(4.-ns)  * k3**(4.-ns))) - (1./(k2**(4.-ns)  * k3**(4.-ns))) - (2./(k1**(2.*(4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k3**(2.*(4.-ns)/3.))) + (1./(k1**((4.-ns)/3) *  k2**(2.*(4.-ns)/3.)  * k3**(4.-ns) ))  + (1./(k2**((4.-ns)/3.) *  k3**(2.*(4.-ns)/3.) * k1**(4.-ns))) + (1./(k3**((4.-ns)/3.) *  k1**(2.*(4.-ns)/3)  * k2**(4.-ns) ) ) + (1./(k3**((4.-ns)/3.)  * k2**(2.*(4.-ns)/3.) *  k1**(4.-ns)) )+ (1./(k2**((4.-ns)/3.)  * k1**(2*(4.-ns)/3.)  * k3**(4.-ns) )) + (1./(k1**((4.-ns)/3.)  * k3**(2.*(4.-ns)/3.)  * k2**(4.-ns))) ))
    elif shapehere == "local":
        return np.asarray( - T(k1) * T(k2) * T(k3)  * (6./5.) *  Azeta**2  * ( (1./(k1**(4.-ns) * k2**(4.-ns))) + (1./(k1**(4.-ns)  * k3**(4.-ns))) + (1./(k2**(4.-ns)  * k3**(4.-ns))) ))
    else :
        print "wrong shape name"
        return 0.

#########################
# Fisher Power spectrum #
#########################
# fisher element for the power spectrum. terms need to be integrated over q and x for each k after fixing the value of the fiducial parameters. // for now this is only the  dP/dlambda factors

##importing definitions corresponding to the correct model
#def load_model(modelname):
#    if modelname == "full":
#        from fullmodel import P_integrand as P_modell
#        global P_integrand
#        P_integrand = P_modell
#        from fullmodel import DP_integrand as DP_modell
#        global DP_integrand
#        DP_integrand = DP_modell
#        from fullmodel import B_integrand as B_modell
#        global B_integrand
#        B_integrand = B_modell
#        from fullmodel import DB_integrand as DB_modell
#        global DB_integrand
#        DB_integrand = DB_modell
#    
#    elif modelname == "simple":
#        from simplemodel import P_integrand as P_modell
#        global P_integrand
#        P_integrand = P_modell
#        from simplemodel import DP_integrand as DP_modell
#        global DP_integrand
#        DP_integrand = DP_modell
#        from simplemodel import B_integrand as B_modell
#        global B_integrand
#        B_integrand = B_modell
#        from simplemodel import DB_integrand as DB_modell
#        global DB_integrand
#        DB_integrand = DB_modell
#    else :
#        print "wrong model name"

from simplemodel import P_integrand, DP_integrand, B_integrand, DB_integrand
#from fullmodel import P_integrand, DP_integrand, B_integrand, DB_integrand


# integrates and returns power spectrum for a given k general values of parameters (for use in "shift")
def pk(k,(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid)):
    # print "k = %f" % k
    def f(y):
        return P_integrand(k,y[0],y[1],[fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid])
    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
    result = integ(f, nitn=ni, neval=ne)
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
        poolP = multiprocessing.Pool(processes=ncores); # compute ptot (pointlist) if not already computed
        ptemp = poolP.map(pfidk, pointlistP)
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
    result = integ(f, nitn=ni, neval = 1.5*ne) # slightly more evaluations and iterations for the derivative to be sure to have about few percent accuracy
    print result.summary()
    return result.mean

def dpk(k): # return an array of the derivatives of P wrt to each param for given k
    print datetime.datetime.now()
    print " k = %f.3" % k
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
        print "computing dpfid (dP/dlambda): there are %i k's" % len(pointlistP)
        
        poolP = multiprocessing.Pool(processes=ncores); # start a multiprocess
        dpfid = poolP.map(dpk, pointlistP)
        poolP.close()
        
        #print dpfid
        print datetime.datetime.now()
        print "dpfid done"
        np.savez(modelhere+'/temp/dpfid_'+shapehere+'.npz',dpfid=np.asarray(dpfid))

######## fisher power spectrum ########

def var_p(k):
    return (pfid[pointlistP.index(k)])**2 * kf**2 / (2. * np.pi * n * k**2)

def Fel_PP(k): #element of the sum of the fisher matrix for given k, reads from computed quantities
    #print datetime.datetime.now()
    #print "computing k = %f.3" % k
    
    Ftemp = np.zeros([len(param), len(param)],dtype=float)
    
    for i in range(len(param)):                 #builds the fisher matrix from derivatives and the variance
        for j in range(len(param)):
            Ftemp[i,j] = dpfid[pointlistP.index(k)][i]*dpfid[pointlistP.index(k)][j]/var_p(k)

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
        print len(pointlistP)
        print len(param)
        print len(pfid)
        print len(dpfid[1])

        Ftemp = np.zeros([len(param), len(param)],dtype=float)
        
        poolP = multiprocessing.Pool(processes=ncores); # start a multiprocess
        Ftemp = np.sum(poolP.map(Fel_PP, pointlistP ),axis=0); # parallel evaluate to speed up PP does not dep on shape so we take it out of the loop. for the defs to be shared, the B-pool needs to be after chosenshape has been set (in the loop)
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
    result = integ(f, nitn=ni, neval = 1.5*ne) # 1 more than the rest
    print result.summary()
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
            print "tmin = %f" % tmin
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

    poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess

#    print "length of dbfid = %i " % len(dbfid)
#    print "len dbfid / chunksize = %f" % (len(dbfid)/chunksize)

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
                    Ftemp[i,j] += (dbfid[t][i] * dbfid[t][j]) / Var2Factor(trianglelist[t][0],trianglelist[t][1],trianglelist[t][2])
        
        print datetime.datetime.now()
        print "F_B done"
        print Ftemp
        np.savez(modelhere+'/temp/fb_'+shapehere+'.npz',fb=Ftemp)
    return Ftemp

def compute_fisher(): # computes the fisher for shaperhere shape and datahere data
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

#######################
#  Systematic shifts  #
#######################

def bkfid(k): # b(k) at fid values
    print datetime.datetime.now()
    def f(y):
        return B_integrand(k,y[0],y[1],(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid))
    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
    result = integ(f, nitn=ni, neval=ne)
    
    
    print result.summary()
    
    print datetime.datetime.now()
    
    return result.mean

def bkshift(k): # b(k) at shifted values
    def f(y):
        return B_integrand(k,y[0],y[1],[fnlshift ,b10shift, b20shift, b01shift, b11shift, b02shift, chi1shift, w10shift, sigshift, Rshift])
    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
    
    result = integ(f, nitn=ni, neval=ne)
    print result.summary()
    return result.mean

def compute_bfid(): # computes b(k) at fid values of parameters for each triangle of trianglelist and saves it
    global bfid
    
    if os.path.isfile(modelhere+'/temp/bfid_'+shapehere+'.npz') and recbfid == "no":
        data = np.load(modelhere+'/temp/bfid_'+shapehere+'.npz')
        bfid = data['bfid'].tolist()
        data.close()
        print "bfid loaded"
        #print bfid
    else :
        print datetime.datetime.now()
        print "computing bfid"

        poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
        bfid = poolB.map(bkfid, trianglelist)
        poolB.close()
        
        #print bfid
        print datetime.datetime.now()
        print "bfid done"
        np.savez(modelhere+'/temp/bfid_'+shapehere+'.npz',bfid=np.asarray(bfid))

def compute_pshift():
    global pshift
    print datetime.datetime.now()
    print "computing pshift"
    
    poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
    pshift = poolB.map(pkshift, pointlistP)
    poolB.close()
    
    #print pshift
    print datetime.datetime.now()
    print "pshift done"

def compute_bshift():
    global bshift
    print datetime.datetime.now()
    print "computing bshift"
    
    poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
    bshift = poolB.map(bkshift, trianglelist)
    poolB.close()
    
    #print bshift
    print datetime.datetime.now()
    print "bshift done"

def fnlfactor_p(shiftedparam, value) : #computes the shift of fnl when shiftedparam is shifted to value
    if os.path.isfile(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz') and recfnlp == "no":
        data = np.load(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz')
        fnltemp = data['fnlp'].tolist()
        data.close()
        print "fnlp loaded"
    #print fnlp
    else :
        set_shift(shiftedparam, value) # sets all _shift to fiducial value except shiftedparam which is set to value. sets "currentparam" to shiftedparam
        compute_pshift()
        compute_pfid()
        compute_dpfid()

        fnltemp = 0;
        for t in range(len(pointlistP)):
            fnltemp += (pshift[t]-pfid[t]) * dpfid[t][allparam.index(shiftedparam)] / var_p(pointlistP[t])
    
        np.savez(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz',fnlp=np.asarray(fnltemp))
    return fnltemp

def fnlfactor_b(shiftedparam, value) : #computes the shift of fnl when shiftedparam is shifted to value
    if os.path.isfile(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz') and recfnlb == "no":
        data = np.load(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz')
        fnltemp = data['fnlb'].tolist()
        data.close()
        print "fnlb loaded"
        #print fnlb
    else :
        set_shift(shiftedparam, value) # sets all _shift to fiducial value except shiftedparam which is set to value. sets "currentparam" to shiftedparam
        compute_bshift()
        compute_bfid()
        compute_pfid()
        compute_dbfid()
    
        fnltemp = 0;
        for t in range(len(trianglelist)):
            fnltemp += (bshift[t]-bfid[t]) * dbfid[t][allparam.index(shiftedparam)] /  Var2Factor(trianglelist[t][0],trianglelist[t][1],trianglelist[t][2])
        
        np.savez(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz',fnlb=np.asarray(fnltemp))
    return fnltemp

def compute_fnlshift(shiftedparam, value) : # shift in fnl
    fnltemp = 0;
    
    if datahere == "P":
        fnltemp = fnlfactor_p(shiftedparam, value)
    if datahere == "B":
        fnltemp = fnlfactor_b(shiftedparam, value)
    if datahere == "P+B":
        fnltemp = fnlfactor_p(shiftedparam, value) + fnlfactor_b(shiftedparam, value)
    
    fnltemp *= linalg.inv(compute_fisher())[allparam.index(shiftedparam),allparam.index("fnl")]

    return fnltemp

# returns a list of shifted fnl values (without the F inverse factor) corresponding to parameter values of interp_list( param )
def compute_fnlshift_list(shiftedparam) :

    fnllist = []

    for shiftvale in interp_list(shiftedparam):
        fnllist.append(compute_fnlshift(shiftedparam, shiftvale))
    
    return fnllist
