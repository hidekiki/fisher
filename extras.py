import numpy as np
import fishfun as ff
from numpy import linalg

# definition of cartesian product to generate the list of the triangles we want to sum over (http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays )
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


#######################
#  the survey class   #
#######################
# examples of used parameters
# euclid = Survey(1.60354*10**10,0.000399415,0.001,0.16,1,1)
# euclidlike = Survey(1.60354*10**10,0.000399415,0,0.16,1,1)
# old = Survey(10**10,0.005,0,0.17,1,1)


class Survey:
    
    def __init__(self, V, ng, kmin, kmax,n, nb):
        self.V = V # # volume of the survey in (Mpc/h)^3
        self.ng = ng # 0.000399415  # galaxy density for shot noise in  (h/Mpc)^3
        self.kf = ((2.*np.pi)/(self.V**(1./3.))) # for a 10Gpc^3 survey, took kf = 2PI/cubic root (V) in 1/(Mpc/h)
        if kmin < self.kf :
            self.kmin = self.kf # kmin smaller than kf is set to kf
        else:
            self.kmin =  kmin # kmin else
        self.kmax = kmax # 0.2 # default
        self.pointlistP = np.arange(self.kmin,self.kmax,n*self.kf).tolist() # list of k values, every n*kf
        #computes triangle list
        pointlistB = np.arange(self.kmin,self.kmax,nb*self.kf).tolist()
        listraw = cartesian((pointlistB, pointlistB, pointlistB))
        self.trianglelist = [s for s in listraw if ((2*max(s[0],s[1],s[2]) <= s[0]+s[1]+s[2]) and (s[0]<=s[1]<=s[2])   )]
    
    def display(self):
        print "the volume of the survey is %.0f (Mpc/h)^3, kf = %.4f, kmin = %.4f, kmax = %.4f" % (self.V, self.kf, self.kmin, self.kmax)
    
    def printtofile(self):
        print "print to file"



# takes a parameters and produces a list of 10 values between 0 and 2*values
def interp_list(shiftedparameter) :
    F = ff.compute_fisher()
    Finv = linalg.inv(F)
    paramindex = ff.allparam.index(shiftedparameter)
    fidparam = ff.fiducial[paramindex]
    sigmaparam = np.sqrt(Finv[paramindex,paramindex])
    temp = np.arange(fidparam-sigmaparam,fidparam+sigmaparam,2.*sigmaparam/10.); #  10 pts between fiducial value +/- 1 sigma marg
    return np.sort(temp)

# create list with only the relevant parameters
def set_active(act,all) :
    temp=[];
    for i in range(0,len(all)) :
        if act[i] == 1 :
            temp.append(all[i])
    return temp

# sets the parameter to the
def set_shift(shiftedparameter, value) :
    ff.currentparameter = shiftedparameter
    if shiftedparameter == "fnl":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [value,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "b10":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,value,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "b20":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,value,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "b01":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,value,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "b11":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,value,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "b02":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,value,ff.w10fid,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "w10":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,value,ff.Rfid,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "R":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,value,ff.sigfid,ff.chi1fid]
    elif shiftedparameter == "sig":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,value,ff.chi1fid]
    elif shiftedparameter == "chi1":
        [ff.fnlshift,ff.b10shift,ff.b20shift,ff.b01shift,ff.b11shift,ff.b02shift,ff.w10shift,ff.Rshift,ff.sigshift,ff.chi1shift] = [ff.fnlfid,ff.b10fid,ff.b20fid,ff.b01fid,ff.b11fid,ff.b02fid,ff.w10fid,ff.Rfid,ff.sigfid,value]
    else :
        print "wrong parametername"
##########################
#  Pk and Tk from CLASS  #
##########################

# definition of F2 kernel : used in the bispectrum expression
#def f2s(k,q,x):
#    return (5./7.) + (x / 2.) * ( (k/q) + (q/k)) + (2. * (x**2) /7.);

#def Ps(k,b10,b01) : # redshift space galaxy power spectrum # not used, everything in the Fisher
#    return a0P(k,b10,b01)*b1(k,b10,b01)*b1(k,b10,b01)*P(k);

#def Ptot(k,b10,b01) :
#    return Ps(k,b10,b01) + (1./ ng); # total power spectrum entering the variance b1 is at fid value, see def of the Fisher . In the full version one would leave it free then take derivative, not done in sefusatti & komatsu



########################
# scale dependent bias # # commented because definition is directly in the Fisher
########################
#def b1(k,b10,b01):
#    return b10 + b01*k**2;

##############################
# redshift space distorsions # # commented because definition is directly in the Fisher
##############################
# factors to produce the redshift space bispectrum and power spectrum
#def a0P(k,b10,b01) :
#    return 1 + (2./3.)*pow(Om,5./7.)/b1(k,b10,b01) + (1./5.)*(pow(Om,5./7.)/b1(k,b10,b01))**2;
#def a0B(b10,b01) :
#    return 1 + (2./3.)*pow(Om,5./7.)/b1(0.,b10,b01) + (1./9.)*(pow(Om,5./7.)/b1(0.,b10,b01))**2;
# here we make a0B non-k dependent to simplify


#############
# variance  #
#############

#def Var2Q(k1,k2,k3,b1): # commented because definition is directly in the Fisher
#    return  Var2Factor(k1,k2,k3)*(Ptot(k1,b1) * Ptot(k2,b1) * Ptot(k3,b1) )/(Ps(k1,b1)*Ps(k2,b1) + Ps(k2,b1)*Ps(k3,b1) +Ps(k3,b1)*Ps(k1,b1) )**2;

# variance of reduced bispectrum function in redshift space # commented because definition is directly in the Fisher
#def Var2B(k1,k2,k3,b10,b01):
#    return Var2Factor(k1,k2,k3)* (Ptot(k1,b10,b10) * Ptot(k2,b10,b01) * Ptot(k3,b10,b01) );

###########################################
##  Systematic shifts OLD WAY, PRECISION? #
###########################################
#
#def bkfid(k): # b(k) at fid values
#    print datetime.datetime.now()
#    def f(y):
#        return B_integrand(k,y[0],y[1],(fnlfid ,b10fid, b20fid, b01fid, b11fid, b02fid, chi1fid, w10fid, sigfid, Rfid))
#    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
#    result = integ(f, nitn=ni, neval=ne)
#    #print result.summary()
#    #print datetime.datetime.now()
#
#    return result.mean
#
#def bkshift(k): # b(k) at shifted values
#    def f(y):
#        return B_integrand(k,y[0],y[1],[fnlshift ,b10shift, b20shift, b01shift, b11shift, b02shift, chi1shift, w10shift, sigshift, Rshift])
#    integ = vegas.Integrator([[qmin, qmax], [-1.,1.]])
#
#    result = integ(f, nitn=ni, neval=ne)
#    #print result.summary()
#    return result.mean
#
#def compute_bfid(): # computes b(k) at fid values of parameters for each triangle of trianglelist and saves it
#    global bfid
#
#    if os.path.isfile(modelhere+'/temp/bfid_'+shapehere+'.npz') and recbfid == "no":
#        data = np.load(modelhere+'/temp/bfid_'+shapehere+'.npz')
#        bfid = data['bfid'].tolist()
#        data.close()
#        print "bfid loaded"
#        #print bfid
#    else :
#        print datetime.datetime.now()
#        print "computing bfid"
#
#        poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
#        bfid = poolB.map(bkfid, trianglelist)
#        poolB.close()
#
#        #print bfid
#        print datetime.datetime.now()
#        print "bfid done"
#        np.savez(modelhere+'/temp/bfid_'+shapehere+'.npz',bfid=np.asarray(bfid))
#
#def compute_pshift():
#    global pshift
#    print datetime.datetime.now()
#    print "computing pshift"
#
#    poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
#    pshift = poolB.map(pkshift, klist)
#    poolB.close()
#
#    #print pshift
#    print datetime.datetime.now()
#    print "pshift done"
#
#def compute_bshift():
#    global bshift
#    print datetime.datetime.now()
#    print "computing bshift"
#
#    poolB = multiprocessing.Pool(processes=ncores); # start a multiprocess
#    bshift = poolB.map(bkshift, trianglelist)
#    poolB.close()
#
#    #print bshift
#    print datetime.datetime.now()
#    print "bshift done"
#
#def fnlfactor_p(shiftedparam, value) : #computes the shift of fnl when shiftedparam is shifted to value
#    if os.path.isfile(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz') and recfnlp == "no":
#        data = np.load(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz')
#        fnltemp = data['fnlp'].tolist()
#        data.close()
#        print "fnlp loaded"
#    #print fnlp
#    else :
#        set_shift(shiftedparam, value) # sets all _shift to fiducial value except shiftedparam which is set to value. sets "currentparam" to shiftedparam
#        compute_pshift()
#        compute_pfid()
#        compute_dpfid()
#
#        fnltemp = 0;
#        for t in range(len(klist)):
#            fnltemp += (pshift[t]-pfid[t]) * dpfid[t][allparam.index(shiftedparam)] / var_p(klist[t])
#
#        np.savez(modelhere+'/temp/fnlp_'+shapehere+'_'+shiftedparam+'.npz',fnlp=np.asarray(fnltemp))
#    return fnltemp
#
#def fnlfactor_b(shiftedparam, value) : #computes the shift of fnl when shiftedparam is shifted to value
#    if os.path.isfile(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz') and recfnlb == "no":
#        data = np.load(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz')
#        fnltemp = data['fnlb'].tolist()
#        data.close()
#        print "fnlb loaded"
#        #print fnlb
#    else :
#        set_shift(shiftedparam, value) # sets all _shift to fiducial value except shiftedparam which is set to value. sets "currentparam" to shiftedparam
#        compute_bshift()
#        compute_bfid()
#        compute_pfid()
#        compute_dbfid()
#
#        fnltemp = 0;
#        for t in range(len(trianglelist)):
#            fnltemp += (bshift[t]-bfid[t]) * dbfid[t][allparam.index(shiftedparam)] /  Var2Factor(trianglelist[t][0],trianglelist[t][1],trianglelist[t][2])
#
#        np.savez(modelhere+'/temp/fnlb_'+shapehere+'_'+shiftedparam+'.npz',fnlb=np.asarray(fnltemp))
#    return fnltemp
#
#def compute_fnlshift(shiftedparam, value) : # shift in fnl
#    fnltemp = 0;
#
#    if datahere == "P":
#        fnltemp = fnlfactor_p(shiftedparam, value)
#    if datahere == "B":
#        fnltemp = fnlfactor_b(shiftedparam, value)
#    if datahere == "P+B":
#        fnltemp = fnlfactor_p(shiftedparam, value) + fnlfactor_b(shiftedparam, value)
#
#    fnltemp *= linalg.inv(fisher())[allparam.index(shiftedparam),allparam.index("fnl")]
#
#    return fnltemp
#
## returns a list of shifted fnl values (without the F inverse factor) corresponding to parameter values of interp_list( param )
#def compute_fnlshift_list(shiftedparam) :
#
#    fnllist = []
#
#    for shiftvale in interp_list(shiftedparam):
#        fnllist.append(compute_fnlshift(shiftedparam, shiftvale))
#
#    return fnllist
#

