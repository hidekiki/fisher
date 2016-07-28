#import vegas
#import math
import fishfun
import datetime
from numpy import linalg
from matplotlib import pyplot as plt
#from cubature import cubature
import numpy as np
import math
from numpy import pi, sin

# most general model, set all fid /prior values
#allfiducial = [0.,5.36,2.3,48.3,281.6,1864,-150.1,-403.8,5.76,3.6] # 10^14 m sun
allfiducial = [0.,0.454,-0.861,1.87,1.155,3.037,-2.024,-0.4821,5.76,0.8]# 10^12 msun
allpriors = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.] # priors

# all the models and combinations of data that we want to compute. for shape and data it's possible to specify more than 1 element : all combinations will be computed
models=[[["local"],["P","B","P+B"],[1., 1., 1., 1., 1., 1.,1., 1., 1., 1.],[0.,]]]

n = 1.
khigh = 0.16


x = 1;
y =2;

#########################
#    Loop over models   #
#########################
for m in models :
    shapenames= m[0]#("local","equilateral", "orthogonal"); # what shapes to consider, possible values : none, orthogonal, local, equilateral
    data = m[1]#("B", "P+B"); # what data to use, possible values: B, P, P+B
    active = m[2]#[1., 1., 1., 0., 0.] # choose which parameters to include in the model
    
    fishfun.initialize(active, allfiducial, allpriors,n,khigh); # initialization
    fishfun.model_output()
    param = fishfun.param # for convienience
    
    modelname = "model_"+"_".join(fishfun.param) #create model name
    
    fishfun.modelhere = modelname
    
    for chosenshape in shapenames :
        
        fishfun.shapehere = chosenshape # setting the shape in the other file
        
        for chosendata in data :
            
            fishfun.datahere = chosendata # setting the data used  in the other file
            
            print datetime.datetime.now()
            #            khere = 0.2
#            xmin = np.array([fishfun.qmin, -1.])
#            xmax = np.array([fishfun.qmax,1.])
#            val, err = cubature(fishfun.P_integrand, 2, 1, xmin, xmax, args = (khere,allfiducial),vectorized = True)
#            print('Approximated: {0}'.format(val))
## power spectrum ######

#tri = fishfun.trianglelist
            #print fishfun.B_integrand(tri[nn],0.1,0.5,allfiducial)
            #print fishfun.bkfid(tri[nn])
            #print fishfun.dbkpar(tri[nn],"b10")
            #print fishfun.dbk(tri)
            #print fishfun.compute_dbfid()
            fishfun.compute_dbfid()
        
           
            quit()
            #print datetime.datetime.now()
#a = np.array((0.1,0.2))

#fishfun.DP_integrand(a,0.15,"b02")

            #print fishfun.dpk(0.2,"b02")
            
            #print fishfun.pk(0.1,allfiducial)
            
            #print datetime.datetime.now()
            
            #plotting points, cubic interpolation
            
            plt.figure()
            plt.loglog(fishfun.pointlist, fishfun.pfid, 'x', fishfun.klist, fishfun.plist,'-')
            #plt.legend(['1loop','linear'])
            plt.axis([min(fishfun.klist), 1., min(fishfun.plist), 1.2*max(fishfun.pfid)])
            plt.show()

            #fishfun.set_shift("b02",0.2)
            
            # fishfun.compute_bshift()
            
            #fishfun.compute_fnlshift("fnl",0.2)

            #fishfun.compute_fnlshift_list("b02")
            
            quit()
        #print fishfun.modelhere
        
#        fishfun.compute_pfid() #compute ptotlsit for this shape
#        
#        fishfun.compute_dpfid()
#        
#        fishfun.compute_dbfid()

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