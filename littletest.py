#def f(x) :
#    return 2.*x
#
#def g(x) :
#    return 3.*x
#
#def somef(name) :
#    return name(2)
#
#print somef(f)
#print somef(g)
#print somef(f)

#from functools import partial
#import multiprocessing
#from simplemodel import a_integrand
#
#def harvester(myfun, index, k):
#    X = case[0]
#    return myfun(index)
#
#partial_harvester = partial(harvester, case=RAW_DATASET)
#
#if __name__ == '__main__':
#    pool = multiprocessing.Pool(processes=6)
#    case_data = RAW_DATASET
#    pool.map(partial_harvester, case_data, 1)
#    pool.close()
#    pool.join()

param = ['fnl','a','b']
shiftlist = []

def compute_shift_list():
    shiftpara = param
    print "shiftpara"
    print shiftpara
    shiftpara.remove('fnl')
    shiftpara.remove('R')
    shiftpara.remove('fnl')
    global shift_list
    shift_list = shiftpara

compute_shift_list()
print shift_list