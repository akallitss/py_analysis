import ROOT
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.optimize import fsolve
from scipy.optimize import minimize, curve_fit

from root_to_np import get_tree, get_df, rename_tree_branches

####################### FUNCTIONS SECTION #####################
def double_gaus( x, *params ):
    (c1, mu, sigma1, c2, sigma2) = params
    res =   c1 * np.exp( - (x - mu)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu)**2.0 / (2.0 * sigma2**2.0) )
    return res
def simple_gaus(x,c,mu,sigma):
    res =   c * np.exp( - (x - mu)**2.0 / (2.0 * sigma**2.0) ) 
    return res

def double_sigma(x, *params):
    prob =  double_gaus(x,*params)/double_gaus(x,*params).sum()
    mu_double   = x.dot(prob)         # mean value
    mom2 = np.power(x, 2).dot(prob)  # 2nd moment
    var  = mom2 - mu_double**2        # variance
    sigma = np.sqrt(var) 
    return sigma

def t_range_cut(t1,t2, i):
    t_low = [-8500,-8500,-9800,-8800]
    t_up = [-6500,-6500,-7500, -6700]

    t = (t1-t2)*1000 #ps

    return np.logical_and(t>=t_low[i],t<=t_up[i])

def double_expo(x,*p):
    return np.exp(p[0]*x+p[1])+np.exp(p[2]*x+p[3])+p[4]

###############################################################

run = [627,632]
pad_name = [37,38,27,28]
color = np.array(['red','blue','purple','green'])
which_mcp = [1,1,2,2]

#what we already know
centers_from_fit = np.loadtxt('/home/evridiki/Desktop/JULY_RUNS/run_627_632_info/pad_centers.txt',
                delimiter=" ", unpack=False)

#root canvas for display
c = ROOT.TCanvas()
ROOT.gStyle.SetOptFit(1111)

# get events
data = []
for irun in run:
    tree = get_tree(irun)
    
    var_indexes = np.array([1,2,4,5,6,7])
    for index in range(4):
        var_indexes = np.append(var_indexes,[index*10+8,index*10+9,index*10+12,index*10+14,index*10+11]) 
    var_indexes = np.append(var_indexes,[67,65,66])
    
    data_single = get_df(tree,*var_indexes)
    
    old_names = ['mcp_gpeak','mcp_qall','mcp_t']
    new_names = ['mcp1_gpeak','mcp1_qall','mcp1_t']
    names = [old_names,new_names]
    data_single = rename_tree_branches(data_single,*names)
    
    print(len(data_single))
    data.append(data_single)

data = pd.concat(data)
print(data.head())
print(len(data))

#define the cuts
mm_basic_cut= []

mcp_overflow = np.logical_and((data.mcp1_gpeak>0.3)*(data.mcp1_gpeak<=0.73),
                             (data.mcp2_gpeak>0.3)*(data.mcp2_gpeak<=0.73))
chi2_cut = (data.chi2<=40)*(data.track_flag>0)
mcp_basic_cut = mcp_overflow*chi2_cut

for index in range(4):
    pad = index+1
       
    mm_gpeak = f'mm{pad}_gpeak'
    mm_qall = f'mm{pad}_qall'
    mm_qe = f'mm{pad}_qe'
    mm_qfit = f'mm{pad}_qfit'
    mm_overflow = (data[mm_gpeak]>0.)*(data[mm_gpeak]<0.36)*(data[mm_qe]>0)
    qe_line = (data[mm_qe]>30.*data[mm_gpeak])
    cut = np.logical_and(mm_overflow*mcp_basic_cut,qe_line)

    mm_basic_cut.append(cut)

fig,axes = plt.subplots(figsize=(12,12))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.2)
fig.suptitle('Signal arrival time for central tracks', fontsize=15, fontweight='bold')

cable_delay = np.empty(4)

for index in range(4):
    pad=index+1
    mm_gpeak = f'mm{pad}_gpeak'
    mm_qall = f'mm{pad}_qall'
    mm_t = f'mm{pad}_t'
    #mcp_t=f'mcp{which_mcp[index]}_t'
    mcp_t='mcp1_t'
    if(pad>2):
        mcp_t='mcp2_t'

    xc = centers_from_fit[0][index]
    yc = centers_from_fit[1][index]
    
    circle_cut = np.sqrt((data.iloc[:,-4]-xc)**2 + (data.iloc[:,-3]-yc)**2)<2.5
    mm_time_cut = mm_basic_cut[index]*(data[mm_gpeak]>0.01)*circle_cut

    sat = (data[mm_t][mm_time_cut] - data[mcp_t][mm_time_cut])*1000 #ps
    mean = np.average(sat,weights = np.abs(1/(sat-sat.mean())))
    sat = sat[np.abs(sat-mean)<=100] # times waaay out of range

    counts, edges = np.histogram(sat,bins=100)
    bin_centers = (edges[:-1] + edges[1:])/2
    y_errors = np.sqrt(counts)

    g2_par = np.array([350.,sat.mean(),10,50,30])

    param_bounds=([0.1,-np.inf,1,0.1,1],[np.inf,np.inf,np.inf,np.inf,np.inf])

    popt, pcov = curve_fit(double_gaus, bin_centers, counts, p0=g2_par, bounds = param_bounds)
    g2_par= popt
    
    plt.subplot(2, 2, 4-index )
    plt.errorbar(bin_centers, counts, yerr=y_errors, fmt='o',markersize=4.,
                color='black',label= f'PAD {pad_name[index]}')

    x = np.linspace(sat.mean()-80,sat.mean()+80,1600)
    sigma_double = double_sigma(x,*g2_par)
    
    plt.plot(x,double_gaus(x,*g2_par),color='blue',
         label='Double Gaussian fit:\n $\mu = {:.1f} ps$ \n $\sigma = {:.3f} ps$'.format(g2_par[1],sigma_double))
    plt.legend(loc='upper right')

    cable_delay[index] = g2_par[1]

# with open(f'/home/evridiki/Desktop/JULY_RUNS/run_{run[0]}_{run[1]}_info/cable_delay.txt', 'w') as f:
#     np.savetxt(f,cable_delay)
#(re-)open to ensure we have right cable delay for all runs
cable_delay = np.loadtxt(f'/home/evridiki/Desktop/JULY_RUNS/run_{run[0]}_{run[1]}_info/cable_delay.txt')

slew = []
for index in range(4):
    pad=index+1
    mm_gpeak = f'mm{pad}_gpeak'
    mm_qall = f'mm{pad}_qall'
    mm_t = f'mm{pad}_t'
    mcp_t=f'mcp{which_mcp[index]}_t'

    xc = centers_from_fit[0][index]
    yc = centers_from_fit[1][index]
    
    square_cut = np.logical_and(np.abs(data.iloc[:,-4]-xc)<7., np.abs(data.iloc[:,-3]-yc)<7.)
    mm_time_cut = mm_basic_cut[index]*(data[mm_gpeak]>0.02)*square_cut
    mm_time_cut =mm_time_cut*t_range_cut(data[mm_t],data[mcp_t],index)
    
    #a subset of our df
    t_fun_q = data[mm_time_cut][[mcp_t,mm_t,mm_qall]]

    # sort
    t_fun_q.sort_values(by=[mm_qall], inplace=True)
    #binning
    num_points = 45
    len_qbin = round(len(t_fun_q)/num_points)
    print('total events = ',len(t_fun_q), 'missed events = ',len(t_fun_q)-num_points*len_qbin)

    #slices of q-> gaus fit
    arr = np.empty((0,5))
    for j in range(num_points):
        start =j*len_qbin
        end = (j+1)*len_qbin
        if(end>len(t_fun_q)):
            end = len(t_fun_q)
        qall = t_fun_q.iloc[start:end,2].values
        t = (t_fun_q.iloc[start:end,1].values-t_fun_q.iloc[start:end,0].values)*1000
        
                
        qbin_hist = ROOT.TH1F(""," ", 100,t.mean()-5*t.std(),t.mean()+5*t.std())
        qbin_hist.FillN(len(t),t,np.ones(len(t)))
        result = qbin_hist.Fit('gaus','QS')

        if(j%11==0):
            c.Clear()
            qbin_hist.Draw()
            c.Update()
            input('Enter to continue ')

        arr = np.append(arr,
                        np.array([[qall.mean() , np.abs(qall[0]-qall.mean()), np.abs(qall[-1]-qall.mean()),
                                  result.Parameter(1), result.ParError(1)]]),
                        axis = 0)

    slew.append(arr)

 
    fig,axes = plt.subplots(3,1,figsize=(6,18))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(hspace=0.2)
    fig.suptitle(f'Slewing effect for PAD {pad_name[index]}', fontsize=15, fontweight='bold')

    q = slew[index][:,0]
    q_error=[slew[index][:,1],slew[index][:,2]]
    sat =  slew[index][:,3] 
    sat_error = slew[index][:,4]

    par = np.array([-0.5,5.,-0.5,5.,cable_delay[index]])

    par, pcov = curve_fit(double_expo, q, sat,sigma=sat_error, p0= par, absolute_sigma=True)
    print(f'pad{pad} : {par}')
    x = np.linspace(q[0]-0.01,q[-1]+1.,1000)
    y = double_expo(x,*par)

    axes[0].plot(x,y,color='black',linewidth='2',label='double exponential fit')   

    axes[0].errorbar(q,sat, yerr =sat_error ,xerr=q_error,fmt='o',
             label =f'PAD {pad_name[index]}',color = color[index], markersize=3. )
    axes[0].set_ylabel('Signal Arrival Time [ps]',loc='top')
    axes[0].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')
    axes[0].legend(prop={'size': 10})
    

    sat = sat-double_expo(q,*par)
    axes[1].errorbar(q,sat, yerr =sat_error ,xerr=q_error,fmt='o',
             label =f'PAD {pad_name[index]}',color = color[index], markersize=3. )
    axes[1].set_ylabel(r'$SAT_{cor} = SAT-f(q_{all}) $ [ps]',loc='top')
    axes[1].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')

    #correct for slew
    arr = np.empty((0,2))
    for j in range(num_points):
        start =j*len_qbin
        end = (j+1)*len_qbin
        if(end>len(t_fun_q)):
            end = len(t_fun_q)
        qe = t_fun_q.iloc[start:end,2].values
        t = (t_fun_q.iloc[start:end,1].values-t_fun_q.iloc[start:end,0].values)*1000-double_expo(qe,*par)
        
                
        qbin_hist = ROOT.TH1F(""," ", 100,t.mean()-5*t.std(),t.mean()+5*t.std())
        qbin_hist.FillN(len(t),t,np.ones(len(t)))
        result = qbin_hist.Fit('gaus','QS')

        if(j%11==0):
            c.Clear()
            qbin_hist.Draw()
            c.Update()
            input('Enter to continue ')

        arr = np.append(arr,np.array([[result.Parameter(2), result.ParError(2) ]]),axis = 0 )


    slew[index] = np.append(slew[index],arr,axis=1)
    print(slew[index].shape)

    res = slew[index][:,5]
    res_error = slew[index][:,6]

    axes[2].errorbar(q,res, yerr =res_error ,xerr=q_error,fmt='o',
             label =f'PAD {pad_name[index]}',color = color[index], markersize=3. )
    axes[2].set_ylabel('Time resolution [ps]',loc='top')
    axes[2].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')

    with open(f'/home/evridiki/Desktop/JULY_RUNS/run_{run[0]}_{run[1]}_info/slew_pad{pad}.txt', 'w') as f:
        np.savetxt(f,slew[index])

# we reapeat blindly.. aka independent of pad center
#first read the scalind params
p2 = np.loadtxt('/home/evridiki/Desktop/JULY_RUNS/run_627_632_info/scale_par.txt',
                delimiter=" ", usecols=(0))
p3 = np.loadtxt('/home/evridiki/Desktop/JULY_RUNS/run_627_632_info/scale_par.txt',
                delimiter=" ", usecols=(1))

t_fun_q=[]
for index in range(4):
    pad=index+1
    mm_gpeak = f'mm{pad}_gpeak'
    mm_qall = f'mm{pad}_qall'
    mm_t = f'mm{pad}_t'
    mcp_t=f'mcp{which_mcp[index]}_t'

    mm_time_cut = mm_basic_cut[index]*(data[mm_gpeak]>0.03)*np.logical_and(data[mm_qall]>0.2,data[mm_qall]<80.)
    mm_time_cut =mm_time_cut*t_range_cut(data[mm_t],data[mcp_t],index)  

    # scale of charges to get "global charge"
    q_scale = p2[index]*data[mm_time_cut][mm_qall]+p3[index]
    sat = (data[mm_time_cut][mm_t]-data[mm_time_cut][mcp_t])*1000 - cable_delay[index] +10.

    d = {'sat':sat,'q_scale':q_scale}
    df = pd.DataFrame(data=d)

    t_fun_q.append(df)

# we get all the events here
t_fun_q = pd.concat(t_fun_q)
# sort
t_fun_q.sort_values(by=['q_scale'], inplace=True)
#binning
num_points = 65
len_qbin = round(len(t_fun_q)/num_points)
print('total events = ',len(t_fun_q),'lenght of each q slice = ',len_qbin ,'missed events = ',len(t_fun_q)-num_points*len_qbin)

#slices of q-> gaus fit
slew_all = np.empty((0,5))
for j in range(num_points):
    start =j*len_qbin
    end = (j+1)*len_qbin
    if(end>len(t_fun_q)):
        end = len(t_fun_q)
    qe = t_fun_q.iloc[start:end,1].values
    t = t_fun_q.iloc[start:end,0].values
        
                
    qbin_hist = ROOT.TH1F(""," ", 100,t.mean()-5*t.std(),t.mean()+5*t.std())
    qbin_hist.FillN(len(t),t,np.ones(len(t)))
    result = qbin_hist.Fit('gaus','QS')

    if(j%11==0):
        c.Clear()
        qbin_hist.Draw()
        c.Update()
        input('Enter to continue ')

    slew_all = np.append(slew_all,
                     np.array([[qe.mean() , np.abs(qe[0]-qe.mean()), np.abs(qe[-1]-qe.mean()),
                                result.Parameter(1), result.ParError(1)]]),
                    axis = 0)


fig,axes = plt.subplots(3,1,figsize=(6,18))
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.2)
fig.suptitle('Slewing effect of all pads', fontsize=15, fontweight='bold')

q = slew_all[:,0]
q_error=[slew_all[:,1],slew_all[:,2]]
sat =  slew_all[:,3] 
sat_error = slew_all[:,4]

par = np.array([-0.5,5.,-0.5,5.,10.])

par, pcov = curve_fit(double_expo, q, sat,sigma=sat_error, p0= par, absolute_sigma=True)
x = np.linspace(q[0]-0.01,q[-1]+1.,1000)
y = double_expo(x,*par)

axes[0].plot(x,y,color='red',linewidth='2',label='double exponential fit')   

axes[0].errorbar(q,sat, yerr =sat_error ,xerr=q_error,fmt='o',
            label =f'Data from RUNS {run}',color = 'black', markersize=3. )
axes[0].set_ylabel('Signal Arrival Time [ps]',loc='top')
axes[0].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')
axes[0].legend(prop={'size': 10})
    

sat = sat-double_expo(q,*par)
axes[1].errorbar(q,sat, yerr =sat_error ,xerr=q_error,fmt='o',
            label =f'PAD {pad_name[index]}',color = 'black', markersize=3. )
axes[1].set_ylabel(r'$SAT_{cor} = SAT-f(q_{all}) $ [ps]',loc='top')
axes[1].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')

#correct for slew
arr = np.empty((0,2))
for j in range(num_points):
    start =j*len_qbin
    end = (j+1)*len_qbin
    if(end>len(t_fun_q)):
        end = len(t_fun_q)
    qe = t_fun_q.iloc[start:end,1].values
    t = t_fun_q.iloc[start:end,0].values-double_expo(qe,*par)
        
                
    qbin_hist = ROOT.TH1F(""," ", 100,t.mean()-5*t.std(),t.mean()+5*t.std())
    qbin_hist.FillN(len(t),t,np.ones(len(t)))
    result = qbin_hist.Fit('gaus','QS')

    if(j%11==0):
        c.Clear()
        qbin_hist.Draw()
        c.Update()
        input('Enter to continue ')

    arr = np.append(arr,np.array([[result.Parameter(2), result.ParError(2) ]]),axis = 0 )


slew_all = np.append(slew_all,arr,axis=1)
print(slew_all.shape)

res = slew_all[:,5]
res_error = slew_all[:,6]

axes[2].errorbar(q,res, yerr =res_error ,xerr=q_error,fmt='o',
            label =f'Data from RUNS {run}',color = 'black', markersize=3. )
axes[2].set_ylabel('Time resolution [ps]',loc='top')
axes[2].set_xlabel(r'total raw charge $q_{all}$ [pC]',loc='right')

with open(f'/home/evridiki/Desktop/JULY_RUNS/run_{run[0]}_{run[1]}_info/slew_all_pads.txt', 'w') as f:
    np.savetxt(f,slew_all)


plt.show()