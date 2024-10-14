import ROOT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from root_to_np import get_tree, get_df, rename_tree_branches


###############################################################
###################### FUNTCIONS SECTION ##########################
def beam_coords(i,params):
    d =0.20
    x0,y0,u = params
    xl = np.sign(xdata[i]-(-x0))*(d+9.82)/2
    yl = np.sign(ydata[i]-(-y0))*(d+9.82)/2
    return((xl - x0)*np.cos(u)+(yl-y0)*np.sin(u), -(xl - x0)*np.sin(u)+(yl-y0)*np.cos(u))

def chi2_i(i,params):
    xb, yb = beam_coords(i,params) 
    return ((xdata[i]-xb)/sigmax[i])**2+((ydata[i]-yb)/sigmay[i])**2

def chi2(params):
    return chi2_i(0,params)+chi2_i(1,params)+chi2_i(2,params)+chi2_i(3,params)

def weird_gaussian( x, p ):
    #c, mu1, mu2, sigma = params
    res =   p[0] * (np.exp( - (x[0] - p[1])**2.0 / (2.0 * p[3]**2.0) ) \
          + np.exp( - (x[0] - p[2])**2.0 / (2.0 * p[3]**2.0) ))
    return res

def weirdo_gaussian( x, *p ):
    c, mu1, mu2, sigma = p
    res =   c * (np.exp( - (x - mu1)**2.0 / (2.0 * sigma**2.0) ) \
          + np.exp( - (x - mu2)**2.0 / (2.0 * sigma**2.0) ))
    return res

def parabola( x, p ):
    res =   p[0] * (x[0]-p[1])**2 +p[2]
    return res

def parabolic( x, *p ):
    a, mu, b = p
    res =   a*(x-mu)**2+b
    return res

def sym_to_beam(xy , theta):
    x,y = xy
    return (x*np.cos(theta)+y*np.sin(theta), -x*np.sin(theta)+y*np.cos(theta))

############################## MAIN  ######################################

run = [627,632]
pad_name = [37,38,27,28]
#load data
data = []
for irun in run:
    tree = get_tree(irun)
    
    var_indexes = np.array([1,2,4,5,6,7])
    for index in range(4):
        var_indexes = np.append(var_indexes,[index*10+8,index*10+9,index*10+12,index*10+14]) # edw dialegeis poia columns thes sto df
    var_indexes = np.append(var_indexes,[67,65,66])
    
    data_single = get_df(tree,*var_indexes)
    
    old_names = ['mcp_gpeak','mcp_qall','mcp_t'] # allagi onomatwn opws me voleyei
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

mcp_overflow = np.logical_and((data.mcp1_gpeak>0)*(data.mcp1_gpeak<=0.74),
                             (data.mcp2_gpeak>0)*(data.mcp2_gpeak<=0.74))
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

#root canvas for display
c = ROOT.TCanvas()
ROOT.gStyle.SetOptFit(1111)
#tracks multiplicity
xytrks = ROOT.TH2F("xytrks",f'RUN {run}: All tracks multiplicity',100,0,50,100,0,50)

xytrks.GetXaxis().SetTitle("x [mm]")
xytrks.GetYaxis().SetTitle("y [mm]")

n = len(data[mcp_basic_cut])
x = data.iloc[:,-4][mcp_basic_cut].values
y = data.iloc[:,-3][mcp_basic_cut].values
w = np.ones(n)
xytrks.FillN(n,x,y,w)
#clear out areas with low number of trks (not enough statistics)
for i in range(xytrks.GetNbinsX()+1):
    for j in range(xytrks.GetNbinsY()+1):
        if xytrks.GetBinContent(i,j)<15:
            xytrks.SetBinContent(i,j,0)

xytrks.SetTitle(f'RUN{run}: xy tracks with multiplicity >15')
xytrks.Draw("colz")
c.Update()
input('Enter to continue ')

c.Clear()
c.Divide(2,2)

xytrksW = []

for which_mcp in range(2):
    h = ROOT.TH2F(f'xyW_mcp{which_mcp+1}',f'RUN {run}: xy tracks weighted by MCP(in pool {4+which_mcp}) charge',
                        100,0,50,100,0,50) 
    h.GetXaxis().SetTitle("x [mm]")
    h.GetYaxis().SetTitle("y [mm]")

    w = data.iloc[:,3*which_mcp+1][mcp_basic_cut].values
    h.FillN(n,x,y,w)

    h.Divide(xytrks)
    xytrksW.append(h)
    
    c.cd(2*which_mcp+1)
    xytrksW[which_mcp].Draw("lego2z")
    c.cd(2*which_mcp+2)
    xytrksW[which_mcp].Draw("colz")

c.Update()
input('Enter to continue ')

# we find the pads here!
xytrksW_mm = [] #weighted histos
pad_hist = []

for index in range(4):
    pad = index +1
    
    h1 = ROOT.TH2F(f'xytrksW{pad}',f'RUN{run}: xy tracks weighted by PAD {pad_name[index]} e-peak row charge',
                   100,0,50,100,0,50)
    h1.GetXaxis().SetTitle("x [mm]")
    h1.GetYaxis().SetTitle("y [mm]")

    n = len(data[mm_basic_cut[index]])
    x = data.iloc[:,-4][mm_basic_cut[index]].values
    y = data.iloc[:,-3][mm_basic_cut[index]].values
    w = data.iloc[:,7+index*4][mm_basic_cut[index]].values
    h1.FillN(n,x,y,w)
    
    h1.Divide(xytrks)
    
    xytrksW_mm.append(h1)
    h2 = h1.Clone()
    h2.SetName(f'pad{pad}_hist')
    h2.SetTitle(f' PAD {pad_name[index]}')
    for i in range(h2.GetNbinsX()+1):
        for j in range(h2.GetNbinsY()+1):
            if h2.GetBinContent(i,j)<2.:
                h2.SetBinContent(i,j,0)
    
    pad_hist.append(h2)


c.Clear()
c.Divide(2,2)
for i in range(4):
    c.cd(i+1)
    xytrksW_mm[3-i].Draw("colz")#"lego2z"
c.Update()
input('Enter to continue ')

c.Clear()
c.Divide(2,2)
for i in range(4):
    c.cd(i+1)
    pad_hist[3-i].Draw("surf3")#"lego2z"
c.Update()
input('Enter to continue ')

#constrain fit

#set initial data
xdata = np.zeros(4)
ydata = np.zeros(4)
sigmax = np.ones(4)
sigmay = np.ones(4)

for index in range(4):
    
    #pad centers initialised by mean of weighted hist
    xdata[index] = pad_hist[index].GetMean(1)
    ydata[index] = pad_hist[index].GetMean(2)
    #sigmas
    sumx = 0
    sumy = 0
    sumq = 0

    for i in range(pad_hist[index].GetNbinsX()):
        for j in range(pad_hist[index].GetNbinsY()):
            delta_bin = pad_hist[index].GetBinError(i,j)
            xi = pad_hist[index].GetXaxis().GetBinCenter(i)
            yi = pad_hist[index].GetYaxis().GetBinCenter(j)
            sumx += delta_bin**2*(xi-xdata[index])**2
            sumy += delta_bin**2*(yi-ydata[index])**2
            sumq += pad_hist[index].GetBinContent(i,j)

    sigmax[index] = np.sqrt(sumx/sumq)
    sigmay[index] = np.sqrt(sumy/sumq)
    
# minimize chi2 to get initial values of params
x0i = - (xytrksW[0].GetMean(1)+xytrksW[1].GetMean(1))/2
y0i = - (xytrksW[0].GetMean(2)+xytrksW[1].GetMean(2))/2
ui = 0.
initial_guess = [x0i,y0i,ui]
boundaries = [(-50.,0.),(-50.,0.),(-np.pi,np.pi)]

result = minimize(chi2, initial_guess,bounds=boundaries)
if result.success:
    x0i,y0i,ui = result.x
    print(result)
else:
    raise ValueError(result.message) 

xc_step1 = np.zeros(4)
yc_step1 = np.zeros(4)
xc_er = np.zeros(4)
yc_er = np.zeros(4)
xc_step2 = np.ones(4)
yc_step2 = np.ones(4)

for index in range(4):
    xc_step1[index], yc_step1[index] = beam_coords(index,result.x)

# fig,axes = plt.subplots(figsize=(10,10))
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.subplots_adjust(hspace=0.2)
# fig.suptitle(f'Projection to Y symmetrical axis in Local Frame', fontsize=15, fontweight='bold')

for alg_steps in range(20): #if in 20 steps not converged.. terminates
    #1st step of the algorithm
    h2 =[None] * 4
    projX =[None] * 4
    projY =[None] * 4

    for index in range(4):
        pad = index +1      

        h1 = ROOT.TH2F('',"",
                   28,xc_step1[index]-7.,xc_step1[index]+7.,28,yc_step1[index]-7.,yc_step1[index]+7.)
        h2[index] = ROOT.TH2F('',f"PAD {pad_name[index]} distribution in Local frame",
                   28,xc_step1[index]-7.,xc_step1[index]+7.,28,yc_step1[index]-7.,yc_step1[index]+7.)

        h2[index].GetXaxis().SetTitle(f'x symmetry axis')
        h2[index].GetYaxis().SetTitle(f'y symmetry axis')
        #update estimations about centers in beam frame
        xc_step1[index], yc_step1[index] = beam_coords(index,result.x)

        x = data.iloc[:,-4].values
        y = data.iloc[:,-3].values
        xsym, ysym = sym_to_beam((x,y),-ui)

        square_cut = np.logical_and(np.abs(xsym-xc_step1[index])<6., np.abs(ysym-yc_step1[index])<6.)
        n = len(data[mm_basic_cut[index]*square_cut])
    
        # x = data.iloc[:,-4][mm_basic_cut[index]*square_cut].values
        # y = data.iloc[:,-3][mm_basic_cut[index]*square_cut].values
    
        # xsym, ysym = sym_to_beam((x,y),-ui)
        xsym = xsym[mm_basic_cut[index]*square_cut]
        ysym = ysym[mm_basic_cut[index]*square_cut]
    
        w = np.ones(n)
        h1.FillN(n,xsym,ysym,w)
        
        w = data.iloc[:,7+index*4][mm_basic_cut[index]*square_cut].values
        h2[index].FillN(n,xsym,ysym,w)
    
        h2[index].Divide(h1)
        #clear out
        for i in range(h2[index].GetNbinsX()+1):
            for j in range(h2[index].GetNbinsY()+1):
                if h2[index].GetBinContent(i,j)<2.:
                    h2[index].SetBinContent(i,j,0)

        
        #projections to symmetry axis
        projX[index] = h2[index].ProjectionX(f'PAD{pad}_px ')
        projX[index].SetTitle(f'PAD {pad_name[index]} x projection (Local Frame)')
        projX[index].GetXaxis().SetTitle(f'x symmetry axis')
        projX[index].Scale(1./projX[index].Integral())

        projY[index] = h2[index].ProjectionY(f'PAD{pad}_py ')
        projY[index].SetTitle(f'PAD {pad_name[index]} y projection (Local Frame)')
        projY[index].GetXaxis().SetTitle(f'y symmetry axis')
        projY[index].Scale(1./projY[index].Integral())


    c.Clear()
    c.Divide(2,2)
    for i in range(4):
        c.cd(i+1)
        h2[3-i].Draw("surf3")#"lego2z"
    c.Update()
    input('Enter to continue ')

    for index in range(4):
        pad = index +1
        #fit
        mean = projX[index].GetMean()
        projX[index].SetAxisRange(mean-10,mean+10, 'X')
        fitf = ROOT.TF1("fitf", weird_gaussian,mean-6,mean+6,4)
        #fitf = ROOT.TF1("fitf", parabola,mean-6,mean+6,3)
    
        binmax = projX[index].GetMaximumBin()
        maximum = projX[index].GetXaxis().GetBinCenter(binmax)
        par = np.array([maximum,mean-2.,mean+2.,projX[index].GetStdDev()])
        #par = np.array([-projX[index].GetStdDev(),mean,maximum])
        fitf.SetParameters(par)

        fit_res = projX[index].Fit(fitf,"RSQ")

        mux = (fit_res.Parameter(1)+fit_res.Parameter(2))/2
        xc_er[index] = np.sqrt(fit_res.ParError(1)**2+fit_res.ParError(1)**2) #approximately
        print(f'fit X pars errors: {xc_er[index]}')
        
        # old technique with projections

        # size = projX[index].GetNbinsX()
        # bins_centers = np.zeros(size)
        # bins_counts = np.zeros(size)
        # bins_errors = np.zeros(size)

        # test = projX[index].Clone()
        # for i in range(size):
        #     bins_centers[i] = projX[index].GetXaxis().GetBinCenter(i)
        #     bins_counts[i] = projX[index].GetBinContent(i)
        #     bins_errors[i] = projX[index].GetBinError(i)
        #     if(bins_counts[i]>0.048):
        #         bins_errors[i] = 0.5
        #     test.SetBinError(i,bins_errors[i])
        
        
        # fit_res = test.Fit(fitf,"RSQ")
        # fitf.GetParameters(par)

        # mux = fit_res.Parameter(1)
        # xc_er[index] = fit_res.ParError(1)
        # print(f'fit X pars errors: {xc_er[index]}')
        
        # plt.subplot(2, 2, index + 1)
        # plt.errorbar(bins_centers, bins_counts, fmt='o',markersize=2.,xerr=0.25,color='black')
        # plt.errorbar(bins_centers[bins_errors<0.5], bins_counts[bins_errors<0.5],
        #     yerr= bins_errors[bins_errors<0.5], xerr=0.25,fmt='o',markersize=2.,color='black',
        #     label=f'PAD {pad_name[index]}')

        # x = np.linspace(bins_centers[bins_counts>0][0],bins_centers[bins_counts>0][6])
        # plt.plot(x,parabolic(x,*par),'r--',label=r'fit: $\alpha\cdot(x-\mu)^2+\beta$')

        # x = np.linspace(bins_centers[bins_counts>0][-8]+0.25,bins_centers[bins_counts>0][-1]-0.25)
        # plt.plot(x,parabolic(x,*par),'r--')

        # plt.xlim(mean-6.8,mean+6.8)
        # plt.xlabel('x symmetry axis [mm]')
        # plt.legend()

        # popt, pcov = curve_fit(weirdo_gaussian, bins_centers, bins_counts, p0= par)

        # mux = (popt[1]+popt[2])/2
        # xc_er[index] = np.sqrt(pcov[1,1]**2+pcov[2,2]**2)/2 #approximately
        # print(f'fit X pars errors: {xc_er[index]}')

        #same for y
        mean = projY[index].GetMean()
        projY[index].SetAxisRange(mean-10,mean+10, 'X')
        fitf = ROOT.TF1("fitf", weird_gaussian,mean-6,mean+6,4)
        #fitf = ROOT.TF1("fitf", parabola,mean-6,mean+6,3)
    
        binmax = projY[index].GetMaximumBin()
        maximum = projY[index].GetXaxis().GetBinCenter(binmax)
        par = np.array([maximum,mean-2.,mean+2.,projY[index].GetStdDev()])
        #par = np.array([-projY[index].GetStdDev(),mean,maximum])
        fitf.SetParameters(par)
    
        fit_res = projY[index].Fit(fitf,"RSQ")

        muy = (fit_res.Parameter(1)+fit_res.Parameter(2))/2
        yc_er[index] = np.sqrt(fit_res.ParError(1)**2+fit_res.ParError(1)**2)
        print(f'fit Y pars errors: {yc_er[index]}')

        # size = projY[index].GetNbinsX()
        # bins_centers = np.zeros(size)
        # bins_counts = np.zeros(size)
        # bins_errors = np.zeros(size)

        # test = projY[index].Clone()
        # for i in range(size):
        #     bins_centers[i] = projY[index].GetXaxis().GetBinCenter(i)
        #     bins_counts[i] = projY[index].GetBinContent(i)
        #     bins_errors[i] = projY[index].GetBinError(i)
        #     if(bins_counts[i]>0.05):
        #         bins_errors[i] = 0.5
        #     test.SetBinError(i,bins_errors[i])
        
        # fit_res = test.Fit(fitf,"RSQ")
        # fitf.GetParameters(par)

        # muy = fit_res.Parameter(1)
        # yc_er[index] = fit_res.ParError(1)
        # print(f'fit Y pars errors: {yc_er[index]}')

        # plt.subplot(2, 2, index + 1)
        # plt.errorbar(bins_centers, bins_counts, fmt='o',markersize=2.,xerr=0.25,color='black')
        # plt.errorbar(bins_centers[bins_errors<0.5], bins_counts[bins_errors<0.5],
        #     yerr= bins_errors[bins_errors<0.5], xerr=0.25,fmt='o',markersize=2.,color='black',
        #     label=f'PAD {pad_name[index]}')

        # x = np.linspace(bins_centers[bins_counts>0][0]+0.25,bins_centers[bins_counts>0][8])
        # plt.plot(x,parabolic(x,*par),'r--',label=r'fit: $\alpha\cdot(x-\mu)^2+\beta$')

        # x = np.linspace(bins_centers[bins_counts>0][-8]+0.25,bins_centers[bins_counts>0][-1]+0.125)
        # plt.plot(x,parabolic(x,*par),'r--')

        # plt.xlim(mean-6.8,mean+6.8)
        # plt.xlabel('y symmetry axis [mm]')
        # plt.legend()

        # popt, pcov = curve_fit(weirdo_gaussian, bins_centers, bins_counts, p0= par)
        # muy = (popt[1]+popt[2])/2
        # yc_er[index] = np.sqrt(pcov[1,1]**2+pcov[2,2]**2)/2 #approximately
        # print(f'fit Y pars errors: {yc_er[index]}')

        #independent est. for pad centers (beam frame)
        mux = h2[index].GetMean(1)
        muy = h2[index].GetMean(2)
        xc_step1[index], yc_step1[index] = sym_to_beam((mux,muy),ui)

         

   # display projections
    # c.Clear()
    # c.Divide(2,2)
    # for i in range(4):
    #     c.cd(i+1)
    #     projX[i].Draw()
    # c.Update()
    # input('Enter to continue ')
    

    # c.Clear()
    # c.Divide(2,2)
    # for i in range(4):
    #     c.cd(i+1)
    #     projY[i].Draw()
    # c.Update()
    # input('Enter to continue ')

    #update data pnts (input in minimization) according to step1
    xdata = xc_step1
    ydata = yc_step1

    #udate sigmas
    for index in range(4):

        #sigmas
        sumx = 0
        sumy = 0
        sumq = 0
        

        for i in range(pad_hist[index].GetNbinsX()):
            for j in range(pad_hist[index].GetNbinsY()):
                delta_bin = pad_hist[index].GetBinError(i,j)
                xi = pad_hist[index].GetXaxis().GetBinCenter(i)
                yi = pad_hist[index].GetYaxis().GetBinCenter(j)
                sumx += delta_bin**2*(xi-xdata[index])**2
                sumy += delta_bin**2*(yi-ydata[index])**2
                sumq += pad_hist[index].GetBinContent(i,j)

        sigmax[index] = np.sqrt(sumx/sumq)
        sigmay[index] = np.sqrt(sumy/sumq)
    
    # minimize chi2 to get param values
    initial_guess = [x0i,y0i,ui]

    result = minimize(chi2, initial_guess)
    if result.success:
        x0i,y0i,ui = result.x
        print(result)
    else:
        raise ValueError(result.message) 

    #estimations of step2
    for index in range(4):
        xc_step2[index], yc_step2[index] = beam_coords(index,result.x)

    #update(xc_step1,yc_step1)
    print(f'step1: x_center ={xc_step1} ')
    print(f'step1: y_center ={yc_step1} ')
    print(f'step2: x_center ={xc_step2} ')
    print(f'step2: y_center ={yc_step2} ')

    ftol = 1e-06 #default
    for i in range(len(result.x)):
        hess_inv_i = result.hess_inv[i,i]
        uncertainty_i = np.sqrt(max(1, abs(result.fun)) * ftol * hess_inv_i)
        print('p^{0} = {1:12.4e} ± {2:.1e}'.format(i, result.x[i], uncertainty_i))


    x_check = np.abs(xc_step1-xc_step2)<=np.abs(0.05) # errors put of 0.05-0.04 for converge
    y_check= np.abs(yc_step1-yc_step2)<=np.abs(0.04)

    if x_check[0]==True and y_check[0]==True:
        if np.all(x_check==x_check[0]) and np.all(y_check==y_check[0]):
            print('--------------------------------------')
            print('CONVERGENCE REACHED')
            print('--------------------------------------')
            centers = np.empty((2,4))
            centers[0]=xc_step2
            centers[1]=yc_step2
            print(centers)
            with open(f'/home/evridiki/Desktop/JULY_RUNS/run_{run[0]}_{run[1]}_info/pad_centers.txt', 'w') as f: #save results to txt
                np.savetxt(f,centers, delimiter=" ")
            exit(0)

print('--------------------------------------')
print('TOO MANY ITERATIONS WITHOUT CONVERGING')
print('--------------------------------------')     



