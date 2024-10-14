import numpy as np
import matplotlib.pyplot as plt
import ROOT


def fff(var, dt, xlow, xup, nevents, plot_all=False, trange=[0,0]): 
# This function takes the following parameters:
# var: An array representing a variable.
# dt: An array representing time or another measurement.
# xlow and xup: The lower and upper limits for filtering the data.
# nevents: The number of events or data points to process in each loop.
# plot_all: A boolean flag that controls whether or not to create and display plots.
# trange: A list containing two values that define the time range for data.

#The function processes the input data, calculates statistics, and creates histograms using the ROOT library. It stores the results in various arrays and returns them. If plot_all is True, it also generates and saves histograms as PDF files.

    var_in_limits = var[(var > xlow)*(var < xup)]
    dt_in_limits = dt[(var > xlow)*(var < xup)]
    
    var_ids = np.argsort(var_in_limits)
    
    var_in_limits = var_in_limits[var_ids]
    dt_in_limits  = dt_in_limits[var_ids]*1e3
    
    loops = var.size // nevents + 1
    
    valids   = np.empty(loops, dtype=np.int32)
    var_low  = np.zeros(loops)
    var_up   = np.zeros(loops)
    mean_dt  = np.zeros(loops)
    rms_dt   = np.zeros(loops)
    emean_dt = np.zeros(loops)
    erms_dt  = np.zeros(loops)
    
    valids.fill(-111)
    
#     fig, axs = plt.subplots(figsize=(22, 11))
#     if plot_all:
#         fig2, axs2 = plt.subplots(loops, 1, figsize=(22, 51))
    call = ROOT.TCanvas("call", '', 1300, 800)
    call.Divide(2,1)
    ROOT.gStyle.SetOptStat(111111)
    for i in range(0, loops):
#         ivar = var_in_limits[i*nevents:(i+1)*nevents]
#         idt = dt_in_limits[i*nevents:(i+1)*nevents]
        ivar, idt = None, None
        
        if i == loops - 1:
            ivar = var_in_limits[i*nevents:]
            idt = dt_in_limits[i*nevents:]
        else:
            ivar = var_in_limits[i*nevents:(i+1)*nevents]
            idt = dt_in_limits[i*nevents:(i+1)*nevents]
        
        if idt.size == 0:
            continue
        hht = ROOT.TH1D(f"hht{i}", '', 100, trange[0], trange[1])
        hht.FillN(idt.size, idt, 0)
        hht.Fit('gaus', 'Q')
        rt1 = hht.Fit('gaus', 'SQ')
        ppt = [-111, -111, -111]
        ept = [-111, -111, -111]
        if rt1.Get():
            ppt = rt1.GetParams()
#             ept = rt1.GetErrors()
#         else:
        ppt[1] = hht.GetMean()
        ppt[2] = hht.GetRMS()
        
        mm = ppt[1]
        sm = ppt[2]
        
#         mm = np.mean(idt)
#         sm = np.std(idt)
        r1 = mm - 4*sm
        r2 = mm + 4*sm
        
#         if np.abs(r1 - mm) > 30:
#             r1 = mm - 30
#         if np.abs(r2 - mm) > 30:
#             r2 = mm + 30
        
        vl, vm = np.min(ivar), np.max(ivar)
#         axs.hist(idt, bins=50, range=(-7700, -7400), histtype='step', lw=10, label=f'{vl:4.2f} < var < {vm:4.2f}')
#         axs.hist(idt, bins=50, range=(trange[0], trange[1]), histtype='step', lw=10, label=f'{vl:4.2f} < var < {vm:4.2f}')

#         if plot_all:
#             axs2[i].hist(idt, bins=50, histtype='step', range=(-7700, -7400), lw=10, label=f'{vl:4.2f} < var < {vm:4.2f}, loop={i}')
#             axs2[i].hist(idt, bins=50, histtype='step', range=(trange[0], trange[1]), lw=10, label=f'{vl:4.2f} < var < {vm:4.2f}, loop={i}')
        
        print(f'{np.min(idt)=}, {np.max(idt)=}, {r1=}, {mm=}, {r2=}')

#         cl.cd(i+1)
        

        hh1 = ROOT.TH1D(f"hhr{i}", '', 100, r1, r2)
        hh1.FillN(idt.size, idt, 0)

        hist_counts = np.zeros(hh1.GetNbinsX())
        hist_center = np.zeros(hh1.GetNbinsX())
        for ib in range(0, hh1.GetNbinsX()):
            hist_center[ib] = hh1.GetBinCenter(ib)
            hist_counts[ib] = hh1.GetBinContent(ib)

    #     print(np.stack((hist_counts, hist_center), axis=-1))
        new_xlow  = 0.0
        new_xup   = 0.0
        new_xmean = 0.0

        inew_xmean = np.argmax(hist_center)
        new_xmean = np.max(hist_center)

        for j in range(0, hist_counts.size):
            if hist_counts[j] > 5:
                new_xlow = hist_center[j]
                break
        for j in range(hist_counts.size-1, 0, -1):
            if hist_counts[j] > 5:
                new_xup = hist_center[j]
                break

        if np.isclose(new_xlow, new_xup):
            new_xlow = new_xmean*0.09
            new_xup  = new_xmean*1.01

#         print(f'{new_xlow=}, {new_xup=}')

    #     hh1 = ROOT.TH1D(f"hh{i}", '', 100, new_xlow, new_xup)
#         hh1 = ROOT.TH1D(f"hh{i}", '', 100, -7700, -7400)
#         hh1 = ROOT.TH1D(f"hh{i}", '', 100, trange[0], trange[1])
        hh1 = ROOT.TH1D(f"hh{i}", '', 100, r1, r2)
        hh1.FillN(idt.size, idt, 0)
        
        hh1.Fit("gaus", 'Q')
        rg1 = hh1.Fit("gaus", 'SQ')
        ppp = [-111, -111, -111]
        epp = [-111, -111, -111]
        if rg1.Get():
            ppp = rg1.GetParams()
            print(f'{ppp[0]}, {ppp[1]}, {ppp[2]}, ')
            epp = rg1.GetErrors()
        else:
            print('Oops no data')
        
        var_low[i]  = np.min(ivar)
        var_up[i]   = np.max(ivar)
        mean_dt[i]  = ppp[1]
        rms_dt[i]   = ppp[2]
        emean_dt[i] = epp[1]
        erms_dt[i]  = epp[2]
        valids[i]   = i
        print(50*'#')
        if plot_all:
            
            call.cd(1)
            hht.Draw()
            call.cd(2)
            hh1.Draw()
            call.SaveAs(f'dump/fff/params{i}.pdf')
#             if i == 0:
#                 call.Print("params.pdf(")
#             elif i == loops-1:
#                 call.Print("params.pdf)")
#             else:
#                 call.Print("params.pdf")
        
        
#     axs.legend()
#     fig.show()
    
    
#         call.Clear()
#         call.Draw()
#         for i in range(loops):
#             axs2[i].legend()
#         fig2.show()
    
    valid_loops = np.where(valids != -111)[0]
    var_low = var_low[valid_loops] 
    var_up  = var_up[valid_loops]
    mean_dt = mean_dt[valid_loops]
    rms_dt  = rms_dt[valid_loops]
    emean_dt = emean_dt[valid_loops]
    erms_dt = erms_dt[valid_loops]
    
    ROOT.gStyle.SetOptStat(1111)
    return var_low, var_up, mean_dt, rms_dt, emean_dt, erms_dt


def ffm(var, dt, xlow, xup, nevents, plot_all=False, trange=[0,0]):
    var_in_limits = var[(var > xlow)*(var < xup)]
    dt_in_limits = dt[(var > xlow)*(var < xup)]
    
    var_ids = np.argsort(var_in_limits)
    
    var_in_limits = var_in_limits[var_ids]
    dt_in_limits  = dt_in_limits[var_ids]*1e3
    
    loops = var.size // nevents + 1
    
    valids   = np.empty(loops, dtype=np.int32)
    var_low  = np.zeros(loops)
    var_up   = np.zeros(loops)
    mean_dt  = np.zeros(loops)
    rms_dt   = np.zeros(loops)
    emean_dt = np.zeros(loops)
    erms_dt  = np.zeros(loops)
    
    valids.fill(-111)
    
    fig, axs = plt.subplots(figsize=(22, 11))
    for i in range(0, loops):

        ivar, idt = None, None
        
        if i == loops - 1:
            ivar = var_in_limits[i*nevents:]
            idt = dt_in_limits[i*nevents:]
        else:
            ivar = var_in_limits[i*nevents:(i+1)*nevents]
            idt = dt_in_limits[i*nevents:(i+1)*nevents]
        
        if idt.size == 0:
            continue
               
        vl, vm = np.min(ivar), np.max(ivar)
        axs.hist(idt, bins=50, range=(trange[0], trange[1]), histtype='step', lw=10, label=f'{vl:4.2f} < var < {vm:4.2f}')

        hh1 = ROOT.TH1D(f"hh{i}", '', 100, trange[0], trange[1])
        hh1.FillN(idt.size, idt, 0)
        
        var_low[i]  = vl
        var_up[i]   = vm
        mean_dt[i]  = hh1.GetMean()
        rms_dt[i]   = hh1.GetRMS()
        emean_dt[i] = hh1.GetRMS()/np.sqrt(hh1.GetEntries())
        erms_dt[i]  = hh1.GetRMS()/np.sqrt(2*hh1.GetEntries())
        valids[i]   = i
#         print(50*'#')
        if plot_all:
            call = ROOT.TCanvas("call",'',1300, 800)
            hh1.Draw()
            if i == 0:
                call.Print("params.pdf(")
            elif i == loops-1:
                call.Print("params.pdf)")
            else:
                call.Print("params.pdf")

    fig.show()

    
    valid_loops = np.where(valids != -111)[0]
    var_low = var_low[valid_loops] 
    var_up  = var_up[valid_loops]
    mean_dt = mean_dt[valid_loops]
    rms_dt  = rms_dt[valid_loops]
    emean_dt = emean_dt[valid_loops]
    erms_dt = erms_dt[valid_loops]
    
    
    return var_low, var_up, mean_dt, rms_dt, emean_dt, erms_dt