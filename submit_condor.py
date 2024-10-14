import os
#import subprocess
from optparse import OptionParser
from os.path import join as pj
import subprocess as sp

def submit_job(job_script_path,jobN,log_Dir):

    # check what kind of file we got
    if not os.path.splitext(job_script_path)[1] == ".submit":
        # need to create the submit file
        job_script_base, _ = os.path.splitext(job_script_path)
        job_dir = os.path.dirname(job_script_path)
        submit_file_path = job_script_base + ".submit" # logs can't be stored in EOS
        #submit_file_path = "./.submit"
        print(submit_file_path)

        while True:
            try:
                with open(submit_file_path, 'w') as submit_file:
                    submit_file.write("executable = "+ job_script_path + "\n")
                    submit_file.write("universe = vanilla\n")
                    submit_file.write("output = "+log_Dir+"/job-"+str(jobN)+".output\n")
                    submit_file.write("error = "+log_Dir+"/job-"+str(jobN)+".error\n")
                    submit_file.write("log = "+log_Dir+"/job-"+str(jobN)+".log\n")
                    submit_file.write("+JobFlavour = \"nextweek\"\n")
                    submit_file.write("notification = never\n")
                    submit_file.write("queue 1")

                break
            except:
                print("problem writing job script -- retrying")
                time.sleep(10)

    else:
        # are given the submit file directly
        submit_file_path = job_script_path

    # while True:
    #     running_jobs = queued_jobs()
    #     if running_jobs < job_threshold:
    #         break
    #     print("have {} jobs running - wait a bit".format(running_jobs))
    #     time.sleep(30)
    while True:
        try:
            # call the job submitter
            sp.check_output(["condor_submit", submit_file_path])
            print("submitted '" + submit_file_path + "'")
            break
        except:
            print("Problem submitting job - retrying in 10 seconds!")
            time.sleep(10)

def queued_jobs(queue_status = "condor_q"):
    while True:
        try:
            running_jobs = len(sp.check_output([queue_status]).split('\n')) - 6
            return running_jobs
        except sp.CalledProcessError:
            print("{} error - retrying!".format(queue_status))
            time.sleep(10)

# ====================
# Analysis regions to submit

regions = ["CR_3l_BDT_multi_class_preMVA_3lNtuples_looseBtag_20GeV","CR_3l_BDT_multi_class_preMVA_3lNtuples_looseBtag_15GeV","CR_3l_BDT_multi_class_preMVA_3lNtuples_mixedBtag_15GeV","CR_3l_BDT_multi_class_preMVA_3lNtuples_mixedBtag_20GeV"]
additiona_flags= ":Systematics=NONE"
actions = "n:w:d" 

#====================================
# Create submit files (one per region)
wd = os.getcwd()
config_name = "combined_v8_withMVA_looseBtag_nominal_3l.config" 
config_path = "/afs/cern.ch/work/a/akotsoke/tth/TRExFitter/config/config_ttHML/"
setup_path = "/afs/cern.ch/work/a/akotsoke/tth/TRExFitter/"
config = pj(config_path, config_name) 
log_dir =config_name.split('.')[0]+"_SUBMIT"
submit_dir = log_dir + "/"

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

if not os.path.exists(pj(log_dir, 'submit_files')):
	os.makedirs(pj(log_dir, 'submit_files'))

# ====================
# one submit file per region
for i in range(len(regions)):
	with open(pj(log_dir, 'submit_files', 'cmd-REGION-%s.sh'%(regions[i])), 'w') as f:
		f.write("#!/bin/bash \n")
		f.write("source "+str(setup_path)+"/setup.sh  \n"); #setup environment
		f.write("trex-fitter \""+str(actions)+"\" "+str(config)+" \"Regions="+str(regions[i])+str(additiona_flags)+"\"") #command to run

# ====================
# Finally sumbit to condor

for jobi in range(len(regions)):
        submit_file_path = pj(log_dir, 'submit_files', 'cmd-REGION-%s.sh'%(regions[jobi]))
        submit_job(submit_file_path,str(regions[jobi]),log_dir)
