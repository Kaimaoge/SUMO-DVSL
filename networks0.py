#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 00:24:19 2018

@author: wuyuankai
"""

from __future__ import division
import os
import sys
import numpy as np

tools = '/usr/share/sumo/tools'
sys.path.append(tools)
import traci

sumoConfig = "floating_car.sumocfg"

class rm_vsl_co(object):
    '''
    this is a transportation network for training multi-lane variable speed limit and ramp metering control agents
    the simulation is running on sumo
    '''
    def __init__(self, test = False, visualization = False, control_horizon = 60, incidents = False):
        
        '''
        OD Parameters
        '''
        self.m1flow = np.round(np.array([3359+640,6007+1229,5349+1080,5563+1139,5299+1107]))
        self.r3flow = np.round(np.array([480,1153,1129,1176,1095]))
        self.m1a = [0.75,0.25]
        self.v_ratio = [0.1,0.1,0.4,0.4]
        
        
        '''
        Network Parameters
        '''
        self.edges = ['m3 m4 m5 m6 m7 m8 m9',\
                 'm3 m4 m5 m6 m7 m8 rout1',\
                 'rlight1 rin3 m7 m8 m9']        
        self.control_section = 'm6'
        self.state_detector = ['m5_0loop','m5_1loop','m5_2loop','m5_3loop','m5_4loop','m5_5loop',\
                               'm7_0loop','m7_1loop','m7_2loop','m7_3loop','m7_4loop','m7_5loop']
        self.VSLlist = ['m6_0','m6_1','m6_2','m6_3','m6_4']
        self.inID = ['m3_0loop','m3_1loop','m3_2loop','m3_3loop','m3_4loop','rlight1_0loop']
        self.outID = ['m9_0loop','m9_1loop','m9_2loop','m9_3loop','m9_4loop','rout1_0loop']
        self.bottleneck_detector = ['m7_6loop','m7_7loop','m7_8loop','m7_9loop','m7_10loop','m7_11loop']
        
        '''
        Simulation Parameters
        '''
        self.simulation_hour = 5  #hours
        self.simulation_step = 0
        self.control_horizon = control_horizon  #seoncs
        self.test = test
        self.visualization = visualization
        if self.visualization == False:
            self.sumoBinary = "/usr/bin/sumo"
        else:
            self.sumoBinary = "/usr/bin/sumo-gui"
        
        self.incidents = incidents
        if self.incidents == False:
            self.incident_time = 2000000
            self.incident_length = 10000000  # it will never happened
        else:
            self.incident_time = np.random.randint(low = 1, high = self.simulation_hour * 3600 - 1800)
            self.incident_length = np.random.randint(low = 1, high = 600)
                
    def writenewtrips(self): 
        with open('fcd.rou.xml', 'w') as routes:
            routes.write("""<routes>""" + '\n')
            routes.write('\n')
            routes.write("""<vType id="type0" color="255,105,180" length = "8.0" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
            routes.write("""<vType id="type1" color="255,190,180" length = "8.0" carFollowModel = "IDM" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
            routes.write("""<vType id="type2" color="22,255,255" length = "3.5" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
            routes.write("""<vType id="type3" color="22,55,255" length = "3.5" carFollowModel = "IDM" speedFactor="normc(1,0.1,0.2,2)" lcSpeedGain = "1"/>""" + '\n')
            routes.write('\n')
            for i in range(len(self.edges)):
                routes.write("""<route id=\"""" + str(i) + """\"""" + """ edges=\"""" + self.edges[i] + """\"/> """ + '\n')
            temp = 0
            for hours in range(len(self.m1flow)):
                m_in = np.random.poisson(lam = int(self.m1flow[hours,]))
                r3_in = np.random.poisson(lam = int(self.r3flow[hours,]))
                vNum = m_in + r3_in
                dtime = np.random.uniform(0+3600*hours,3600+3600*hours,size=(int(vNum),))            
                dtime.sort()
                for veh in range(int(vNum)):
                    typev = np.random.choice([0,1,2,3], p = self.v_ratio)
                    vType = 'type' + str(typev)
                    route = np.random.choice([0,1,2], p =[m_in*self.m1a[0]/vNum, m_in*self.m1a[1]/vNum, r3_in/vNum])
                    routes.write("""<vehicle id=\"""" + str(temp+veh) + """\" depart=\"""" + str(round(dtime[veh],2)) + """\" type=\"""" + str(vType) + """\" route=\"""" + str(route) + """\" departLane=\""""'random'"""\"/>""" + '\n')        
                    routes.write('\n')
                temp+=vNum
            routes.write("""</routes>""")
        # reset incidents
            
    #####################  obtain state  #################### 
    def get_step_state(self):
        state_occu = []
        for detector in self.state_detector:
            occup = traci.inductionloop.getLastStepOccupancy(detector)
            if occup == -1:
                occup = 0
            state_occu.append(occup)
        return np.array(state_occu)
    
    #####################  set speed limit  #################### 
    def set_vsl(self, v):
        number_of_lane = len(self.VSLlist)
        for j in range(number_of_lane):
            traci.lane.setMaxSpeed(self.VSLlist[j], v[j])
            
    #####################  the out flow ####################         
    def calc_outflow(self):
        state = []
        statef = []
        for detector in self.outID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            state.append(veh_num)
        for detector in self.inID:
            veh_num = traci.inductionloop.getLastStepVehicleNumber(detector)
            statef.append(veh_num)
        return np.sum(np.array(state)) - np.sum(np.array(statef))
    
    #####################  the bottleneck speed ####################  
    def calc_bottlespeed(self):
        speed = []
        for detector in self.bottleneck_detector:
            dspeed = traci.inductionloop.getLastStepMeanSpeed(detector)
            if dspeed < 0:
                dspeed = 5                                              
                #The value of no-vehicle signal will affect the value of the reward
            speed.append(dspeed)
        return np.mean(np.array(speed))
    
    #####################  the CO, NOx, HC, PMx emission  #################### 
    def calc_emission(self):
        vidlist = traci.edge.getIDList()
        co = []
        hc = []
        nox = []
        pmx = []
        for vid in vidlist:
            co.append(traci.edge.getCOEmission(vid))
            hc.append(traci.edge.getHCEmission(vid))
            nox.append(traci.edge.getNOxEmission(vid))
            pmx.append(traci.edge.getPMxEmission(vid))
        return np.sum(np.array(co)),np.sum(np.array(hc)),np.sum(np.array(nox)),np.sum(np.array(pmx))
    
    #####################  a new round simulation  #################### 
    def start_new_simulation(self, write_newtrips = True):
        self.simulation_step = 0
        if write_newtrips == True:
            self.writenewtrips()
        sumoCmd = [self.sumoBinary, "-c", sumoConfig, "--start"]
        traci.start(sumoCmd)
      
    #####################  run one step: reward is outflow  #################### 
    def run_step(self, v):
        state_overall = 0
        reward = 0
        co = 0
        hc = 0
        nox = 0
        pmx = 0
        oflow = 0
        bspeed = 0
        self.set_vsl(v)
        for i in range(self.control_horizon):
            traci.simulationStep()
            self.simulation_step += 1
            if self.simulation_step == self.incident_time:
                vehid = traci.vehicle.getIDList()
                r_tempo = np.random.randint(0, len(vehid) - 1)
                self.inci_veh = vehid[r_tempo]
                self.inci_edge = traci.vehicle.getRoadID(self.inci_veh) # get incident edge
            if self.simulation_step > self.incident_time and self.simulation_step < self.incident_time + self.incident_length:
                traci.vehicle.setSpeed(self.inci_veh, 0)                       # set speed as zero, to simulate incidents
            state_overall = state_overall + self.get_step_state()
            oflow = oflow + self.calc_outflow()
            bspeed = bspeed + self.calc_bottlespeed()
             # the reward is defined as the outflow 
            co_temp, hc_temp, nox_temp, pmx_temp = self.calc_emission()
            co = co + co_temp/1000 # g
            hc = hc + hc_temp/1000 # g
            nox = nox + nox_temp/1000 #g
            pmx = pmx + pmx_temp/1000
        reward = reward + oflow/80 * 0.1 + bspeed/(30*self.control_horizon)*0.9
        return state_overall/self.control_horizon/100, reward, self.simulation_step, oflow, bspeed/self.control_horizon, co, hc, nox, pmx
    
    def close(self):
        traci.close()