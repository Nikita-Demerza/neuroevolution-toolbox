import torch
import nnet
import sys
import numpy as np
import pandas as pd
import copy

class nt_controller():
    #Сам контроллер. Содержит нейронку, ленты (ленты - часть нейронки)
    #имеет функцию "погнали"
    #имеет канал награды
    #у нейронки более одного входа. Это реализовано, чтобы канал награды не терялся. На входе FF, дальше read_belt. 
    #Есть ещё запись в ленту - это тоже особенность данного класса
    def __init__(self,tacts=10,genom=None,input_size=6,output_size=2,layers_desc=None):
        #3 ленты: входная, выходная, лента памяти
        mem_size = 300
        heads_mem = 2
        if layers_desc == None:
            layers_desc = [#{'type':'gru','out':input_size+2,'cells':5,'activation':'lrelu'},
                {'type':'conv','filter_size':[1,3,7,7],'stride':2,'padding':0,'activation':'lrelu'},
                {'type':'max_pool','pool_size':2,'stride':2,'activation':'lrelu'},
                {'type':'conv','filter_size':[1,1,5,5],'stride':2,'padding':0,'activation':'lrelu'},
                {'type':'conv','filter_size':[1,1,3,3],'stride':1,'padding':0,'activation':'lrelu'},
                {'type':'conv','filter_size':[1,1,3,3],'stride':1,'padding':0,'activation':'lrelu'},
                {'type':'flatten','out':260,'activation':'linear'},
                {'type':'connection_in','out':260,'name':'conn0','activation':'linear'},
                {'type':'connection_out','out':560,'name':'conn1','activation':'linear'},
                {'type':'connection_out','out':860,'name':'conn2','activation':'linear'},
                {'type':'connection_out','out':1160,'name':'conn3','activation':'linear'},
                {'type':'connection_out','out':1460,'name':'conn4','activation':'linear'},
                {'type':'connection_out','out':1460+output_size,'name':'conn5','activation':'linear'},
                #{'type':'connection_out','out':1760+output_size,'name':'conn5','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol1','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn1','activation':'linear'},
                {'type':'connection_out','out':600,'name':'conn2','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn3','activation':'linear'},
                {'type':'connection_out','out':1200,'name':'conn4','activation':'linear'},
                {'type':'connection_out','out':1200+output_size,'name':'conn5','activation':'linear'},
                #{'type':'connection_out','out':1500+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':1460+output_size,'name':'conn0','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol2','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn2','activation':'linear'},
                {'type':'connection_out','out':600,'name':'conn3','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn4','activation':'linear'},
                {'type':'connection_out','out':900+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':1200+output_size,'name':'conn1','activation':'linear'},
                #{'type':'connection_out','out':1500+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':1460+output_size,'name':'conn0','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol3','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn3','activation':'linear'},
                {'type':'connection_out','out':600,'name':'conn4','activation':'linear'},
                {'type':'connection_out','out':600+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':900+output_size,'name':'conn1','activation':'linear'},
                {'type':'connection_out','out':1200+output_size,'name':'conn2','activation':'linear'},
                #{'type':'connection_out','out':1500+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':1460+output_size,'name':'conn0','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol4','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn4','activation':'linear'},
                {'type':'connection_out','out':300+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':600+output_size,'name':'conn1','activation':'linear'},
                {'type':'connection_out','out':900+output_size,'name':'conn2','activation':'linear'},
                {'type':'connection_out','out':1200+output_size,'name':'conn3','activation':'linear'},
                #{'type':'connection_out','out':1500+output_size,'name':'conn5','activation':'linear'},
                {'type':'connection_out','out':1460+output_size,'name':'conn0','activation':'linear'},
                #{'type':'ff_tolerance','out':300,'name':'tol5','activation':'lrelu'},
                #{'type':'connection_in','out':300,'name':'conn5','activation':'linear'},
                #{'type':'connection_out','out':300+output_size,'name':'conn6','activation':'linear'},
                #{'type':'connection_out','out':600+output_size,'name':'conn1','activation':'linear'},
                #{'type':'connection_out','out':900+output_size,'name':'conn2','activation':'linear'},
                #{'type':'connection_out','out':1200+output_size,'name':'conn3','activation':'linear'},
                #{'type':'connection_out','out':1500+output_size,'name':'conn4','activation':'linear'},
                #{'type':'connection_out','out':1760+output_size,'name':'conn0','activation':'linear'},
                {'type':'ff_tolerance','out':output_size,'name':'tol5','activation':'lrelu'},
                {'type':'connection_in','out':output_size,'name':'conn5','activation':'linear'},]
        
        self.tacts = tacts#число тактов исполнения. Это ж типа МТ, так что надо дать несколько тактов
        self.nn = nnet.np_nn(layers_desc=layers_desc,in_size=input_size+2)
        
        if not (genom is None):
            self.nn.assemble_genom(genom)
        self.input_size = input_size
    def init_output(self,output):
        shp = torch.shape(output)
        if shp[0]>1:
            output = torch.reshape(output,[1,shp[0]])
        self.nn.belts['out_tape'] = output
    def act(self,state,reward,done):
        #на входе state, чем бы он ни был - наблюдением в динамической задаче или геномом в статической
        #reward - это последний доступный reward
        #done - это последний доступный done
        #на выходе action
        
        shp = (state).shape
        if shp[0]>1:
            state = torch.reshape(state,[1,*shp])
        inp = state[0]
        in_ar = torch.reshape(inp,[1,*inp.shape])
        for i in range(int(self.tacts)):
            out = self.nn.predict_vector(in_ar)
        return torch.tensor(out)