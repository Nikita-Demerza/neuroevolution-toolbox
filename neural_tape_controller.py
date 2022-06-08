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
                {'type':'connection_out','out':560,'name':'conn11','activation':'linear'},
                {'type':'connection_out','out':860,'name':'conn12','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol1','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn12','activation':'linear'},
                {'type':'connection_out','out':600,'name':'conn21','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn22','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol2','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn11','activation':'linear'},
                {'type':'connection_in','out':300,'name':'conn22','activation':'linear'},
                {'type':'connection_out','out':600,'name':'conn31','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn32','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol3','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn32','activation':'linear'},
                {'type':'connection_in','out':300,'name':'conn21','activation':'linear'},                           
                {'type':'connection_out','out':600,'name':'conn41','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn42','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol4','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn42','activation':'linear'}, 
                {'type':'connection_in','out':300,'name':'conn31','activation':'linear'},                           
                {'type':'connection_out','out':600,'name':'conn51','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn52','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol5','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn52','activation':'linear'}, 
                {'type':'connection_in','out':300,'name':'conn41','activation':'linear'},                           
                {'type':'connection_out','out':600,'name':'conn61','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn62','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol6','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn62','activation':'linear'}, 
                {'type':'connection_in','out':300,'name':'conn51','activation':'linear'},                            
                {'type':'connection_out','out':600,'name':'conn71','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn72','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol7','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn72','activation':'linear'}, 
                {'type':'connection_in','out':300,'name':'conn61','activation':'linear'},                           
                {'type':'connection_out','out':600,'name':'conn81','activation':'linear'},
                {'type':'connection_out','out':900,'name':'conn82','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol8','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn82','activation':'linear'}, 
                {'type':'connection_in','out':300,'name':'conn71','activation':'linear'},
                {'type':'ff_tolerance','out':300,'name':'tol9','activation':'lrelu'},
                {'type':'connection_in','out':300,'name':'conn81','activation':'linear'},       
                {'type':'ff_tolerance','out':output_size,'name':'tol10','activation':'lrelu'}]
        
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