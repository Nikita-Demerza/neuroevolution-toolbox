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
    def __init__(self,tacts=10,genom=None,input_size=6,output_size=2):
        #3 ленты: входная, выходная, лента памяти
        mem_size = int(100)
        heads_mem = 2
        layers_desc = [{'type':'gru','out':input_size+2,'cells':5,'activation':'lrelu'},
                       
                           {'type':'ff','out':80,'activation':'lrelu'},
                           {'type':'modulable','out':80,'name':'mod1','activation':'linear'},
                           {'type':'turing_read','name':'memory_tape','heads':heads_mem,'cells':mem_size,'activation':'lrelu'},
                           {'type':'ff','out':80,'activation':'lrelu'},
                           {'type':'turing_move','name':'memory_tape','heads':heads_mem,'cells':mem_size},
                           {'type':'turing_write','name':'memory_tape','heads':heads_mem,'cells':mem_size},
                           
                           {'type':'ff','out':80,'activation':'lrelu'},
                           {'type':'modulator_inertial','out':80,'name':'mod1','activation':'linear'},
                           {'type':'ff','out':output_size,'activation':'lrelu'}]
        self.tacts = tacts#число тактов исполнения. Это ж типа МТ, так что надо дать несколько тактов
        self.nn = nnet.np_nn(layers_desc=layers_desc,in_size=input_size+2)
        
        if not (genom is None):
            self.nn.assemble_genom(genom)
    def init_output(self,output):
        shp = np.shape(output)
        if shp[0]>1:
            output = np.reshape(output,[1,shp[0]])
        self.nn.belts['out_tape'] = output
    def act(self,state,reward,done):
        #на входе state, чем бы он ни был - наблюдением в динамической задаче или геномом в статической
        #reward - это последний доступный reward
        #done - это последний доступный done
        #на выходе action - это всё содержимое выходной ленты.
        
        #для начала пишем в инпут-ленту
        shp = np.shape(state)
        if shp[0]>1:
            state = np.reshape(state,[1,shp[0]])
        self.nn.belts['input_tape'] = state
        #затем запустим прогноз, но не один раз, а циклом
        inp = np.concatenate([np.array([reward,done]),state[0]])
        in_ar = np.reshape(inp,[1,len(inp)])
        for i in range(int(self.tacts)):
            out = self.nn.predict_vector(in_ar)
        #затем прочтём, то на выходе
        return np.array(out)