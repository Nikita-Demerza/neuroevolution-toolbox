import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm

import copy
import warnings
warnings.filterwarnings('ignore')

def make_gaussian(m,s,l):
    #никакая не гауссиана, зато быстро
    #и на выходе не полноценная маска, а sparce
    if m>=l:
        m = l
    if m<0:
        m = 0
        
    s = np.abs(s)
    width = int(np.ceil(s))
    dx = 1./s
    gauss_raw = np.concatenate([np.arange(0,1,dx),np.arange(1,0,-dx)])
    out = np.zeros(l)
    out_start = int(m-s)
    gauss_start = 0
    out_end = int(m+s)
    gauss_end = int(s)+int(s)
    if out_start<0:
        gauss_start += -out_start
        out_start = 0
    if out_end>=l:
        gauss_end -= out_end - l
        out_end = l
    if out_start>out_end:
        out_m = out_start
        out_start = out_end
        out_end = out_m
    if gauss_start-gauss_end<=0:
        if gauss_start<l-1:
            gauss_end=gauss_start+1
        else:
            gauss_start = gauss_end-1
    if gauss_start-gauss_end!=out_start-out_end:
        delta = out_end - out_start
        if gauss_end+delta<l:
            gauss_end = gauss_start + delta
        else:
            gauss_start = gauss_end-delta
    if gauss_start<0:
        out_start += -gauss_start
        gauss_start = 0
    idx = (out_start,out_end)
    if np.sum(gauss_raw[gauss_start:gauss_end])==0:
        gauss_raw[gauss_start:gauss_end]=1
    return gauss_raw[gauss_start:gauss_end],idx


class np_nn:
    #example
    #layers_desc = [{'type':'gru','out':100,'cells':20,'activation':'logtan'},
    #                       {'type':'gru','out':70,'cells':5,'activation':'logtan'},
    #                       {'type':'ff','out':70,'activation':'relu'},
    #                       {'type':'ff','out':out_size,'activation':'logtan'}]
    def __init__(self,layers_desc,in_size,scale_weights=0.01):
        #layers_desc:[{'type':'gru','out':100,'cells':10,'activation':'relu'},{'type':'ff','out':50,'activation':'relu'}]
        #'ggru_out':global_gru, turing_read, turing_write
        #turing_read {'type':'turing_read','name':'zero_belt','heads':8,'cells':500,'activation':'relu'}
        #turing_move
        self.timetracking = False
        self.timetracking_last = pd.Timestamp.now()
        
        self.sin_count_for_attn = 20#
        self.sin_t_for_attn = 30
        #сколько мы можем покрыть чисел? Это что-то вроде бинарного кода, так что 30*2^(20-1) = 15,728,640
        
        #######layers: ff, gru, turing_read, turing_write, turing_move, ggru_out, modulable, modulator, modulator_inertial
        self.layers = []
        in_size_cur = in_size
        self.mult_actions = np.zeros(3,dtype=np.float16)
        self.activity_log = []
        self.global_cells = []
        self.global_cells_sz = 0
        self.belts = {}
        self.heads_m = {}
        self.heads_d = {}
        self.layers_desc = layers_desc
        for desc in layers_desc:
            layer = {}
            #последовательность работы: домножить на веса (размер меняется на out), применить нелинейность. Если gru, то передать в память через следующий набор весов
            if desc['type']=='gru':
                in_size_cur += desc['cells']
            layer['type'] = desc['type']
            #desc['out'] - это число "контактов" на выходе слоя
            #w_shape_in - это число "контактов" на выходе матрицы весов
            # вроде бы одинаковые вещи, но есть исключения
            if layer['type']=='turing_read':
                #на входе: ничего
                #на выходе: копия входа, взвешенные суммы с каждой головы,позиции каждой головы (через синус), позиции и дисперсии каждой головы
                desc['out'] = desc['heads']*(self.sin_count_for_attn+3) + in_size_cur
                w_shape_in = in_size_cur
            elif layer['type']=='turing_write':
                #на входе: что писать, с какой силой писать
                #на выходе: копия входа
                desc['out'] = in_size_cur
                w_shape_in = desc['heads']*2
            elif layer['type']=='turing_move':
                #на входе: куда двигать, какую дисперию ставить
                #на выходе: копия входа
                desc['out'] = in_size_cur
                w_shape_in = desc['heads']*2
            else:
                w_shape_in = desc['out']
            if (desc['type']!='modulator_inertial') and (desc['type']!='modulator') and (desc['type']!='modulable') and (desc['type']!='modulable_solid'):
                layer['w'] = np.array(np.random.normal(size=[in_size_cur,w_shape_in])*scale_weights, dtype=np.float16)
                layer['b'] = np.array(np.random.normal(size=[1, in_size_cur])*scale_weights, dtype=np.float16)
            elif (desc['type']=='modulable'):
                layer['w_modulable'] = np.array(np.random.normal(size=[1,in_size_cur])*scale_weights, dtype=np.float16)
            elif (desc['type']=='modulable_solid'):
                layer['w_modulable'] = np.array(np.random.normal(size=[1,1])*scale_weights, dtype=np.float16)
            if desc['type']=='gru':
                layer['w_mem'] = np.array(np.random.normal(size=[desc['out'],desc['cells']*2])*scale_weights, dtype=np.float16)#одни контакты - это что писать. Другие - насколько сильно.
                layer['cells_sz'] = desc['cells']
                layer['cells'] = np.zeros([1,desc['cells']],dtype=np.float16)
            elif desc['type']=='ggru_out':
                layer['w_mem'] = np.array(np.random.normal(size=[desc['out'],desc['cells']*2])*scale_weights, dtype=np.float16)#одни контакты - это что писать. Другие - насколько сильно.
                self.global_cells_sz = desc['cells']
                self.global_cells = np.zeros([1,desc['cells']],dtype=np.float16)
            elif desc['type']=='turing_read':
                #есть лента (и название), есть определённое число heads
                #лента 1-мерная
                if not desc['name'] in self.belts.keys():
                    self.belts[desc['name']] = np.zeros([1,desc['cells']],dtype=np.float16)
                    self.heads_m[desc['name']] = np.arange(0,desc['heads']*50,50)
                    self.heads_d[desc['name']] = np.zeros(desc['heads'],dtype=np.float16) + 5
                layer['belt_name'] = desc['name']
                #на выходе: копия входа, взвешенные суммы с каждой головы,позиции каждой головы (через синус)
            elif desc['type']=='turing_write':
                #на входе "что писать" и "как сильно писать"
                if not desc['name'] in self.belts.keys():
                    self.belts[desc['name']] = np.zeros([1,desc['cells']],dtype=np.float16)
                    self.heads_m[desc['name']] = np.arange(0,desc['heads']*10,10)
                    self.heads_d[desc['name']] = np.zeros(desc['heads'],dtype=np.float16) + 5
                layer['belt_name'] = desc['name']
            elif desc['type']=='turing_move':
                #как подвинуть и как выставить фокусировку
                layer['belt_name'] = desc['name']
            elif desc['type']=='modulable':
                #модулируемый слой. Его элементы равны w + wm, где wm - это выход модулятора
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = np.zeros([1,in_size_cur],dtype=np.float16)
            elif desc['type']=='modulable_solid':
                #модулируемый слой. Его элементы равны w + wm, где wm - это выход модулятора
                #у него единые threshold и k_amplif на весь слой
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = np.zeros([1,in_size_cur],dtype=np.float16)
            elif desc['type']=='modulator':
                #модулирующий слой. У него есть и ff-выход (чтобы сеть была сквозной), и запись в ленту.
                #Половина нейронов пишет в ленту, половина просто дальше
                layer['belt_name'] = desc['name']
            elif desc['type']=='modulator_inertial':
                #модулирующий слой. У него есть и ff-выход (чтобы сеть была сквозной), и запись в ленту.
                #Половина нейронов пишет в ленту, половина просто дальше
                layer['belt_name'] = desc['name']
                
            if not ('activation' in desc.keys()):
                desc['activation'] = 'relu'#max(0,x)
            layer['activation'] = desc['activation']
            
            self.layers.append(layer)
            in_size_cur = desc['out']
    def track_time(self,msg=''):
        if self.timetracking:
            now = pd.Timestamp.now()
            dt = int((now - self.timetracking_last).total_seconds() * 1e3)
            self.timetracking_last = now
            print(msg,dt)
    
    def predict(self, x):
        #формат x: (batch_size,seq_len,in_size)
        self.drop_state()
        return np.array([self.predict_sequence(x[i]) for i in range(np.shape(x)[0])])
    
    def predict_sequence(self, x):
        #формат x: (seq_len,in_size)
        return np.array([self.predict_vector(x[i]) for i in range(np.shape(x)[0])])
    
    def predict_vector(self, x):
        #формат x: (1,in_size)
        x = x.astype('float16')
        shp = np.shape(x)
        if shp[0]>1:
            x = np.reshape(x,[1,shp[0]])
        in_data = x
        if self.global_cells_sz>0:
            in_data = np.hstack([in_data,self.global_cells])
            #self.global_cells
        for i in range(len(self.layers)):
            self.track_time('start')
            layer = self.layers[i]
            if layer['type']=='gru':
                in_data = np.concatenate([in_data, layer['cells']],axis=1)
            in_data = np.clip(in_data, a_min=-1e9,a_max=1e9)#убрать бесконечности
            if 'w' in layer:
                y = np.dot(in_data + layer['b'],layer['w'])
            else:
                y = in_data
            y = np.clip(y, a_min=-1e9,a_max=1e9)#убрать бесконечности
            #нелинейность
            if layer['activation']=='relu':
                y = np.clip(y, a_min=0,a_max=np.inf)
            elif layer['activation']=='lrelu':
                y = np.clip(y, a_min=y*0.001,a_max=np.inf)
            elif layer['activation']=='logtan':
                y = np.arctan(y)/np.pi + 0.5
            elif layer['activation']=='linear':
                pass
            self.track_time('dot')
            #если gru, то запись в память
            if (layer['type']=='gru') or (layer['type']=='ggru_out'):
                self.track_time('gru_write')
                y_to_cell = np.dot(y,layer['w_mem'])
                y_what_write = y_to_cell[:,:layer['cells_sz']]
                y_how_write = 0.5 + np.arctan(y_to_cell[:,layer['cells_sz']:])/np.pi
                layer['cells'] = layer['cells']*(1-y_how_write) + y_what_write*y_how_write
                self.track_time('gru_write ended')
            if layer['type']=='gru':
                self.layers[i]['cells'] = layer['cells']
            elif layer['type']=='ggru_out':
                self.global_cells = layer['cells']
            elif layer['type']=='turing_read':
                self.track_time('turing_read start')
                #для каждой головы составить коэффициенты и посчитать свёртку
                heads_cnt = len(self.heads_m[layer['belt_name']])
                conv_list = np.zeros([1, heads_cnt],dtype=np.float16)
                m_list = np.zeros([1, heads_cnt],dtype=np.float16)
                s_list = np.zeros([1, heads_cnt],dtype=np.float16)
                #sin_positions = np.zeros([1, self.sin_count_for_attn*heads_cnt],dtype=np.float16)
                sin_positions = []
                for i in range(heads_cnt):
                    m = self.heads_m[layer['belt_name']][i]
                    s = self.heads_d[layer['belt_name']][i]
                    mask,nonzeros = make_gaussian(m,s,np.shape(self.belts[layer['belt_name']])[1])
                    conv = np.sum(self.belts[layer['belt_name']][0,nonzeros[0]:nonzeros[1]]*mask)
                    #print("self.belts[layer['belt_name']][0,nonzeros]",self.belts[layer['belt_name']][0,nonzeros])
                    #print('mask',mask)
                    conv_list[0,i] = conv
                    m_list[0,i] = m
                    s_list[0,i] = s
                    
                    #for j in range(self.sin_count_for_attn):
                    #    t = self.sin_t_for_attn*(2**j)
                    #    sin_positions[:,j] = np.sin(m/t)
                    sin_positions_add = [np.sin(m/self.sin_t_for_attn*(2**j)) for j in range(self.sin_count_for_attn)]
                    sin_positions.extend(sin_positions_add)
                    #print(np.shape(sin_positions))
                sin_positions = np.array(sin_positions,dtype=np.float16)
                sin_positions = np.reshape(sin_positions,[1,len(sin_positions)])
                #присобачить выход голов к y от прошлого слоя. И ещё приделать их позиции.
                self.track_time('turing_read ended')
                #print('conv_list',conv_list)
                y = np.hstack([y,conv_list,m_list,s_list,sin_positions])
                
            elif layer['type']=='turing_move':
                self.track_time('turing_move')
                #двигать головы############################################################################
                for i in range(len(self.heads_m[layer['belt_name']])):
                    maxmove = 10000
                    #амплитуда большая, но распределение такое: 
                    #f(0.05)=1.25,f(0.1)=10, f(0.2)=80, f(0.5)=1250, f(0.8)=5120
                    self.heads_m[layer['belt_name']][i] += (((np.arctan(in_data[0,2*i])/np.pi)*2)**3)*maxmove
                    maxd = 30
                    self.heads_d[layer['belt_name']][i] = (np.arctan(in_data[0,2*i+1])/np.pi + 0.5)*maxd
                y = in_data
                self.track_time('turing_move ended')
            elif layer['type']=='turing_write':
                self.track_time('turing_write')
                ############################################################################
                conv_list = np.zeros([1, len(self.heads_m)],dtype=np.float16)
                #for i in np.arange(0,2*len(self.heads_m[layer['belt_name']]),2):
                head_cnt = len(self.heads_m[layer['belt_name']])
                self.track_time('turing_write st')
                for i in range(head_cnt):
                    self.track_time('turing_write a')
                    #mask = np.arange(np.shape(self.belts[layer['belt_name']])[1])
                    m = self.heads_m[layer['belt_name']][i]
                    s = self.heads_d[layer['belt_name']][i]
                    
                    self.track_time('turing_write b')
                    #mask = np.exp(-((mask - m)**2)/(2*s**2))#Гаусс
                    mask,nonzeros = make_gaussian(m,s,np.shape(self.belts[layer['belt_name']])[1])
                    self.track_time('turing_write c')
                    mask = mask*(np.arctan(y[0,2*i])/np.pi + 0.5)#с какой силой писать
                    self.track_time(f'turing_write d mask: {len(mask)}')
                    value_wr = y[0,2*i+1]
                    if value_wr>1e5:
                        value_wr=1e5
                    elif value_wr<-1e5:
                        value_wr = -1e5
                    #print('nonzeros',nonzeros,'value_wr',value_wr,'mask',mask)
                    b = self.belts[layer['belt_name']][0,nonzeros[0]:nonzeros[1]]*(1-mask) + mask*value_wr
                    b = np.array(np.clip(b,-1e6,1e6),dtype=np.float16)
                    if np.any(np.isinf(b)):
                        b[:] = 0
                    self.belts[layer['belt_name']][:,nonzeros[0]:nonzeros[1]] = b
                    delta = b-self.belts[layer['belt_name']][0][nonzeros[0]:nonzeros[1]]
                    #print('wr',self.belts[layer['belt_name']][0][nonzeros[0]:nonzeros[1]],'b',b,'d',delta,' b[np.isinf(b)]', b[np.isinf(b)])
                    
                    if 0:
                        if layer['belt_name']=='out_tape':
                            print('--head',i)
                            print('m',m,'s',s)
                            print('mask,nonzeros',mask,nonzeros,'value_wr',value_wr)
                            print('b',b)
                            print("self.belts[layer['belt_name']][:,nonzeros[0]:nonzeros[1]]",self.belts[layer['belt_name']][:,nonzeros[0]:nonzeros[1]])
                            print("self.belts[layer['belt_name']]",self.belts[layer['belt_name']])
                    self.track_time('turing_write f')
                y = in_data
                
                self.track_time('turing_write ended')
            elif layer['type']=='modulator':
                min_len = np.min([int(len(np.ravel(in_data))/2), len(np.ravel(self.belts[layer['belt_name']]))])
                self.belts[layer['belt_name']][:,:min_len] = in_data[:,:min_len]
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data
            elif layer['type']=='modulator_inertial':
                min_len = np.min([int(len(np.ravel(in_data))/2), len(np.ravel(self.belts[layer['belt_name']]))])
                self.belts[layer['belt_name']][0,:min_len] += in_data[0,:min_len]
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data
            elif layer['type']=='modulable':
                idx = np.isinf(self.belts[layer['belt_name']])
                self.belts[layer['belt_name']][idx] = 1e4
                idx = self.belts[layer['belt_name']]>1e4
                self.belts[layer['belt_name']][idx] = 1e4
                idx = self.belts[layer['belt_name']]<-1e4
                self.belts[layer['belt_name']][idx] = -1e4
                min_len = np.min([int(len(np.ravel(in_data))), len(np.ravel(self.belts[layer['belt_name']]))])
                threshold = (layer['w_modulable'][0,0:min_len]+1)
            elif layer['type']=='modulable_solid':
                idx = np.isinf(self.belts[layer['belt_name']])
                self.belts[layer['belt_name']][idx] = 1e4
                idx = self.belts[layer['belt_name']]>1e4
                self.belts[layer['belt_name']][idx] = 1e4
                idx = self.belts[layer['belt_name']]<-1e4
                self.belts[layer['belt_name']][idx] = -1e4
                min_len = np.min([int(len(np.ravel(in_data))), len(np.ravel(self.belts[layer['belt_name']]))])
                    
                threshold = (layer['w_modulable'][0,0]+1) 
            if (layer['type']=='modulable_solid') or (layer['type']=='modulable'):
                inversed_threshold = 1/(np.abs(threshold)+0.01)
                if inversed_threshold>1e4:
                    inversed_threshold=1e4
                k = np.sin((0.1+self.belts[layer['belt_name']][0,:min_len])*inversed_threshold)
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data*k
            if 0:
                if np.sum(np.isnan(y))>0:
                    print('nan in nn ')
                    print('y',y)
                    print(f"layer {i} {layer['type']}" )
                    if layer['type']=='gru':
                        print('y_how_write',y_how_write)
                        print('y_what_write',y_what_write)
                        print("layer['cells']", layer['cells'])
                    print('x',x)
                    print('in_data',in_data)
                    print("layer['b']",layer['b'])
                    print("layer['w']",layer['w'])
                    1/0
                
            border = 1e4
            idx = (y>border)|(np.isinf(y))|(np.isnan(y))
            y[idx] = border
            idx = y<-border
            y[idx] = -border
            
            in_data = y
        y = np.ravel(y)
        return y
    def drop_state(self):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if layer['type']=='gru':
                self.layers[i]['cells'] *= 0
            elif layer['type']=='ggru_out':
                self.global_cells *= 0
        for k in self.belts.keys():
            if k== 'full_genom':
                self.belts[k] = self.disassemble_genom()#одна из лент может быть инициализирована геномом.
            else:
                self.belts[k] *= 0
            self.heads_m[k] = np.arange(0,len(self.heads_m[k])*10,10)
            self.heads_d[k] = np.zeros(len(self.heads_d[k]),dtype=np.float16) + 5
                
    def assemble_genom(self,genom):
        #геном перегнать в нейросеть
        genom = np.array(genom, dtype=np.float16)
        pointer = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if 'w' in layer.keys():
                delta = len(np.ravel(layer['w']))
                layer['w'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['w']))
                pointer += delta
            
                delta = len(np.ravel(layer['b']))
                layer['b'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['b']))
                pointer += delta
           
            if 'w_mem' in layer.keys():
                delta = len(np.ravel(layer['w_mem']))
                layer['w_mem'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['w_mem']))
                pointer += delta
            if 'w_modulable' in layer.keys():
                delta = len(np.ravel(layer['w_modulable']))
                layer['w_modulable'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['w_modulable']))
                pointer += delta
            
            self.layers[i] = layer
        for k in self.belts.keys():
            if k== 'full_genom':
                self.belts[k] = self.disassemble_genom()
                
    def disassemble_genom(self):
        #прочитать геном сети
        genom = []
        for layer in self.layers:
            if 'w' in layer.keys():
                genom.append(np.ravel(layer['w']))
                genom.append(np.ravel(layer['b']))
            if 'w_mem' in layer.keys():
                genom.append(np.ravel(layer['w_mem']))
            if 'w_modulable' in layer.keys():
                genom.append(np.ravel(layer['w_modulable']))
        genom = np.concatenate(genom,axis=0)
        return genom