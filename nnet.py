import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm
import cv2
import torch
import torch.nn as nn
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
            layer['desc'] = desc
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
            elif layer['type']!='conv' and layer['type']!='max_pool':
                w_shape_in = desc['out']
            else:
                w_shape_in = 1
            if (desc['type']!='modulator_inertial')\
                 and (desc['type']!='modulator')\
                 and (desc['type']!='modulable')\
                 and (desc['type']!='modulable_solid')\
                 and (desc['type']!='conv')\
                 and (desc['type']!='flatten')\
                 and layer['type']!='max_pool'\
                 and layer['type']!='connector_in'\
                 and layer['type']!='connector_out':
                layer['w'] = np.array(np.random.normal(size=[in_size_cur,w_shape_in])*scale_weights, dtype=np.float16)
                layer['b'] = np.array(np.random.normal(size=[1, in_size_cur])*scale_weights, dtype=np.float16)
            elif (desc['type']=='modulable'):
                layer['w_modulable'] = np.array(np.random.normal(size=[1,in_size_cur*3])*scale_weights, dtype=np.float16)
            elif (desc['type']=='modulable_solid'):
                layer['w_modulable'] = np.array(np.random.normal(size=[1,3])*scale_weights, dtype=np.float16)
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
            elif desc['type']=='conv':
                layer['kernel']=torch.tensor(np.array(np.random.normal(size=desc['filter_size'])*scale_weights, dtype=np.float32),dtype=torch.float32)
                layer['stride']=desc['stride']
                layer['padding']=desc['padding']
                layer['filter_size']=desc['filter_size']
                #in_channels, out_channels, (kernel_size)
                layer['layer']=nn.Conv2d(layer['kernel'].shape[1], layer['kernel'].shape[0], (layer['kernel'].shape[2],layer['kernel'].shape[3]), layer['stride'], layer['padding'],bias=False)
            elif desc['type']=='max_pool':
                layer['pool_size']=desc['pool_size']
                layer['stride']=desc['stride']
            elif desc['type']=='connector_in' or desc['type']=='connector_out':
                layer['belt_name'] = desc['name']
            elif (desc['type']=='tolerance'):
                layer['w_tolerance'] = torch.tensor(np.array(np.random.normal(size=[1,in_size_cur*2])*scale_weights, dtype=np.float16),dtype=torch.float32)
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = layer['w_tolerance'][0,in_size_cur:]
            if not ('activation' in desc.keys()):
                desc['activation'] = 'relu'#max(0,x)
            layer['activation'] = desc['activation']
            
            self.layers.append(layer)
            if layer['type']!='conv' and layer['type']!='max_pool': in_size_cur = desc['out']
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
            x = np.reshape(x,[1,*shp])
        shp = np.shape(x)
        if shp[1]>3:
            try:
                x = np.reshape(x,[shp[0],shp[3],shp[1],shp[2]])
            except:
                pass
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
                inp = torch.tensor(in_data,dtype=torch.float32)
                b = torch.tensor(layer['b'],dtype=torch.float32)
                w = torch.tensor(layer['w'],dtype=torch.float32)
                y = torch.matmul(inp + b,w)
                del b,w,inp
                
            else:
                y = in_data
            y = np.clip(y, a_min=-1e9,a_max=1e9)#убрать бесконечности
            #нелинейность
            if layer['activation']=='relu':
                y = torch.clip(torch.tensor(y,dtype=torch.float32), min=0)
            elif layer['activation']=='lrelu':
                y = torch.tensor(y,dtype=torch.float32)
                y = torch.clip(y, min=y*0.001)
            elif layer['activation']=='logtan':
                y = torch.arctan(torch.tensor(y,dtype=torch.float32))/np.pi + 0.5
            elif layer['activation']=='linear':
                pass
            self.track_time('dot')
            #если gru, то запись в память
            if (layer['type']=='gru') or (layer['type']=='ggru_out'):
                self.track_time('gru_write')
                y_to_cell = torch.matmul(y,torch.tensor(layer['w_mem'],dtype=torch.float32))
                y_what_write = y_to_cell[:,:layer['cells_sz']]
                y_how_write = 0.5 + torch.arctan(y_to_cell[:,layer['cells_sz']:])/np.pi
                layer['cells'] = torch.tensor(layer['cells'],dtype=torch.float32)*(1-y_how_write) + y_what_write*y_how_write
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
                    y = torch.tensor(y,dtype=torch.float32)
                    mask = torch.tensor(mask,dtype=torch.float32)
                    mask = mask*(torch.arctan(y[0,2*i])/np.pi + 0.5)#с какой силой писать
                    mask = mask.detach().numpy()
                    y = y.detach().numpy()
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
                in_data = in_data.detach().numpy()
                self.belts[layer['belt_name']][0,:min_len] += in_data[0,:min_len]
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data
            elif layer['type']=='modulable':
                min_len = np.min([int(len(np.ravel(in_data))), len(np.ravel(self.belts[layer['belt_name']]))])
                k_amplif = 5*layer['w_modulable'][0,0:min_len]
                k_start = layer['w_modulable'][0,2*min_len:3*min_len]
                
                threshold = 1e3*(layer['w_modulable'][0,min_len:2*min_len]+1)
            elif layer['type']=='modulable_solid':
                idx = np.isinf(self.belts[layer['belt_name']])
                min_len = np.min([int(len(np.ravel(in_data))), len(np.ravel(self.belts[layer['belt_name']]))])
                k_amplif = 5*layer['w_modulable'][0,0]
                k_start = layer['w_modulable'][0,2]
                threshold = (layer['w_modulable'][0,1]+1) 
            elif (layer['type']=='modulable_solid') or (layer['type']=='modulable'):
                inversed_threshold = 1/(np.abs(threshold)+0.01)
                k = np.sin((k_start+self.belts[layer['belt_name']][0,:min_len])*inversed_threshold)*k_amplif
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data*k
            elif layer['type']=='conv':
                shp = np.shape(in_data)
                inp = torch.tensor(in_data,dtype=torch.float32)
                layer['layer'].load_state_dict({'weight': torch.tensor(layer['kernel'])}, strict=False)
                y = layer['layer'](inp)
                del inp
            elif layer['type']=='flatten':
                size_out = 1
                for i in in_data.shape: size_out = size_out*i
                y=in_data.reshape((1,size_out))
            elif layer['type']=='max_pool':
                inp = torch.tensor(in_data,dtype=torch.float32)
                y=nn.MaxPool2d(layer['pool_size'], stride=layer['stride'])(inp)
                del inp
            elif layer['type']=='connector_in':
                min_len = np.min([int(len(np.ravel(in_data))/2), len(np.ravel(self.belts[layer['belt_name']]))])
                self.belts[layer['belt_name']][:,:min_len] = in_data[:,:min_len]
                y = in_data
            elif layer['type']=='connector_out':
                min_len = np.min([int(len(np.ravel(in_data))/2), len(np.ravel(self.belts[layer['belt_name']]))])
                y = torch.concat((in_data, self.belts[layer['belt_name']][:,:min_len]))
            elif layer['type']=='tolerance':
                min_len = np.min([int(len(np.ravel(in_data))/2), len(np.ravel(self.belts[layer['belt_name']]))])
                self.belts[layer['belt_name']][:,:min_len] = torch.tensor(self.belts[layer['belt_name']][:,:min_len])-in_data[:,:min_len]/layer['w_tolerance'][0,0:min_len]
                y=torch.tensor(self.belts[layer['belt_name']][:,:])*in_data
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
            try:
                idx = (y>border)|(np.isinf(y))|(np.isnan(y))
                y[idx] = border
                idx = y<-border
                y[idx] = -border
            except:
                y = y.detach().numpy()
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
            self.heads_d[k] = np.zeros(len(self.heads_d[k]),dtype=np.float32) + 5
                
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
            if 'w_tolerance' in layer.keys():
                delta = len(np.ravel(layer['w_tolerance']))
                layer['w_tolerance'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['w_tolerance']))
                self.belts[layer['belt_name']] = layer['w_tolerance'][:,layer['w_tolerance'].shape[1]//2:]
                pointer += delta
            if 'kernel' in layer.keys():
                delta = len(np.ravel(layer['kernel']))
                layer['kernel'] = np.reshape(genom[pointer:pointer+delta],newshape=np.shape(layer['kernel']))
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
            if 'w_tolerance' in layer.keys():
                genom.append(np.ravel(layer['w_tolerance']))
            if 'kernel' in layer.keys():
                genom.append(np.ravel(layer['kernel']))
        genom = np.concatenate(genom,axis=0)
        return genom
    def conv_(self,img, conv_filter):
        filter_size = conv_filter.shape[1]
        result = np.zeros((img.shape))
        for r in np.arange(filter_size/2.0, img.shape[0]-filter_size/2.0+1):
            for c in np.arange(filter_size/2.0, img.shape[1]-filter_size/2.0+1):
                curr_region = img[int(r-np.floor(filter_size/2.0)):int(r+filter_size/2.0), int(c-np.floor(filter_size/2.0)):int(c+filter_size/2.0)]
                curr_result = curr_region * conv_filter
                conv_sum = np.sum(curr_result)
                result[int(r), int(c)] = conv_sum
        final_result = result[int(filter_size/2.0):int(result.shape[0]-filter_size/2.0), int(filter_size/2.0):int(result.shape[1]-filter_size/2.0)]
        return final_result

    def conv(self,img, conv_filter):
        if len(img.shape) > 2 or len(conv_filter.shape) > 3:
            if img.shape[-1] != conv_filter.shape[-1]:
                print("Error: Number of channels in both image and filter must match.")
                sys.exit()
        if conv_filter.shape[1] != conv_filter.shape[2]:
            print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
            sys.exit()
        if conv_filter.shape[1]%2==0:
            print('Error: Filter must have an odd size. I.e. number of rows and columns must be odd.')
            sys.exit()
        feature_maps = np.zeros((img.shape[0]-conv_filter.shape[1], 
                                    img.shape[1]-conv_filter.shape[1], 
                                    conv_filter.shape[0]))
        for filter_num in range(conv_filter.shape[0]):
            #print("Filter ", filter_num + 1)
            curr_filter = conv_filter[filter_num, :]
            if len(curr_filter.shape) > 2:
                conv_map = self.conv_(img[:, :, 0], curr_filter[:, :, 0])
                for ch_num in range(1, curr_filter.shape[-1]):
                    conv_map = conv_map + self.conv_(img[:, :, ch_num], 
                                    curr_filter[:, :, ch_num])
            else:
                conv_map = self.conv_(img, curr_filter)
            feature_maps[:, :, filter_num] = conv_map
        return feature_maps
    def max_pool(self,feature_map,size,stride):
        pool_out = np.zeros((int((feature_map.shape[0]-size+1)/stride), int((feature_map.shape[1]-size+1)/stride), feature_map.shape[-1]))
        for map_num in range(feature_map.shape[-1]):
            ri = 0
            for r in np.arange(0,feature_map.shape[0]-size-1, stride):
                ci = 0
                for c in np.arange(0, feature_map.shape[1]-size-1, stride):
                    pool_out[ri, ci, map_num] = np.max([feature_map[r:r+size,  c:c+size, map_num]])
                    ci = ci + 1
                ri = ri +1
        return pool_out