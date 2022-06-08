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
PI = torch.acos(torch.zeros(1)).item() * 2
def make_gaussian(m,s,l):
    #никакая не гауссиана, зато быстро
    #и на выходе не полноценная маска, а sparce
    if m>=l:
        m = l
    if m<0:
        m = 0
        
    s = torch.abs(s)
    width = int(torch.ceil(s))
    dx = 1./s
    gauss_raw = torch.concat([torch.arange(0,1,dx),torch.arange(1,0,-dx)])
    out = torch.zeros(l)
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
    if torch.sum(gauss_raw[gauss_start:gauss_end])==0:
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
        self.mult_actions = torch.zeros(3)
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
                 and layer['type']!='connector_out'\
                 and (desc['type']!='ff_tolerance'):
                layer['w'] = torch.tensor(np.random.normal(size=[in_size_cur,w_shape_in])*scale_weights, dtype=torch.float32)
                layer['b'] = torch.tensor(np.random.normal(size=[1, in_size_cur])*scale_weights, dtype=torch.float32)
            elif (desc['type']=='modulable'):
                layer['w_modulable'] = torch.tensor(np.random.normal(size=[1, in_size_cur*3])*scale_weights, dtype=torch.float32)
            elif (desc['type']=='modulable_solid'):
                layer['w_modulable'] = torch.tensor(np.random.normal(size=[1, 3])*scale_weights, dtype=torch.float32)
            if desc['type']=='gru':
                layer['w_mem'] = torch.tensor(np.random.normal(size=[desc['out'],desc['cells']*2])*scale_weights, dtype=torch.float32)#одни контакты - это что писать. Другие - насколько сильно.
                layer['cells_sz'] = desc['cells']
                layer['cells'] = torch.zeros([1,desc['cells']], dtype=torch.float32)
            elif desc['type']=='ggru_out':
                layer['w_mem'] = torch.tensor(np.random.normal(size=[desc['out'],desc['cells']*2])*scale_weights, dtype=torch.float32)#одни контакты - это что писать. Другие - насколько сильно.
                self.global_cells_sz = desc['cells']
                self.global_cells = torch.zeros([1,desc['cells']], dtype=torch.float32)
            elif desc['type']=='turing_read':
                #есть лента (и название), есть определённое число heads
                #лента 1-мерная
                if not desc['name'] in self.belts.keys():
                    self.belts[desc['name']] = torch.zeros([1,desc['cells']], dtype=torch.float32)
                    self.heads_m[desc['name']] = torch.arange(0,desc['heads']*50,50)
                    self.heads_d[desc['name']] = torch.zeros(desc['heads'], dtype=torch.float32) + 5
                layer['belt_name'] = desc['name']
                #на выходе: копия входа, взвешенные суммы с каждой головы,позиции каждой головы (через синус)
            elif desc['type']=='turing_write':
                #на входе "что писать" и "как сильно писать"
                if not desc['name'] in self.belts.keys():
                    self.belts[desc['name']] = torch.zeros([1,desc['cells']], dtype=torch.float32)
                    self.heads_m[desc['name']] = torch.arange(0,desc['heads']*10,10)
                    self.heads_d[desc['name']] = torch.zeros(desc['heads'], dtype=torch.float32) + 5
                layer['belt_name'] = desc['name']
            elif desc['type']=='turing_move':
                #как подвинуть и как выставить фокусировку
                layer['belt_name'] = desc['name']
            elif desc['type']=='modulable':
                #модулируемый слой. Его элементы равны w + wm, где wm - это выход модулятора
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = torch.zeros([1,in_size_cur], dtype=torch.float32)
            elif desc['type']=='modulable_solid':
                #модулируемый слой. Его элементы равны w + wm, где wm - это выход модулятора
                #у него единые threshold и k_amplif на весь слой
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = torch.zeros([1,in_size_cur], dtype=torch.float32)
            elif desc['type']=='modulator' or desc['type']=='modulator_inertial':
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
                layer['w_tolerance'] = torch.tensor(np.array(np.random.normal(size=[1,2,in_size_cur])*scale_weights, dtype=np.float16),dtype=torch.float32)
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = layer['w_tolerance'][0,1]
            elif (desc['type']=='ff_tolerance'):
                layer['w_tolerance'] = torch.tensor(np.array(np.random.normal(size=[1,2,in_size_cur,w_shape_in])*scale_weights, dtype=np.float16),dtype=torch.float32)
                layer['b_tolerance'] = torch.tensor(np.array(np.random.normal(size=[1, in_size_cur])*scale_weights, dtype=np.float16),dtype=torch.float32)
                layer['belt_name'] = desc['name']
                self.belts[desc['name']] = layer['w_tolerance'][0,1]
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
        return torch.tensor([self.predict_sequence(x[i]) for i in range(x.shape[0])])
    
    def predict_sequence(self, x):
        #формат x: (seq_len,in_size)
        return torch.tensor([self.predict_sequence(x[i]) for i in range(x.shape[0])])
    
    def predict_vector(self, x):
        #формат x: (1,in_size)
        x = torch.tensor(x)
        shp = x.shape
        if shp[0]>1:
            x = x.reshape([1,*shp])
        shp = x.shape
        if shp[1]>3:
            try:
                x = x.reshape([shp[0],shp[3],shp[1],shp[2]])
            except:
                pass
        in_data = x
        if self.global_cells_sz>0:
            in_data = torch.hstack([in_data,self.global_cells])
            #self.global_cells
        for i in range(len(self.layers)):
            self.track_time('start')
            layer = self.layers[i]
            if layer['type']=='gru':
                in_data = torch.concat([in_data, layer['cells']],dim=1)
            if 'w' in layer:
                inp = torch.tensor(in_data,dtype=torch.float32)
                b = torch.tensor(layer['b'],dtype=torch.float32)
                w = torch.tensor(layer['w'],dtype=torch.float32)
                y = torch.matmul(inp + b,w)
                del b,w,inp
                
            else:
                y = in_data
            #нелинейность
            if layer['activation']=='relu':
                y = torch.clip(torch.tensor(y,dtype=torch.float32), min=0)
            elif layer['activation']=='lrelu':
                y = torch.tensor(y,dtype=torch.float32)
                y = torch.clip(y, min=y*0.001)
            elif layer['activation']=='logtan':
                y = torch.arctan(torch.tensor(y,dtype=torch.float32))/PI + 0.5
            elif layer['activation']=='linear':
                pass
            self.track_time('dot')
            #если gru, то запись в память
            if (layer['type']=='gru') or (layer['type']=='ggru_out'):
                self.track_time('gru_write')
                y_to_cell = torch.matmul(y,torch.tensor(layer['w_mem'],dtype=torch.float32))
                y_what_write = y_to_cell[:,:layer['cells_sz']]
                y_how_write = 0.5 + torch.arctan(y_to_cell[:,layer['cells_sz']:])/PI
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
                conv_list = torch.zeros([1, heads_cnt],dtype=torch.float32)
                m_list = torch.zeros([1, heads_cnt])
                s_list = torch.zeros([1, heads_cnt])
                sin_positions = []
                for i in range(heads_cnt):
                    m = self.heads_m[layer['belt_name']][i]
                    s = self.heads_d[layer['belt_name']][i]
                    mask,nonzeros = make_gaussian(m,s,self.belts[layer['belt_name']].shape[1])
                    conv = torch.sum(self.belts[layer['belt_name']][0,nonzeros[0]:nonzeros[1]]*mask)
                    #print("self.belts[layer['belt_name']][0,nonzeros]",self.belts[layer['belt_name']][0,nonzeros])
                    #print('mask',mask)
                    conv_list[0,i] = conv
                    m_list[0,i] = m
                    s_list[0,i] = s
                    
                    sin_positions_add = [torch.sin(m/self.sin_t_for_attn*(2**j)) for j in range(self.sin_count_for_attn)]
                    sin_positions.extend(sin_positions_add)
                sin_positions = torch.tensor(sin_positions)
                sin_positions = sin_positions.reshape([1,len(sin_positions)])
                #присобачить выход голов к y от прошлого слоя. И ещё приделать их позиции.
                self.track_time('turing_read ended')
                #print('conv_list',conv_list)
                y = torch.hstack([y,conv_list,m_list,s_list,sin_positions])
                
            elif layer['type']=='turing_move':
                self.track_time('turing_move')
                #двигать головы############################################################################
                for i in range(len(self.heads_m[layer['belt_name']])):
                    maxmove = 10000
                    #амплитуда большая, но распределение такое: 
                    #f(0.05)=1.25,f(0.1)=10, f(0.2)=80, f(0.5)=1250, f(0.8)=5120
                    self.heads_m[layer['belt_name']][i] += (((torch.arctan(in_data[0,2*i])/PI)*2)**3)*maxmove
                    maxd = 30
                    self.heads_d[layer['belt_name']][i] = (torch.arctan(in_data[0,2*i+1])/PI + 0.5)*maxd
                y = in_data
                self.track_time('turing_move ended')
            elif layer['type']=='turing_write':
                self.track_time('turing_write')
                ############################################################################
                conv_list = torch.zeros([1, len(self.heads_m)])
                head_cnt = len(self.heads_m[layer['belt_name']])
                self.track_time('turing_write st')
                for i in range(head_cnt):
                    self.track_time('turing_write a')
                    m = self.heads_m[layer['belt_name']][i]
                    s = self.heads_d[layer['belt_name']][i]
                    
                    self.track_time('turing_write b')
                    mask,nonzeros = make_gaussian(m,s,self.belts[layer['belt_name']].shape[1])
                    self.track_time('turing_write c')
                    y = torch.tensor(y,dtype=torch.float32)
                    mask = torch.tensor(mask,dtype=torch.float32)
                    mask = mask*(torch.arctan(y[0,2*i])/PI + 0.5)#с какой силой писать
                    self.track_time(f'turing_write d mask: {len(mask)}')
                    value_wr = y[0,2*i+1]
                    value_wr = torch.clip(value_wr,-1e5,1e5)
                    #print('nonzeros',nonzeros,'value_wr',value_wr,'mask',mask)
                    b = self.belts[layer['belt_name']][0,nonzeros[0]:nonzeros[1]]*(1-mask) + mask*value_wr
                    b = torch.tensor(torch.clip(b,-1e6,1e6),dtype=torch.float32)
                    self.belts[layer['belt_name']][:,nonzeros[0]:nonzeros[1]] = b
                    delta = b-self.belts[layer['belt_name']][0][nonzeros[0]:nonzeros[1]]                    
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
                min_len = len(torch.ravel(self.belts[layer['belt_name']]))
                self.belts[layer['belt_name']][:,:min_len] = in_data[:,:min_len]
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data
            elif layer['type']=='modulator_inertial':
                min_len = len(torch.ravel(self.belts[layer['belt_name']]))
                self.belts[layer['belt_name']][0,:min_len] = self.belts[layer['belt_name']][0,:min_len] + in_data[0,:min_len]
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data
            elif layer['type']=='modulable':
                min_len = len(torch.ravel(self.belts[layer['belt_name']]))
                k_amplif = 5*layer['w_modulable'][0,0:min_len]
                k_start = layer['w_modulable'][0,2*min_len:3*min_len]
                
                threshold = 1e3*(layer['w_modulable'][0,min_len:2*min_len]+1)
            elif layer['type']=='modulable_solid':
                min_len = len(torch.ravel(self.belts[layer['belt_name']]))
                k_amplif = 5*layer['w_modulable'][0,0]
                k_start = layer['w_modulable'][0,2]
                threshold = (layer['w_modulable'][0,1]+1) 
            elif (layer['type']=='modulable_solid') or (layer['type']=='modulable'):
                inversed_threshold = 1/(threshold+0.01)
                k = torch.sin((k_start+self.belts[layer['belt_name']][0,:min_len])*inversed_threshold)*k_amplif
                #первая половина связей идёт в модуляцию
                #плюс все связи пробрасываются вперёд
                y = in_data*k
            elif layer['type']=='conv':
                shp = in_data.shape
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
                self.belts[layer['belt_name']] = in_data
            elif layer['type']=='connector_out':
                y = torch.concat((in_data, self.belts[layer['belt_name']]))
            elif layer['type']=='tolerance':
                self.belts[layer['belt_name']][:,:] = torch.tensor(self.belts[layer['belt_name']][:,:])-in_data[:,:]/layer['w_tolerance'][0,0]
                y=torch.tensor(self.belts[layer['belt_name']][:,:])*in_data
            elif layer['type']=='ff_tolerance':
                w = (torch.tensor(in_data)*torch.ones((layer['w_tolerance'][0,0].shape[-1],1)))
                self.belts[layer['belt_name']] = torch.tensor(self.belts[layer['belt_name']])-w.reshape((w.shape[-1],(w.shape[0] if w.shape[0]>1 else w.shape[1])))/layer['w_tolerance'][0,0]
                y=torch.matmul(torch.tensor(in_data)+layer['b_tolerance'],torch.tensor(self.belts[layer['belt_name']]))
            if 0:
                if torch.sum(torch.isnan(y))>0:
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
            in_data = y
        y = torch.ravel(y)
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
            self.heads_m[k] = torch.arange(0,len(self.heads_m[k])*10,10)
            self.heads_d[k] = torch.zeros(len(self.heads_d[k]),dtype=torch.float32) + 5
                
    def assemble_genom(self,genom):
        #геном перегнать в нейросеть
        genom = torch.tensor(genom)
        pointer = 0
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if 'w' in layer.keys():
                delta = len(torch.ravel(layer['w']))
                layer['w'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['w']).shape)
                pointer += delta
            
                delta = len(torch.ravel(layer['b']))
                layer['b'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['b']).shape)
                pointer += delta
           
            if 'w_mem' in layer.keys():
                delta = len(torch.ravel(layer['w_mem']))
                layer['w_mem'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['w_mem']).shape)
                pointer += delta
            if 'w_modulable' in layer.keys():
                delta = len(torch.ravel(layer['w_modulable']))
                layer['w_modulable'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['w_modulable']).shape)
                pointer += delta
            if 'w_tolerance' in layer.keys():
                delta = len(torch.ravel(layer['w_tolerance']))
                layer['w_tolerance'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['w_tolerance']).shape)
                self.belts[layer['belt_name']] = layer['w_tolerance'][:,1]
                pointer += delta
            if 'b_tolerance' in layer.keys():
                delta = len(torch.ravel(layer['b_tolerance']))
                layer['b_tolerance'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['b_tolerance']).shape)
                pointer += delta
            if 'kernel' in layer.keys():
                delta = len(torch.ravel(layer['kernel']))
                layer['kernel'] = torch.reshape(genom[pointer:pointer+delta],shape=(layer['kernel']).shape)
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
                genom.append(torch.ravel(layer['w']))
                genom.append(torch.ravel(layer['b']))
            if 'w_mem' in layer.keys():
                genom.append(torch.ravel(layer['w_mem']))
            if 'w_modulable' in layer.keys():
                genom.append(torch.ravel(layer['w_modulable']))
            if 'w_tolerance' in layer.keys():
                genom.append(torch.ravel(layer['w_tolerance']))
            if 'b_tolerance' in layer.keys():
                genom.append(torch.ravel(layer['b_tolerance']))
            if 'kernel' in layer.keys():
                genom.append(torch.ravel(layer['kernel']))
        genom = torch.concat(genom,axis=0)
        return genom