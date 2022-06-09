import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from collections import deque
import copy
import pickle
import torch

#сделаем оптимизатор а-ля Абатур. Смесь из эволюции и градиентного спуска. Но сделаем его проще... И пусть эволюция может сохранять свои данные.
#И всё это дело обвязано "многоруким бандитом"
def torch_nanstd(o, dim=None, keepdim=False):
    mask = torch.isnan(o)
    if mask.any():
        output = torch.from_numpy(np.asarray(np.nanstd(o.cpu().numpy(), axis=dim, keepdims=keepdim))).to(o.device)
        if output.shape == mask.shape:
            output[mask] = 1
        return output
    else:
        return torch.std(o, dim=dim, keepdim=keepdim) if dim is not None else torch.std(o)
class optimizer():
    def __init__(self, function,genom_size,parallel_cores=1,init_file='./genom.pkl',history_file='./history.pkl'):
        self.history_gain = {}
        self.history_time = {}
        self.optimizer_list = ['evol_mid_chaos','gradient_wide_50','rel_coord_default','evol_soft','gradient_long_adaptive_inertial','gradient_slow_20','gradient_long_adaptive','evol_wide','evol_narrow']
        #self.optimizer_list = ['evol_infinite']
        #self.optimizer_list = ['gradient_wide']
        for opt_name in self.optimizer_list:
            self.history_gain[opt_name] = deque(maxlen=10)
            self.history_gain[opt_name].append(torch.nan)
            self.history_time[opt_name] = deque(maxlen=10)
            self.history_time[opt_name].append(torch.nan)
        self.current_loss = torch.tensor(torch.nan,dtype=torch.float32)
        self.parallel_cores = parallel_cores
        self.function = function#что оптимизировать
        self.best_genoms = torch.tensor(deque(maxlen=10))
        bounds = [-0.1,0.1]
        self.best_genoms.append(torch.tensor(np.random.random(size=genom_size)*(bounds[1]-bounds[0]) + bounds[0],dtype=torch.float32))
        self.genom_size = genom_size
        
        self.init_file = init_file
        try:
            with open(self.init_file , 'rb') as f:
                self.best_genoms.append(torch.tensor(pickle.load(f),dtype=torch.float32))
            print('loaded successfully')
        except Exception:
            pass
        self.history_file = history_file
        try:
            with open(self.history_file , 'rb') as f:
                self.history_gain,self.history_time = pickle.load(f)
            print('history loaded successfully')
        except Exception:
            pass
    def optimize(self):
        mx = []
        time_penalty = 0.0005
        for opt_name in self.optimizer_list:
            #print(f'self.history_gain[{opt_name}]',self.history_gain[opt_name])
            mx.append(torch.nanmean(torch.tensor(self.history_gain[opt_name],dtype=torch.float32)-time_penalty*torch.nanmean(torch.tensor(self.history_time[opt_name]))))
        std = torch_nanstd(torch.tensor(mx,dtype=torch.float32))*0.5+0.000001
        #std = 0
        mx = torch.tensor(mx,dtype=torch.float32)
        mx[torch.isnan(mx)] = 1e10#невероятно хороший результат
        if np.random.rand()<0.15:
            print('random trial')
            k_noise = 3
        else:
            k_noise = 0
        mx_aug = mx + torch.tensor(np.random.rand(len(mx))*k_noise,dtype=torch.float32)
        print('scores for optimizers augmented',mx_aug)
        amax = torch.argmax(mx_aug)
        chosen_optimizer = self.optimizer_list[amax]
        print('chosen',chosen_optimizer,'previous_result:',self.history_gain[opt_name][-1],'per tacts:',self.history_time[opt_name][-1])
        t = pd.Timestamp.now()
        if chosen_optimizer=='evol_infinite':
            self.evol_infinite()
        if chosen_optimizer=='evol_wide':
            self.evol_wide()
        elif chosen_optimizer=='evol_narrow':
            self.evol_narrow()
        elif chosen_optimizer=='evol_mid_chaos':
            self.evol_mid_chaos()
        elif chosen_optimizer=='evol_soft':
            self.evol_soft()
        elif chosen_optimizer=='gradient_wide':
            self.gradient_wide()
        elif chosen_optimizer=='gradient_wide_50':
            self.gradient_wide_50()
        elif chosen_optimizer=='gradient_slow_20':
            self.gradient_slow_20()
        elif chosen_optimizer=='gradient_long_adaptive':
            self.gradient_long_adaptive()
        elif chosen_optimizer=='gradient_long_adaptive_inertial':
            self.gradient_long_adaptive_inertial()
        elif chosen_optimizer=='rel_coord_default':
            self.rel_coord_default()
            
        print('result',chosen_optimizer,'previous_gain:',self.history_gain[opt_name][-1],'per tacts:',self.history_time[opt_name][-1],'duration',pd.Timestamp.now()-t)
        try:
            self.current_loss=self.function(self.best_genoms[-1], verbose=True)
        except Exception:
            pass
        #print('checking efficiency at all environments, including unknown (validation)')
        try:
            self.function(self.best_genoms[-1], verbose=True, test_set=True)
        except Exception:
            pass
        with open(self.history_file , 'wb') as f:
            pickle.dump((self.history_gain,self.history_time),f,protocol=pickle.HIGHEST_PROTOCOL)
    
    def evol_infinite(self):
        opt_name = 'evol_infinite'
        popsize=40
        maxiter=int(1e9)
        [genom_best,genoms, losses] = self.evol_parallel(self.function,bounds=[-1,1],size_x=self.genom_size, popsize=popsize,maxiter=maxiter, mutation_p=0.09,mutation_p_e=0.01,
                  mutation_r=0.2, alpha_count=9,elitarism=4,verbose=True,mutation_amplitude_source='std',
                  out=[],
                  start_point=self.best_genoms,get_extended=True,)
        gain = torch.max(losses)-self.current_loss#было -2, стало -1. gain = 1. Положительный gain - хорошо
        self.current_loss = torch.max(losses)
        time_left = popsize*(maxiter+1)
        self.best_genoms.extend(genoms)
        self.best_genoms.append(genom_best)
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(gain)
        self.history_time[opt_name].append(time_left)
        
    def evol_wide(self):
        opt_name = 'evol_wide'
        popsize=40
        maxiter=4
        [genom_best,genoms, losses] = self.evol_parallel(self.function,bounds=[-1,1],size_x=self.genom_size, popsize=popsize,maxiter=maxiter, mutation_p=0.03,mutation_p_e=0.01,
                  mutation_r=0.2, alpha_count=6,elitarism=4,verbose=True,mutation_amplitude_source='std',
                  out=[],
                  start_point=self.best_genoms,get_extended=True,)
        gain = torch.max(losses)-self.current_loss#было -2, стало -1. gain = 1. Положительный gain - хорошо
        self.current_loss = torch.max(losses)
        time_left = popsize*(maxiter+1)
        self.best_genoms.extend(genoms)
        self.best_genoms.append(genom_best)
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(gain)
        self.history_time[opt_name].append(time_left)
    def evol_narrow(self):
        opt_name = 'evol_narrow'
        popsize=8
        maxiter=16
        [genom_best,genoms, losses] = self.evol_parallel(self.function,bounds=[-1,1],size_x=self.genom_size, popsize=popsize,maxiter=maxiter, mutation_p=0.01,mutation_p_e=0.3,
                  mutation_r=0.1, alpha_count=3,elitarism=2,mutation_amplitude_source='std',verbose=True,
                  out=[],
                  start_point=self.best_genoms,get_extended=True)
        gain = torch.max(losses)-self.current_loss#Положительный gain - хорошо
        self.current_loss = torch.max(losses)
        time_left = popsize*(maxiter+1)
        self.best_genoms.extend(genoms)
        self.best_genoms.append(genom_best)
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(gain)
        self.history_time[opt_name].append(time_left)
    def evol_soft(self):
        opt_name = 'evol_soft'
        popsize=16
        maxiter=5
        [genom_best,genoms, losses] = self.evol_parallel(self.function,bounds=[-1,1],size_x=self.genom_size, popsize=popsize,maxiter=maxiter, mutation_p=1,mutation_p_e=0.0,
                  mutation_r=0.004, alpha_count=5,elitarism=3,mutation_amplitude_source='std',verbose=True,
                  out=[],
                  start_point=self.best_genoms,get_extended=True)
        gain = torch.max(losses)-self.current_loss#Положительный gain - хорошо
        self.current_loss = torch.max(losses)
        time_left = popsize*(maxiter+1)
        self.best_genoms.extend(genoms)
        self.best_genoms.append(genom_best)
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        #print('before',self.history_gain[opt_name])
        self.history_gain[opt_name].append(gain)
        #print('gain',gain,'self.current_loss',self.current_loss)
        #print('after',self.history_gain[opt_name])
        self.history_time[opt_name].append(time_left)
    def evol_mid_chaos(self):
        opt_name = 'evol_mid_chaos'
        popsize=30
        maxiter=2
        [genom_best,genoms, losses] = self.evol_parallel(self.function,bounds=[-1,1],size_x=self.genom_size, popsize=popsize,maxiter=maxiter, mutation_p=0.05,mutation_p_e=0.1,
                  mutation_r=1.2, alpha_count=3,elitarism=3,mutation_amplitude_source='std',verbose=True,
                  out=[],
                  start_point=self.best_genoms,get_extended=True)
        gain = torch.max(losses)-self.current_loss#Положительный gain - хорошо
        self.current_loss = torch.max(losses)
        time_left = popsize*(maxiter+1)
        self.best_genoms.extend(genoms)
        self.best_genoms.append(genom_best)
        if not(opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(gain)
        self.history_time[opt_name].append(time_left)
    def gradient(self,width,stripe,maxiter,step,opt_name,adapt=1.0,chance_retry=0,momentum_usage_coef=0,momentum_eta=0.5):
        n_jobs = self.parallel_cores
        genom_cur = torch.tensor(self.best_genoms[-1],dtype=torch.float32)
        genom_prev = torch.tensor(genom_cur,dtype=torch.float32)
        score_prev = self.function(genom_prev)
        y_start = score_prev
        retry = False
        
        momentum = None
        for j in range(maxiter):
            if not retry:
                genoms = [genom_cur]
                idx_lst = []
                for i in range(width):
                    idx = int(np.random.rand()*(len(self.best_genoms[-1])-1-stripe))
                    idx_lst.append(idx)
                    genom_local = torch.tensor(genom_cur,dtype=torch.float32)
                    if np.random.rand()<0.5:
                        genom_local[idx:idx+stripe] += step
                    else:
                        genom_local[idx:idx+stripe] -= step
                    genoms.append(genom_local)
                if n_jobs!=1:
                    pool = Pool(processes=n_jobs)
                    y_lst = pool.map(self.function, [x for x in genoms])
                    pool.close()
                    pool.join()
                else:
                    y_lst = list(map(self.function, [x for x in genoms]))
                y_deltas = torch.tensor(y_lst[1:],dtype=torch.float32) - y_lst[0]
                if torch.max(y_deltas)>0:
                    idx = idx_lst[torch.argmax(y_deltas)]
                    genom_reserve = torch.tensor(genom_cur,dtype=torch.float32)
                    genom_reserve[idx:idx+stripe] += step
                grad = y_deltas/torch.sum(torch.abs(y_deltas)+0.000001)
                    
            #ищем МАКСИМУМ
            for i in range(width):
                genom_cur[idx_lst[i:i+stripe]] = torch.tensor(genom_cur[idx_lst[i:i+stripe]]) + torch.tensor(grad[i]*step)
            if momentum is not None:
                genom_cur += momentum*momentum_usage_coef
                
            score_new = self.function(genom_cur)
            print('score_new',score_new,'score_prev',score_prev,'gained',score_new-score_prev)
            #попробовать сделать шаг не по градиенту, а по одной из производных
            if score_prev>=score_new:
                if torch.max(y_deltas)>0:
                    amax = torch.argmax(y_deltas)
                    genom_cur = genom_reserve
                    score_new = self.function(genom_cur)
                    print('but success with one derivative: score_new',score_new,'score_prev',score_prev,'gained',score_new-score_prev)
                
            if score_prev>=score_new:
                print('undo')
                genom_cur=torch.tensor(genom_prev,dtype=torch.float32)
                step /=adapt
                retry = False
            else:
                if momentum is None:
                    momentum = genom_cur - genom_prev
                else:
                    momentum += genom_cur - genom_prev
                momentum *= momentum_eta
                score_prev = score_new
                genom_prev = torch.tensor(genom_cur,dtype=torch.float32)
                step *=adapt
                if np.random.rand()<chance_retry:
                    retry = True
        y_end = score_prev
        self.best_genoms.append(genom_cur)
        
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(y_end-y_start)#y растёт, gain растёт
        self.history_time[opt_name].append((width+1)*maxiter)
        self.current_loss=y_end
    
    def gradient_wide(self):
        opt_name = 'gradient_wide'
        width = 25
        stripe = 1
        maxiter = 4
        step = 0.07
        adapt=1
        self.gradient(width,stripe,maxiter,step,opt_name,adapt,momentum_usage_coef=0.2)
    def gradient_long_adaptive(self):
        opt_name = 'gradient_long_adaptive'
        width = 9
        stripe = 2
        maxiter = 15
        step = 0.01
        adapt=1.5
        self.gradient(width,stripe,maxiter,step,opt_name,adapt)
        
    def gradient_long_adaptive_inertial(self):
        opt_name = 'gradient_long_adaptive_inertial'
        width = 3
        stripe = 2
        maxiter = 20
        step = 0.05
        adapt=1.5
        chance_retry = 0.95
        self.gradient(width,stripe,maxiter,step,opt_name,adapt,chance_retry)
        
        
        

    def gradient_wide_50(self):
        opt_name = 'gradient_wide_50'
        width = 8
        stripe = 50
        maxiter = 5
        step = 0.005
        adapt=1
        self.gradient(width,stripe,maxiter,step,opt_name,adapt,momentum_usage_coef=0.2)
        
    def gradient_slow_20(self):      
        opt_name = 'gradient_slow_20'
        width = 20
        stripe = 3
        maxiter = 4
        step = 0.0001
        adapt=1
        self.gradient(width,stripe,maxiter,step,opt_name,adapt)
        
    def rel_coord_default(self):      
        opt_name = 'rel_coord_default'
        stripe = 10
        maxiter = 25
        step = 0.05
        adapt=2
        chance_retry = 0.95
        self.relative_coordinate_descent(stripe,maxiter,step,opt_name,adapt,chance_retry)
        
        
    def relative_coordinate_descent(self,stripe,maxiter,step,opt_name,adapt=1.0,chance_retry=0):
        #то же, что и градиентный спуск, но берётся одна координата/страйп и умножается на 1.001, скажем.
        n_jobs = self.parallel_cores
        width = 1
        genom_cur = self.best_genoms[-1]
        genom_prev = torch.tensor(genom_cur,dtype=torch.float32)
        score_prev = self.function(genom_prev)
        y_start = score_prev
        retry = False
        for j in range(maxiter):
            if not retry:

                idx = int(np.random.rand()*(len(self.best_genoms[-1])-1-stripe))
                genom_local = torch.tensor(genom_cur,dtype=torch.float32)
                if np.random.rand()<0.5:
                    step *= -1
                genom_local[idx:idx+stripe] *= 1+step

            score_new = self.function(genom_local)
            #y_deltas = y_lst[1] - y_lst[0]
            if score_new<score_prev:
                genom_local = torch.tensor(genom_cur,dtype=torch.float32)
                step *= -1
                genom_local[idx:idx+stripe] *= 1+step
                score_new = self.function(genom_local)
            #ищем МАКСИМУМ
            genom_cur = genom_local
                
            print('score_new',score_new,'score_prev',score_prev,'gained',score_new-score_prev)
            if score_prev>=score_new:
                print('undo')
                genom_cur=torch.tensor(genom_prev,dtype=torch.float32)
                step /=adapt
                retry = False
            else:
                score_prev = score_new
                genom_prev = torch.tensor(genom_cur,dtype=torch.float32)
                step *=adapt
                if np.random.rand()<chance_retry:
                    retry = True
        y_end = score_prev
        self.best_genoms.append(genom_cur)
        
        if not (opt_name in self.history_gain.keys()):
                self.history_gain[opt_name] = []
                self.history_time[opt_name] = []
        self.history_gain[opt_name].append(y_end-y_start)#y растёт, gain растёт
        self.history_time[opt_name].append((width+1)*maxiter)
        self.current_loss=y_end
        
            
    def evol_parallel(self,function,bounds=[-1,1],size_x=10,popsize=20,maxiter=10,mutation_p=0.1,mutation_p_e=0.01,
                  mutation_r=0.1,alpha_count=3,elitarism=2,mutation_amplitude_source='rel',seed=-1,verbose=True,
                  out=[],
                  start_point=[],get_extended=False):
        #function - функция, которую надо оптимизировать
		#bounds - из какого распределения взяты исходные геномы
		#size_x - размер генома
		#popsize - размер популяции
		#maxiter - число итераций
		#mutation_p - вероятность мутации
		#mutation_p_e - скорость снижения вероятности мутации
		#mutation_r - средняя амплитуда мутации
		#alpha_count - число альфачей, генокодов, которые будут скрещиваться
		#elitarism - число элит, генокодов, которые доживут до следующего поколения в неизменном виде
		#seed - случайное зерно
		#out - если сюда зарядить указатель на что-либо, то туда будет писаться отладочный вывод. Нужно, если мы захотим закрашить эволюцию на середине
		#start_point - можно явно задать список стартовых генокодов
		#get_extended - если True, то выводить не только наилучший генокод, а всё поколение
        #mutation_amplitude_source = 'rel' / 'abs'/ 'std'. 'rel' - значит, амплитуда 0.1 означает 0.1 от значения данного гена. 'abs' - 0.1 берётся просто. 'std' - значит, берётся 0.1 от среднего разброса по популяции
        start_point = torch.tensor(list(np.array(i) for i in start_point))
        if seed<0:
            seed=int(pd.Timestamp.now().second+pd.Timestamp.now().minute*60)
        np.random.seed(seed)
        #get_extended - вывести все геномы, что остались на выходе. И их loss
        #ищем МАКСИМУМ
        n_jobs = self.parallel_cores
        x_old = torch.tensor([np.random.random(size=size_x)*(bounds[1]-bounds[0]) + bounds[0] for i in range(int(popsize))],dtype=torch.float32)

        if len(start_point)>0:
            #инициализация некими стартовыми точками
            ln = np.min([len(start_point),len(x_old)])
            x_old[:ln]=torch.tensor(start_point)[:ln]
                      
        time_left = 0
        for t in range(maxiter):
            if n_jobs!=1:
                pool = Pool(processes=n_jobs)
                y_old = pool.map(function, [x for x in x_old])
                pool.close()
                pool.join()
            else:
                y_old = list(map(function, [x for x in x_old]))
            y_old = torch.tensor(y_old,dtype=torch.float32)
            if torch.isnan(torch.tensor(self.current_loss,dtype=torch.float32)):
                self.current_loss = torch.tensor(torch.max(y_old),dtype=torch.float32)
                time_left += len(y_old)

            #отобрать альфачей
            alpha_nums = (-y_old).argsort()[:alpha_count]


            if verbose:
                print(f'iteration {t} y=',y_old[alpha_nums[:elitarism]])

            x_new = []
            for elit in range(elitarism):
                x_new.append(x_old[alpha_nums[elit]].numpy())
            std_vector = torch.std(x_old,axis=0) + 0.0000001
            for child in range(popsize - elitarism):
                #скрещиваем
                crossed_alphas = alpha_nums[[np.random.randint(low=0,high=alpha_count),np.random.randint(low=0,high=alpha_count)]]
                x_c = x_old[alpha_nums[0]].numpy()
                idx = np.random.rand(len(x_c))<0.5
                x_c[idx] = x_old[alpha_nums[1]][idx].numpy()
                x_new.append(x_c)
                idx_muta = np.random.rand(len(x_c))<mutation_p
                if mutation_amplitude_source=='rel':
                    x_c[idx_muta] += np.array((np.random.rand(len(x_c[idx_muta]))-0.5)*2*mutation_r*(x_c[idx_muta]+0.000001))
                elif mutation_amplitude_source=='abs':
                    x_c[idx_muta] += np.array((np.random.rand(len(x_c[idx_muta]))-0.5)*2*mutation_r)
                elif mutation_amplitude_source=='std':
                    x_c[idx_muta] += np.array(((torch.tensor(np.random.rand(len(x_c[idx_muta])),dtype=torch.float32)-0.5)*2*mutation_r*std_vector[idx_muta]).numpy())
                try:
                    x_c = x_c.detach().numpy()
                except:
                    pass
                x_new.append(x_c)
            
            x_old = torch.tensor(x_new,dtype=torch.float32)
            if len(out)>0:
                out[0] = x_old

            mutation_p = mutation_p*(1-mutation_p_e)
            
            #сохраняем промежуточную инфу
            best_genom = x_old[alpha_nums[0]]
            with open(self.init_file , 'wb') as f:
                pickle.dump(best_genom,f,protocol=pickle.HIGHEST_PROTOCOL)

        if n_jobs!=1:
            pool = Pool(processes=n_jobs)
            y_old = pool.map(function, [x for x in x_old])
            pool.close()
            pool.join()
        else:
            y_old = list(map(function, [x for x in x_old]))
        y_old = torch.tensor(y_old,dtype=torch.float32)
        time_left += len(y_old)
        alpha_nums = (-y_old).argsort()[:alpha_count]
        if verbose:
            print('iteration final y=',y_old[alpha_nums[:elitarism]])
        if get_extended:
            return [x_new[alpha_nums[torch.argmax(y_old[alpha_nums])]],x_old, torch.tensor(y_old,dtype=torch.float32)]
        else:
            return torch.tensor(x_new[alpha_nums[torch.argmax(y_old[alpha_nums])]])