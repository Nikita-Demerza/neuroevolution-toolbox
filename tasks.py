import numpy as np
import pandas as pd
import nnet
import neural_tape_controller
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
#Положительные числа - положительные награды.

def custom_test(genom,verbose=False,draw=False):
    q_dynamic = dynamic_test(draw=draw).full_test(genom,verbose,False)
    
    return q_dynamic - np.mean(genom**2)*1e-9
              
    
class dynamic_test:
    #здесь тестируемся на энвайронментах. Они - классы.
    def __init__(self,draw=False):
        self.draw=draw
    def full_test(self,genom,verbose=False,test_set=False):
        if test_set:
            list_envs = [self.jet_table_1,self.jet_table_2,self.jet_table_3,self.jet_table_rot_1,self.jet_table_rot_2,self.aa_gun_1,self.aa_gun_2, self.labirint_1, self.labirint_2]
        else:
            #list_envs = [self.jet_table_1,self.jet_table_2,self.jet_table_rot_1,self.aa_gun_1, self.labirint_1]
            list_envs = [self.aa_gun_2]
            #list_envs = [self.rocket_control]
            #list_envs = [self.labirint_1]
            #list_envs = [self.labirint_1,self.jet_table_1]
            #list_envs = [self.jet_table_1]
        q_arr = []
        q_arr_to_report = []
        for env in list_envs:
            q_arr_local = self.multy_test(env,genom)
            q_arr.extend(q_arr_local)
            q_arr_to_report.append(q_arr_local)
        q_agg = np.mean([np.mean(q_arr), np.min(q_arr), np.percentile(q_arr,0.1), np.percentile(q_arr,0.25), np.percentile(q_arr,0.5)])
        if verbose:
            i = 0
            report = ''
            for env in list_envs:
                nam = str(env)
                nam = nam.split('.')[2]
                nam = nam.split("'")[0]
                report += f'{nam}: {np.round(q_arr_to_report[i],4)} '
                i += 1
            report += f' full={q_agg}'
            print(report)
        return q_agg
    def multy_test(self,env,genom):
        #один энвайронмент, разные сиды
        if self.draw:
            globals()['video'] = []
        q_arr = []
        for seed in range(6):
            q_arr.append(self.test(env,genom,size=30,seed=seed))
        return q_arr
    def test(self,env,genom,size=int(1e6),seed=1,draw=False):
        np.random.seed(seed)
        #это проигрыватель функций
        #создать контроллер
        controller = neural_tape_controller.nt_controller(tacts=1,genom=np.array(genom),input_size=36,output_size=5)
        #зарядить ленту
        controller.init_output(np.zeros([1,size]))
        reward_sum = 0
        env = env(seed)
        out_tape = np.zeros(30)
        for i in range(400):
            action = np.ravel(out_tape)
            #исполнить env
            state, reward, done = env.step(action, draw=self.draw)
            reward_sum += reward
            if np.shape(state)[0]>1:
                shp = np.shape(state)
                state = np.reshape(state,[1,shp[0]])
            out_tape = controller.act(state,reward,done)
        return reward_sum

    class aa_gun_2:
    #задача про зенитку
    #снарядов может быть одновременно несколько
    #самолёт всегда один
    #вид от 1-го лица
        def __init__(self,seed):
            np.random.seed(seed+20)
            self.bullet_speed = 17
            self.gun = {}
            self.gun['x'] = 0
            self.gun['y'] = 2
            self.gun['fi'] = 1
            self.gun['cooldown'] = 10
            self.gun['t'] = 0

            self.bullets = []
            self.make_plane()
        def make_plane(self):
            self.plane = {}
            self.plane['x'] = -100
            self.plane['vx'] = np.random.rand()*7 + 3
            self.plane['y'] = np.random.rand()*100 + 10
            self.plane['vy'] = 0
            self.plane['x_maneur_start'] = np.random.rand()*60 - 40
            self.plane['maneur_t'] = np.random.rand()*12
            self.plane['maneur_vy'] = np.random.rand()*10 - 5
        def step_simulation(self,draw):
            reward = 0
            collide_rad = 6
            g = 1
            coef_time_fraction = 0.1
            if draw:
                x_shift = 100
                y_shift = 10
                im = Image.new('RGB', (200, 100), (128, 128, 256))
                dr = ImageDraw.Draw(im)
                #нарисовать землю
                shape = [(0, 0), (x_shift*2, y_shift)]
                dr.rectangle(shape, fill ="green")
                #нарисовать пушку
                dy = np.sin(self.gun['fi'])*8
                dx = np.cos(self.gun['fi'])*8
                dr.line((self.gun['x']+x_shift,self.gun['y']+y_shift, self.gun['x']+dx+x_shift,self.gun['y']+dy+y_shift), fill="black",width=2)
                #нарисовать снаряды
                for bullet in self.bullets:
                    shape = [(bullet['x']-1+x_shift, bullet['y']-1+y_shift), (bullet['x']+1+x_shift, bullet['y']+1+y_shift)]
                    dr.rectangle(shape, fill ="yellow")
                #нарисовать самолёт
                shape = [(self.plane['x']-2+x_shift, self.plane['y']-1+y_shift), (self.plane['x']+2+x_shift, self.plane['y']+1+y_shift)]
                dr.rectangle(shape, fill ="silver")
                
                
                
            for t in range(int(1/coef_time_fraction)):
                #двигаем самолёт
                self.plane['x'] += self.plane['vx']*coef_time_fraction
                self.plane['y'] += self.plane['vy']*coef_time_fraction
                if (self.plane['x']>self.plane['x_maneur_start']) and (self.plane['maneur_t']>0):
                    self.plane['maneur_t']-=coef_time_fraction
                if self.plane['y']<1:
                    self.plane['y']=1
                #двигаем снаряды
                for i in range(len(self.bullets)):
                    bullet = self.bullets[i]
                    bullet['x'] += bullet['vx']*coef_time_fraction
                    bullet['y'] += bullet['vy']*coef_time_fraction
                    bullet['vy'] -= g*coef_time_fraction
                    bullet['t'] -= coef_time_fraction
                    #проверить попадание
                    if (np.abs(bullet['x']-self.plane['x'])<collide_rad)and(np.abs(bullet['y']-self.plane['y'])<collide_rad):
                        #контакт
                        bullet['t'] = 0
                        self.make_plane()
                        
                        reward = 100
                        if draw:
                            r=15
                            dr.ellipse((bullet['x']-r+x_shift, bullet['y']-r+y_shift, bullet['x']+r+x_shift, bullet['y']+r+y_shift), fill='white')
                            
                    elif (np.abs(bullet['x']-self.plane['x'])<collide_rad*2)and(np.abs(bullet['y']-self.plane['y'])<collide_rad*2):
                        reward = 0.3#обозначить, что снаряд прошёл близко
                    elif (np.abs(bullet['x']-self.plane['x'])<collide_rad*4)and(np.abs(bullet['y']-self.plane['y'])<collide_rad*4):
                        reward = 0.03#обозначить, что снаряд прошёл близко
                    elif (np.abs(bullet['x']-self.plane['x'])<collide_rad*6)and(np.abs(bullet['y']-self.plane['y'])<collide_rad*6):
                        reward = 0.003#обозначить, что снаряд прошёл близко
                    
                    if bullet['y']<0:
                        bullet['t'] = 0
                    self.bullets[i] = bullet
                for i in np.arange(len(self.bullets)-1,-1,-1):
                    i = int(i)
                    if self.bullets[i]['t']<=0:
                        del self.bullets[i]
                        #удалить старые
            if draw:
                #if reward>0:
                #    print('reward',reward)
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                globals()['video'].append(im)
                plt.imshow(im)
                plt.show()
            return reward

        def step(self,action, draw=False):
            #действия: поворот пушки, выстрел
            if action[0]>0.5:
                action[0]=0.5
            elif action[0]<-0.5:
                action[0]=-0.5
            self.gun['fi'] += action[0]
            min_angle=0.15
            if self.gun['fi']<0+min_angle:
                self.gun['fi'] = 0+min_angle
            elif self.gun['fi']>3.14-min_angle:
                self.gun['fi'] = 3.14-min_angle
            self.gun['t'] -= 1
            shoot = (action[1]>0.1) and (self.gun['t']<=0)
            reward = 0
            if shoot:
                #создать новый снаряд
                bullet_speed  = self.bullet_speed
                bullet_t = 20
                bullet = {}
                bullet['x'] = self.gun['x']
                bullet['y'] = self.gun['y']
                bullet['vx'] = bullet_speed*np.cos(self.gun['fi'])
                bullet['vy'] = bullet_speed*np.sin(self.gun['fi'])
                bullet['t'] = bullet_t
                self.bullets.append(bullet)
                self.gun['t'] = self.gun['cooldown']
                #reward = -1

            
            if self.plane['x']>=100:
                #всё, улетел, пересоздаём
                self.make_plane()
                reward = reward -10
            done = 0
            reward = reward + self.step_simulation(draw=draw)
            if len(self.bullets)>0:
                bx,by = self.bullets[-1]['x'],self.bullets[-1]['y']
            else:
                bx = -1
                by = -1
            br = np.sqrt((bx-self.gun['x'])**2 + (by-self.gun['y'])**2)
            bfi = np.arctan2((bx-self.gun['x']),(by-self.gun['y']))

            pr = np.sqrt((self.plane['x']-self.gun['x'])**2 + (self.plane['y']-self.gun['y'])**2)
            pfi = np.arctan2((self.plane['x']-self.gun['x']),(self.plane['y']-self.gun['y']))

            state = np.array([br,bfi,pr,pfi,self.gun['fi'],self.gun['cooldown']],dtype=np.float16)
            
            return state, reward, done
        
    class rocket_control:
        #задача про ракету
        #у нас есть ракета, ей можно двигатель включить послабее/посильнее. Ракета должна попасть в некую точку
        def __init__(self,seed):
            np.random.seed(seed)
            self.tx = (np.random.rand()-0.5)*700
            self.ty = np.random.rand()*150
            self.rocket = {}
            self.rocket['x'] = 0
            self.rocket['y'] = 0
            self.rocket['vx'] = 0
            self.rocket['vy'] = 0
            self.rocket['fi'] = 3.14/2
            self.rocket['w'] = 0
            self.rocket['fr_w'] = 0.15#коэффициент сопротивления вращению
            self.rocket['stab_w'] = 0.05#коэффициент стабилизации
            self.rocket['fr_v'] = 0.05#коэффициент сопротивления скорости (обычное воздушное трение)
            self.rocket['m_dry'] = 4
            self.rocket['fuel'] = 4
            self.rocket['dv'] = 13
            self.rocket['throttle'] = 1#управляемая величина
            self.rocket['steer_max'] = 0.07
            self.rocket['steer_power'] = 1#управляемая величина
            self.g = 0.5
            self.trajectory = []
        def step_simulation(self,draw):
            m = self.rocket['m_dry']+self.rocket['fuel']
            #движение по скорости
            self.rocket['x'] += self.rocket['vx']
            self.rocket['y'] += self.rocket['vy']
            self.rocket['fi'] += self.rocket['w']
            #стабилизация
            v_direction = np.arctan2(self.rocket['vy'],self.rocket['vx'])
            self.rocket['w'] += (v_direction-self.rocket['fi'])*self.rocket['stab_w']
            #трение
            self.rocket['w'] *= (1-self.rocket['fr_w'])
            self.rocket['vx'] -= self.rocket['vx']*self.rocket['fr_v']/(m)
            self.rocket['vy'] -= self.rocket['vy']*self.rocket['fr_v']/(m)
            #разгон
            self.rocket['vx']+= np.cos(self.rocket['fi'])*self.rocket['dv']*self.rocket['throttle']/(m)
            self.rocket['vy']+= np.sin(self.rocket['fi'])*self.rocket['dv']*self.rocket['throttle']/(m)
            #управление поворотом
            self.rocket['w'] += self.rocket['steer_power']*self.rocket['steer_max']*np.sqrt(self.rocket['vx']*self.rocket['vx']+self.rocket['vy']*self.rocket['vy'])
            #гравитация
            self.rocket['vy'] -= self.g
            #запомнить точку
            self.trajectory.append((self.rocket['x'],self.rocket['y']))
            
                
        def step(self,action, draw=False):
            #действия: поворот ракеты, газ
            if action[0]>1:
                action[0]=1
            elif action[0]<-1:
                action[0]=-1
            self.rocket['steer_power'] = action[0]
            action[1]+=0.5#центрируем на 0.5, чтобы лучше обучился
            if action[1]>1:
                action[1]=1
            elif action[1]<0:
                action[1]=0
            self.rocket['throttle'] = action[1]
            
            if self.rocket['fuel']<self.rocket['throttle']*0.1:
                self.rocket['throttle'] = 0
            self.rocket['fuel']-=self.rocket['throttle']*0.1
            
            self.step_simulation(draw=draw)
            
            reward = 1/(np.abs(self.rocket['x']-self.tx)+np.abs(self.rocket['y']-self.ty))
            if reward>0.5:
                reward = 0.5
            done = 0
            state = np.array([self.tx-self.rocket['x'],self.ty-self.rocket['y'], self.rocket['fi'],self.rocket['w'], self.rocket['vx'],self.rocket['vy'],self.rocket['fuel']],dtype=np.float16)
            
            if draw:
                x_shift = 600
                y_shift = 10
                im = Image.new('RGB', (1200, 500), (100, 100, 200))
                dr = ImageDraw.Draw(im)
                #нарисовать землю
                shape = [(0, 0), (x_shift*2, y_shift)]
                dr.rectangle(shape, fill ="green")
                #нарисовать ракету
                dy = np.sin(self.rocket['fi'])*7
                dx = np.cos(self.rocket['fi'])*7
                dr.line((self.rocket['x']+x_shift,self.rocket['y']+y_shift, self.rocket['x']+dx+x_shift,self.rocket['y']+dy+y_shift), fill="silver",width=2)
                #нарисовать цель
                shape = [(self.tx-3+x_shift, self.ty-3+y_shift), (self.tx+3+x_shift, self.ty+3+y_shift)]
                dr.rectangle(shape, fill ="red")
                #нарисовать траекторию
                for point in self.trajectory:
                    shape = [(point[0]-2+x_shift, point[1]-2+y_shift), (point[0]+2+x_shift, point[1]+2+y_shift)]
                    dr.rectangle(shape, fill ="white")
            
            
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                globals()['video'].append(im)
                #plt.imshow(im)
                #plt.show()
            return state, reward, done
        
    class jet_table_1:
        def __init__(self,seed):
            np.random.seed(seed)
            self.my_x = 0
            self.my_y = 0
            self.my_vx = 0
            self.my_vy = 0

            self.k_fr = 0.3

            self.t_x = np.random.normal()*10
            self.t_y = np.random.normal()*10
            self.t_vx = 0
            self.t_vy = 0
            self.path_blue = []
            self.path_red = []
        def step(self,action, draw=False):
            if action[0]>1:
                action[0]=1
            elif action[0]<-1:
                action[0]=-1
            if action[1]>1:
                action[1]=1
            elif action[1]<-1:
                action[1]=-1

            self.my_vx += action[0]
            self.my_vy += action[1]
            self.my_x += self.my_vx
            self.my_y += self.my_vy
            self.my_vx *= (1-self.k_fr)
            self.my_vy *= (1-self.k_fr)

            self.t_vx += np.random.normal()*1.1
            self.t_vy += np.random.normal()*1.1
            self.t_vx *= (1-self.k_fr)
            self.t_vy *= (1-self.k_fr)
            self.t_x += self.t_vx
            self.t_y += self.t_vy

            reward = 5/((self.t_x-self.my_x)**2 + (self.t_y-self.my_y)**2)
            if reward>5:
                reward = 5
            done = 0
            state = np.array([self.my_x,self.my_y,self.t_x,self.t_y]+[0]*(25-4),dtype=np.float16)
            
            if draw:
                x_shift = 150
                y_shift = 150
                im = Image.new('RGB', (300, 300), (256, 256, 256))
                dr = ImageDraw.Draw(im)
                #нарисовать траектории
                for point in self.path_red:
                    shape = [(point[0]+x_shift, point[1]+y_shift), (point[0]+1+x_shift, point[1]+1+y_shift)]
                    dr.rectangle(shape, fill ="red")
                for point in self.path_blue:
                    shape = [(point[0]+x_shift, point[1]+y_shift), (point[0]+1+x_shift, point[1]+1+y_shift)]
                    dr.rectangle(shape, fill ="blue")
                #нарисовать агента
                shape = [(self.my_x-2+x_shift, self.my_y-2+y_shift), (self.my_x+2+x_shift, self.my_y+2+y_shift)]
                dr.rectangle(shape, fill ="blue")
                self.path_blue.append((self.my_x,self.my_y))
                #нарисовать цель
                shape = [(self.t_x-2+x_shift, self.t_y-2+y_shift), (self.t_x+2+x_shift, self.t_y+2+y_shift)]
                dr.rectangle(shape, fill ="red")
                self.path_red.append((self.t_x,self.t_y))

                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                globals()['video'].append(im)
                plt.imshow(im)
                plt.show()

            return state, reward, done

    class labirint_1:
        #есть матрица из 0, 1, 2, 3. 0 - проход, 1 - стена, 2 - награда, 3 - наказание. Агент видит матрицу вокруг себя, скажем, 5х5
        def __init__(self,seed):
            np.random.seed(seed+15)
            self.my_x = 25
            self.my_y = 15
            self.full_size = 40
            self.window_rad = 3
            self.mx = np.zeros([self.full_size,self.full_size])
            idx = np.random.rand(self.full_size,self.full_size)<0.25
            self.mx[idx] = 1 #стена
            idx = np.random.rand(self.full_size,self.full_size)<0.07
            self.mx[idx] = 2 #награда
            idx = np.random.rand(self.full_size,self.full_size)<0.05
            self.mx[idx] = 3 #наказание
            self.mx[:self.window_rad,:] = 1
            self.mx[-self.window_rad:,:] = 1
            self.mx[:,:self.window_rad] = 1
            self.mx[:,-self.window_rad:] = 1
            #
            self.mx[self.my_x-3:self.my_x+3,self.my_y-4:self.my_y+4] = 0


        def step(self,action, draw=False):
            vx = 0
            vy = 0
            amax = np.argmax(action[:5])
            if amax==1:
                vx = 1
            elif amax==2:
                vx = -1
            elif amax==3:
                vy = 1
            elif amax==4:
                vy = -1
            reward = 0
            done = 0

            if self.mx[self.my_x+vx, self.my_y+vy]!=1:
                self.my_x += vx
                self.my_y += vy
            if self.mx[self.my_x, self.my_y]==2:
                reward = 6
                self.mx[self.my_x, self.my_y] = 0
            elif self.mx[self.my_x, self.my_y]==3:
                reward = -6
                self.mx[self.my_x, self.my_y] = 0
            state = np.array(self.mx[self.my_x-self.window_rad:self.my_x+self.window_rad,self.my_y-self.window_rad:self.my_y+self.window_rad],dtype=np.float16)
            state = np.ravel(state)
            
            if draw:
                scale = 7
                im = Image.new('RGB', (self.full_size*scale, self.full_size*scale), (256, 256, 256))
                dr = ImageDraw.Draw(im)
                #нарисовать агента
                shape = [((self.my_x-1)*scale, (self.my_y-1)*scale), ((self.my_x+1)*scale, (self.my_y+1)*scale)]
                dr.rectangle(shape, fill ="cyan")
                #нарисовать лабиринт
                for x in range(self.full_size):
                    for y in range(self.full_size):
                        shape = [((x)*scale, (y)*scale), ((x+1)*scale, (y+1)*scale)]
                        if self.mx[x,y]==1:
                            color = 'black'
                        elif self.mx[x,y]==2:
                            color = 'green'
                        elif self.mx[x,y]==3:
                            color = 'red'
                        if self.mx[x,y]>0:
                            dr.rectangle(shape, fill=color)
                if reward>0:
                    shape = [((self.my_x-10)*scale, (self.my_y-10)*scale), ((self.my_x+10)*scale, (self.my_y+10)*scale)]
                    dr.rectangle(shape, outline="green")
                    shape = [((self.my_x-7)*scale, (self.my_y-7)*scale), ((self.my_x+7)*scale, (self.my_y+7)*scale)]
                    dr.rectangle(shape, outline="green")
                    shape = [((self.my_x-5)*scale, (self.my_y-5)*scale), ((self.my_x+5)*scale, (self.my_y+5)*scale)]
                    dr.rectangle(shape, outline="green")
                elif reward<0:
                    shape = [((self.my_x-10)*scale, (self.my_y-10)*scale), ((self.my_x+10)*scale, (self.my_y+10)*scale)]
                    dr.rectangle(shape, outline="red")
                    shape = [((self.my_x-7)*scale, (self.my_y-7)*scale), ((self.my_x+7)*scale, (self.my_y+7)*scale)]
                    dr.rectangle(shape, outline="red")
                    shape = [((self.my_x-5)*scale, (self.my_y-5)*scale), ((self.my_x+5)*scale, (self.my_y+5)*scale)]
                    dr.rectangle(shape, outline="red")
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                globals()['video'].append(im)
                plt.imshow(im)
                plt.show()
            return state, reward, done