# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:48:21 2025

@author: CampmansGHP
"""

import numpy as np
import matplotlib.pyplot as plt


class transport_model:
    
    def __init__(self,C=0.1,Usal = 1, Uair=10, Shields = 0.5, Omega = 0.0):
        self.epsilon = 1e-16
        self.dt = 0.01
        self.rho_air = 1.225
        self.rho_sand = 2600
        self.g = 9.81
        self.Heigth = 0.2
        self.d = 250e-6
        self.Vgrain = 4/3*np.pi*(self.d/2)**3
        self.Agrain = np.pi*(self.d/2)**2
        self.Mass_grain = self.Vgrain*self.rho_sand
        self.Cd_bed = 0.003  
        
        self.u_star = np.sqrt(Shields*(self.rho_sand-self.rho_air)*self.g*self.d/self.rho_air)
        
        self.C    = C
        self.Usal = Usal
        self.Uair = Uair#np.sqrt(2*self.u_star**2/self.Cd_bed)
        
        self.Shields = Shields
        self.Omega = Omega
    
    
    #%% Function definitions of the source terms.
    
    # Matm:
    def calc_u_star_from_Shields(self):
        self.u_star = np.sqrt(self.Shields*(self.rho_sand-self.rho_air)*self.g*self.d/self.rho_air)
    
    def calc_Mair_atm(self):
        self.calc_u_star_from_Shields()
        return self.rho_air*self.u_star**2
    
    # Mbed:
    def calc_Mair_bed(self):
        Mdrag = self.calc_Mdrag()
        Mbed_air_only = 0.5*self.rho_air*self.Uair*abs(self.Uair)*self.Cd_bed
        # if Mdrag>Mbed_air_only:
        #     print('Mdrag =',Mdrag)
        #     print('Mbed_air_only = ',Mbed_air_only)
        # Mbed = Mbed_air_only - Mdrag
        # if Mbed_air_only<Mdrag:
        #     print('----')
        #     print('Mbed_air_only=',Mbed_air_only)
        #     print('Mdrag=',Mdrag)
        #     print('----')
        # Mair_bed = max(Mbed_air_only - Mdrag,0.0)
        B = 1/0.6
        # print('Mdrag = ',Mdrag)
        p = 2
        Mair_bed = Mbed_air_only*(1/(1+(B*Mdrag)**p))
        if Mair_bed<0:
            print('Warning! ~line 65')
        return Mair_bed
    
    """
    Bed interaction terms 
    """   
    # Bed interaction terms
    def calc_U_im(self,Usal):
        sqrtgd = np.sqrt(self.g*self.d)
        return sqrtgd *(0.04*(Usal/sqrtgd)**1.54 + 32.23)
    
    def calc_U_D_Xiuqi(self,Usal):
        if self.Omega>0.0:
            U_D = 10.10*np.sqrt(self.g*self.d)
        else:
            U_D = 6.66*np.sqrt(self.g*self.d)
        return U_D
    
    def calc_U_D_test(self,Usal):
        Udep = 0.8*Usal
        Uy0 = Udep*np.tan(15/180*np.pi)
        U_D = (Uy0**2 + Udep**2)**0.5
        return U_D
    
    def calc_U_D(self,Usal):
        # U_D = self.calc_U_D_test(Usal)
        U_D = self.calc_U_D_Xiuqi(Usal)
        return U_D
    
    def calc_U_E_im(self,U_im):
        sqrtgd = np.sqrt(self.g*self.d)
        A = 9.02*self.Omega + 4.53
        if self.Omega>0:
            B = -0.24*self.Omega+0.07
        else:
            B = 0
        U_E_im = sqrtgd*A*(U_im/sqrtgd)**B
        return U_E_im
    
    def calc_N_E_Xiuqi(self,Uinc,Omega):
        sqrtgd = np.sqrt(self.g*self.d)
        N_E = Uinc/sqrtgd*(0.04-0.04*Omega**0.23) # Xiuqi
        return N_E 
   
    
   
    def calc_N_E_test(self,Uinc,Omega):
        sqrtgd = np.sqrt(self.g*self.d)
        
        N_E = np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*5
        N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
        p = 8
        ##
        p2 = 2
        A = 100
        Uinc_half_min = 1.0
        Uinc_half_max = 6
        Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
        ##
        #Uinc_half = 0.5+40*Omega**0.5
        B = 1/Uinc_half**p
        N_E = (1-1./(1.+B*Uinc**p))*2.5 * Uinc**(1/10)
        return N_E 
    
    def calc_N_E_test2(self,Uinc,Omega):
        # sqrtgd = np.sqrt(self.g*self.d)
        # N_E = Uinc/sqrtgd*(0.04-0.04*Omega**0.23) # Xiuqi
        # N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
        # N_E = (1-(1000*Omega+0.2)/(Uinc**2+(1000*Omega+0.2)))*2
        
        p = 8
        p2 = 2
        A = 100
        Uinc_half_min = 5# 0.75
        Uinc_half_max = 8
        Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
        B = 1/Uinc_half**(p-5*(1-Omega*10))
        N_E = (1-1./(1.+B*Uinc**(p-5*(1-Omega*10))))*2.5 * Uinc**(1/10)
        return N_E  
    
    def calc_N_E_test3(self,Uinc,Omega):
        N_E_Xiuqi = self.calc_N_E_Xiuqi(Uinc,Omega)
        # N_E = (1-(10*Omega+0.2)/(Uinc**2+(10*Omega+0.2)))*np.sqrt(Uinc/sqrtgd)*(0.04-0.04*Omega**0.23)*7
        # N_E = (1-(1000*Omega+0.2)/(Uinc**2+(1000*Omega+0.2)))*2
        
        p = 8
        ##
        p2 = 2
        A = 100
        Uinc_half_min = 1.0
        Uinc_half_max = 2.0
        Uinc_half = Uinc_half_min + (Uinc_half_max - Uinc_half_min)*(A*Omega)**p2/((A*Omega)**p2+1)
        ##
        #Uinc_half = 0.5+40*Omega**0.5
        B = 1/Uinc_half**p
        N_E = (1-1./(1.+B*Uinc**p))*N_E_Xiuqi
        return N_E  
   
    def calc_N_E(self,Uinc,Omega):
        # N_E = self.calc_N_E_Xiuqi(Uinc,Omega)
        # N_E = self.calc_N_E_test(Uinc,Omega)
        # N_E = self.calc_N_E_test2(Uinc,Omega)
        N_E = self.calc_N_E_test3(Uinc,Omega)
        return N_E
   
    def calc_U_E_D(self,U_D):
        sqrtgd = np.sqrt(self.g*self.d)
        A = 9.02*self.Omega + 4.53
        if self.Omega>0:
            B = -0.24*self.Omega+0.07
        else:
            B = 0
        U_E_im = sqrtgd*A*(U_D/sqrtgd)**B
        return U_E_im
    
    def calc_Pr_eff(self,Usal): # from Xiuqi paper 3
        A = 0.76
        Uc = 3.84
        Pr_min = 0.0
        Pr_max = 1.0
        Pr_eff = Pr_min + (Pr_max-Pr_min)*(1-np.exp(-(Usal/Uc))**A)
        return Pr_eff
    
    def calc_Pr_Xiuqi_paper2(self,Usal):  # from Xiuqi paper 2
        Pr = 0.94*np.exp(-7.12*np.exp(-0.1*Usal/np.sqrt(self.g*self.d)))
        return Pr
    
    def calc_Pr_Jiang24(self,Usal,H): #  from Jiang et al 2024 (A Wind Tunnel Experiment Study on Splash Functions During Sand Saltation)
        if H<=0.5*self.d:
            Pr = 1
        else:
            sqrtgd = np.sqrt(self.g*self.d)
            Pr = 0.9945*(H/self.d)**(-0.0166)*(1-np.exp(-0.1992*(H/self.d)**(-0.8686)*Usal/sqrtgd))
        return Pr
    
    def calc_Pr_Test(self,Usal):
        # sqrtgd = np.sqrt(self.g*self.d)
        # H = self.d*1.5
        # Pr = (H/self.d)**(-0.0166)*(1-np.exp(-0.1992*(H/self.d)**(-0.8686)*Usal/sqrtgd))
        p = 1.5
        a=1/1.18
        # a=1/0.25
        y_max = 0.999
        y_min = 0
        Pr = y_min + (y_max-y_min)*(a*Usal)**p/((a*Usal)**p+1)
        
        return Pr
    
    def calc_Pr(self,Usal):
        # Pr = self.calc_Pr_eff(Usal) # from Xiuqi paper 3
        Pr = self.calc_Pr_Xiuqi_paper2(Usal)
        # Pr = self.calc_Pr_Jiang24(Usal,H=self.d*1.5)
        # Pr = self.calc_Pr_Test(Usal)
        return Pr
    
    def calc_theta_im(self,U_im):
        sqrtgd = np.sqrt(self.g*self.d)
        alpha = 50.40 - 25.53*self.Omega**0.5 
        beta = -100.12*(1-np.exp(-2.34*self.Omega)) + 159.33
        theta_im = np.arcsin(alpha/(U_im/sqrtgd + beta))
        return theta_im
    
    def calc_theta_D(self,U_D):
        # sqrtgd = np.sqrt(self.g*self.d)
        # print('U_D =',U_D)
        # print('arg arcsin = ',163.68/(U_D/sqrtgd + 154.65))
        # theta_D = 0.28*np.arcsin(163.68/(U_D/sqrtgd + 154.65))
        theta_D = 20/180*np.pi
        return theta_D
    
    def calc_theta_re(self,U_im):
        sqrtgd = np.sqrt(self.g*self.d)
        theta_re = np.arcsin(-0.0003*U_im/sqrtgd + 0.52)
        return theta_re
    
    def calc_restitution(self,U_im):
        sqrtgd = np.sqrt(self.g*self.d)
        mu = 312.48
        sigma = 156.27
        e = 3.05*(U_im/sqrtgd)**(-0.47) + 0.12*np.log(1 + 1061.81*self.Omega)*np.exp(- (U_im/sqrtgd - mu)**2/(2*sigma**2) )
        return e
    
    def calc_T_jump_ballistic_assumption(self,Usal):
        Uy0 = Usal*np.tan(15/180*np.pi)
        Tjump = 0.1+np.sqrt(4*Uy0/self.g) # Xiuqi: ?
        return Tjump
    
    def calc_T_jump_Xiuqi(self,Usal):
        U_im = self.calc_U_im(Usal)
        Tjump = 0.04*U_im**0.84
        return Tjump
    
    def calc_T_jump_Test(self,Usal):
        Tjump_ballistic = self.calc_T_jump_ballistic_assumption(Usal)
        Tjump_Xiuqi = self.calc_T_jump_Xiuqi(Usal)
        Tjump  = 0.5*Tjump_Xiuqi + 0.5*Tjump_ballistic
        return Tjump
    
    def calc_T_jump(self,Usal):
        # Tjump = self.calc_T_jump_ballistic_assumption(Usal)
        # Tjump = self.calc_T_jump_Xiuqi(Usal)
        Tjump = self.calc_T_jump_Test(Usal)
        return Tjump
    
    def calc_T_jump_dep(self,Usal): # not used (yet), only for test plotting after the model has ran.
        U_D  = self.calc_U_D(Usal)
        Tjump_dep = self.calc_T_jump(U_D)*np.ones_like(Usal)
        return Tjump_dep
    
    def test_im_D_concentration(self,Usal):
        C = 1
        U_im = self.calc_U_im(Usal)
        U_D  = self.calc_U_D(Usal)
        T_D  = 0.7*U_D**0.66
        # T_im = 0.04*U_im**0.84 
        T_im = self.calc_T_jump(Usal)
        Pr_eff = self.calc_Pr(Usal)
        
        C_im = Pr_eff*T_im/(Pr_eff*T_im +(1-Pr_eff)*T_D)*C
        C_D = C-C_im
        return C_im, C_D
    
    def calc_particle_bed_interaction_terms(self): # From draft paper Xiuqi:
        sqrtgd = np.sqrt(self.g*self.d)
        U_im = self.calc_U_im(self.Usal)
        U_D  = self.calc_U_D(self.Usal)
        
        # T_D  = 0.7*U_D**0.66
        T_D = self.calc_T_jump(U_D) # Xiuqi: self.calc_T_jump_dep(self.Usal)?
        # T_im = 0.04*U_im**0.84 
        T_im = self.calc_T_jump(self.Usal)
        
        Pr_eff = self.calc_Pr(self.Usal) # Xiuqi: Usal as input?
        
        C_im = Pr_eff*T_im/(Pr_eff*T_im +(1-Pr_eff)*T_D)*self.C
        C_D = self.C-C_im
        if C_D<0:
            print('C_D=',C_D)
        
        D = C_D/T_D
        
        N0 = 0.04-0.04*self.Omega**0.23
        
        # N_E_im = N0*U_im/sqrtgd
        # N_E_D  = N0*U_D/sqrtgd
        N_E_im = self.calc_N_E(U_im,self.Omega) # Xiuqi: adjusted
        N_E_D = self.calc_N_E(U_D,self.Omega)
        
        E = N_E_im*C_im/T_im + N_E_D*C_D/T_D
        if E<0:
            print('E = ',E)
        
        e = self.calc_restitution(U_im)
        U_re = U_im*e
        U_E_im = self.calc_U_E_im(U_im)
        U_E_D  = self.calc_U_E_D(U_D)
        
        theta_im = self.calc_theta_im(U_im)
        theta_E  = 40/180*np.pi
        theta_D  = self.calc_theta_D(U_D) # Xiuqi: a constant
        theta_re = self.calc_theta_re(U_im)
        
        M_E  = N_E_im*C_im/T_im*U_E_im*np.cos(theta_E) + N_E_D*C_D/T_D*U_E_D*np.cos(theta_E)
        # M_D  = D*U_D*np.cos(theta_D)
        M_D  = min(D*U_D*np.cos(theta_D), D*self.Usal) 
        M_im = C_im*U_im*np.cos(theta_im)/T_im
        M_re = C_im*U_re*np.cos(theta_re)/T_im
        
        if M_D>D*self.Usal:
            print('More momentum is leaving per particle by deposition, than that there is on average in the saltation layer')
            print('M_D =',M_D)
            print('D =',D)
            print('Usal =',self.Usal)
            print('U_D =',U_D)
            print('-----------')
        return E,D,M_E,M_D,M_im,M_re
    
    """
    Drag terms 
    """    
    # Mdrag: The momentum source term from the wind to the saltating sediment
    nu = 1.4e-5
    Ruc = 24
    Cd_inf = 0.5
    def calc_Re(self,Urel):
        Re = Urel*self.d/self.nu
        return Re
    
    def calc_Cd(self,Urel):
        Re = self.calc_Re(Urel)
        Cd = (np.sqrt(self.Cd_inf)+np.sqrt(self.Ruc/Re))**2
        # print('Cd_inf=',self.Cd_inf)
        # print('Cd = ',Cd)
        # print('Urel = ',Urel)
        return Cd
    
    def calc_Mdrag(self):
        Ngrains = self.C/self.Mass_grain
        alpha = 0.9
        Ueff = alpha*self.Uair
        Urel = Ueff-self.Usal
        Cd = self.calc_Cd(Urel)
        
        Mdrag = 0.5*self.rho_air*Urel*abs(Urel)*Cd*self.Agrain*Ngrains # drag term based on uniform velocity
        # reducing the drag efficiency, depending on the dragg force itself. If there is more drag, we assume that many particles experience lower air velocity, and thus a lower effective dragg
        # B = 1/(0.5*self.rho_air*self.u_star**2)
        # p = 2
        # alpha = 0.5 # alpha is the minimum ratio of the originally determined dragg force.
        # Mdrag = Mdrag*((1-alpha)*(1/(1+(B*Mdrag)**p))+alpha)
        return Mdrag
            
    
    """
    Update source terms 
    """
    #%% The function that calls all the calculations of the source terms.
    def calc_source_terms(self):
        E,D,M_E,M_D,M_im,M_re = self.calc_particle_bed_interaction_terms()
        # if self.Omega==0:
        #     # print('test')
        #     E = E + 0.1*max(0.0,(self.Uair-5.0)**2)
        Mdrag = self.calc_Mdrag() 
        Mair_atm  = self.calc_Mair_atm()
        Mair_bed  = self.calc_Mair_bed()
        return E,D,M_E,M_D,M_im,M_re,Mdrag,Mair_atm,Mair_bed
    
    """
    Timestep 
    """
    #%% The time integration timestep
    def timestep(self):
        dt,epsilon,C,Usal,Uair,rho_air,Heigth = self.dt,self.epsilon,self.C,self.Usal,self.Uair,self.rho_air,self.Heigth
        E,D,M_E,M_D,M_im,M_re,Mdrag,Mair_atm,Mair_bed = self.calc_source_terms()
        
        if D<0:
            print('D=',D)
        # if self.t<1.2 and self.Omega>0:
        #     print('artificial Errosion prevention is on (line~337)')
        #     E = E*self.t/1.2
        if M_E>Usal*E:
            print('M_E =',M_E)
            print('Usal =',Usal)
            print('E = ',E)
            print('----')
        if M_re>M_im:
            print('M_re =',M_re)
            print('M_im =',M_im)
            print('----')
        
        C_new     = C + dt*(E-D)
        CUsal_new = C*Usal + dt*(Mdrag + M_E + M_re - M_D - M_im) 
        Usal_new  = CUsal_new/(C_new+epsilon)
        Uair_new  = Uair + dt/(rho_air*Heigth)*(-Mdrag + Mair_atm - Mair_bed) # Xiuqi: No (1-phi) term
        self.C = C_new
        self.Usal = Usal_new
        self.Uair = Uair_new
    
    """
    Run function 
    """
    #%% The run function, that does the timestep loop, and defines a timevector for the model output.
    def run(self,T):
        t_vec = np.arange(0.0,T,self.dt)
        N = len(t_vec)
        C_vec = np.zeros(N)
        Usal_vec = np.zeros(N)
        Uair_vec = np.zeros(N)
        C_vec[0]=self.C
        Usal_vec[0]=self.Usal
        Uair_vec[0]=self.Uair
        for n in range(N-1):
            self.t = t_vec[n]
            self.timestep()
            C_vec[n+1],Usal_vec[n+1],Uair_vec[n+1] = self.C,self.Usal,self.Uair
        return C_vec, Usal_vec, Uair_vec, t_vec
    #%% End of the transport_model class
            

if __name__ == "__main__":

    plt.close('all')
    
    scaled_velocity = True
    
    Shields = 0.06
    Omega_dry = 0.0
    Omega_wet = 0.1
    C = 0.1/5
    Usal = 0.55
    Uair = 13
    T = 5.0
    
    fig,ax = plt.subplots(nrows=4)
    model_run_dry = transport_model(C = C, Usal = Usal, Uair = Uair, Shields=Shields, Omega=Omega_dry)
    if scaled_velocity:
        velocity_scale = np.sqrt(9.81*model_run_dry.d)
    else:
        velocity_scale = 1
    model_run_dry.C = 0.5
    C_vec, Usal_vec, Uair_vec, t_vec = model_run_dry.run(T=T)
    ax[0].set_title('Shields='+str(Shields))
    
    ax[0].plot(t_vec,C_vec*Usal_vec,label='Q dry')
    ax[0].set_ylabel('Q [kg/m/s]')
    ax[1].plot(t_vec,C_vec,label='C dry')
    ax[1].set_ylabel('C [kg/m^2]')
    ax[2].plot(t_vec,Usal_vec,label='Usal dry')
    ax[2].set_ylabel('Usal [m/s]')
    ax[3].plot(t_vec,Uair_vec,label='Uair dry')
    ax[3].set_ylabel('Uair [m/s]')
    model_run_wet = transport_model(C = C, Uair = Uair, Shields=Shields, Omega=Omega_wet)
    C_vec, Usal_vec, Uair_vec, t_vec = model_run_wet.run(T=T)
    ax[0].plot(t_vec,C_vec*Usal_vec,label='Q wet')
    ax[1].plot(t_vec,C_vec,label='C wet')
    ax[2].plot(t_vec,Usal_vec,label='Usal wet')
    ax[3].plot(t_vec,Uair_vec,label='Uair wet')
    ax[3].set_xlabel('t [s]')
    for i in range(4):
        ax[i].legend()
        ax[i].grid('on')
        ax[i].set_ylim(bottom=0)
    
    fig,ax = plt.subplots()
    Usal_range = np.linspace(1e-2,15,150)
    Pr_eff_range = model_run_wet.calc_Pr_eff(Usal_range)
    Pr_Xiuqi_paper2_range = model_run_wet.calc_Pr_Xiuqi_paper2(Usal_range)
    Pr_Test_range = model_run_wet.calc_Pr_Test(Usal_range)
    Pr_range = model_run_wet.calc_Pr(Usal_range)
    Pr_Jiang24_range = model_run_wet.calc_Pr_Jiang24(Usal_range,H=model_run_wet.d*1.5)
    ax.plot(Usal_range/velocity_scale,Pr_eff_range,label='Pr_eff (Draft paper3)')
    ax.plot(Usal_range/velocity_scale,Pr_Xiuqi_paper2_range,label='Pr (Xiuqi paper 2)')
    ax.plot(Usal_range/velocity_scale,Pr_Jiang24_range,label='Pr_Jiang24')
    ax.plot(Usal_range/velocity_scale,Pr_Test_range,label='Pr_Test')
    ax.plot(Usal_range/velocity_scale,Pr_range,'k--',label='Pr (used in simulation)')
    ax.set_ylim(bottom=0)
    ax.grid('on')
    if scaled_velocity:
        ax.set_xlabel('U/sqrt(gd) [-]')
    else:
        ax.set_xlabel('U [m/s]')
    ax.legend()
    
    fig,ax = plt.subplots()
    Uinc_range = np.linspace(0,15,50)
    color = ['r','b','g','c','k','y']
    i=0
    for omega in [0,0.1]:#[0,0.01,0.05,0.1,0.2]:
        N_E_range = model_run_wet.calc_N_E_Xiuqi(Uinc_range,omega)
        ax.plot(Uinc_range/velocity_scale,N_E_range,'-',color=color[i],label='Omega (Xiuqi)='+str(omega))
        N_E_range = model_run_wet.calc_N_E_test(Uinc_range,omega)
        ax.plot(Uinc_range/velocity_scale,N_E_range,'--',color=color[i],label='Omega (test)='+str(omega))
        N_E_range = model_run_wet.calc_N_E_test2(Uinc_range,omega)
        ax.plot(Uinc_range/velocity_scale,N_E_range,':',color=color[i],label='Omega (test2)='+str(omega))
        N_E_range = model_run_wet.calc_N_E_test3(Uinc_range,omega)
        ax.plot(Uinc_range/velocity_scale,N_E_range,'-.',color=color[i],label='Omega (test3)='+str(omega))
        N_E_range = model_run_wet.calc_N_E(Uinc_range,omega)
        ax.plot(Uinc_range/velocity_scale,N_E_range,'k--',label='Omega (model)='+str(omega))
        i=i+1
    ax.legend()
    ax.set_ylim(bottom=0,top=5.8)
    ax.set_xlim(left=0)
    ax.grid('on')
    ax.set_ylabel('N_E')
    if scaled_velocity:
        ax.set_xlabel('U_inc/sqrt(gd) [-]')
    else:
        ax.set_xlabel('U_inc [m/s]')
    
    C_im, C_D = model_run_wet.test_im_D_concentration(Usal_range)
    fig,ax = plt.subplots()
    ax.plot(Usal_range/velocity_scale,C_im,label='C_im')
    ax.plot(Usal_range/velocity_scale,C_D,label='C_D')
    ax.plot(Usal_range/velocity_scale,C_im+C_D,label='C_tot')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid('on')
    if scaled_velocity:
        ax.set_xlabel('U_sal/sqrt(gd) [-]')
    else:
        ax.set_xlabel('U_sal [m/s]')
    
    fig,ax = plt.subplots()
    ax.plot(Usal_range/velocity_scale,model_run_wet.calc_T_jump_ballistic_assumption(Usal_range),label='Tjump (ballistic)')
    ax.plot(Usal_range/velocity_scale,model_run_wet.calc_T_jump_Xiuqi(Usal_range),label='Tjump (Xiuqi)')
    ax.plot(Usal_range/velocity_scale,model_run_wet.calc_T_jump_Test(Usal_range),label='Tjump (Test)')
    ax.plot(Usal_range/velocity_scale,model_run_wet.calc_T_jump(Usal_range),'k--',label='Tjump (used in model)')
    ax.plot(Usal_range/velocity_scale,model_run_wet.calc_T_jump_dep(Usal_range),'--',color=[0.5,0.5,0.5],label='Tjump Dep (used in wet model)')
    ax.plot(Usal_range/velocity_scale,model_run_dry.calc_T_jump_dep(Usal_range),':',color=[0.5,0.5,0.5],label='Tjump Dep (used in dry model)')
    ax.legend()
    ax.grid('on')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    if scaled_velocity:
        ax.set_xlabel('U_sal/sqrt(gd) [-]')
    else:
        ax.set_xlabel('U_sal [m/s]')
