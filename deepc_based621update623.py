import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.integrate import odeint
from scipy.signal import max_len_seq
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
import time



try:
    # Windows系统使用微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        # macOS系统使用苹方
        plt.rcParams['font.sans-serif'] = ['PingFang HK']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 通用解决方案
        chinese_font = FontProperties(fname='simhei.ttf')  # 需要下载字体文件
        plt.rcParams['font.family'] = chinese_font.get_name()


class Quadcopter:
    """改进的四旋翼无人机动力学模型"""
    def __init__(self):
        # 物理参数 (Crazyflie 2.0)
        self.m = 0.028  # 质量 (kg)
        self.g = 9.81   # 重力加速度 (m/s²)
        self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])  # 惯性矩 (kg·m²)
        self.dx = 0.1   # 螺旋桨x方向位置 (m)
        self.dy = 0.1   # 螺旋桨y方向位置 (m)
        self.dq = 0.01  # 扭矩系数 (N·m/N)
        self.J_inv = np.linalg.inv(self.J + np.eye(3)*1e-10)  # 避免奇异性
        self.angle_limit = np.pi/5  # 角度限制 (60度)
        
    def rotation_matrix(self, rpy):
        """欧拉角转旋转矩阵 (ZYX顺序)"""
        gamma, beta, alpha = rpy
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                       [np.sin(alpha), np.cos(alpha), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                       [0, 1, 0],
                       [-np.sin(beta), 0, np.cos(beta)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(gamma), -np.sin(gamma)],
                       [0, np.sin(gamma), np.cos(gamma)]])
        return Rz @ Ry @ Rx
    
    def dynamics(self, state, t, u):
        """改进的动力学方程，增加数值稳定性"""
        # 解包状态
        p = state[0:3]    # 位置 [x, y, z]
        v = state[3:6]    # 速度 [vx, vy, vz]
        rpy = state[6:9]  # 欧拉角 [γ, β, α] (roll, pitch, yaw)
        omega = state[9:12] # 角速度 [ωx, ωy, ωz]
        
        ftot, omega_ref_x, omega_ref_y = u
        
        # 1. 角速度控制 (简化内环模型)
        k_omega = 0.05  # 角速度跟踪增益
        tau_control = np.array([k_omega * (omega_ref_x - omega[0]), 
                                k_omega * (omega_ref_y - omega[1]), 
                                0])
        
        # # 物理扭矩
        # tau_physical = np.array([
        #     ftot * self.dy,
        #     -ftot * self.dx,
        #     self.dq * ftot
        # ]) - np.cross(omega, self.J @ omega)

         # 修正：确保控制输入实际影响系统
        # 物理扭矩计算 (修正符号和方向)
        # tau_physical = np.array([
        #     ftot * self.dy * np.sign(omega_ref_x),  # 修正：考虑方向
        #     -ftot * self.dx * np.sign(omega_ref_y), 
        #     self.dq * ftot
        # ]) - np.cross(omega, self.J @ omega)

        # 物理扭矩计算 (修正符号问题)
        tau_physical = np.array([
            0.5 * ftot * omega_ref_x,  # 简化力矩计算
            0.5 * ftot * omega_ref_y,
            self.dq * ftot
        ]) - np.cross(omega, self.J @ omega)
        
        # 总扭矩
        tau = tau_physical + tau_control

        # 2. 位置动力学
        R = self.rotation_matrix(rpy)
        
        # 限制推力范围避免过大的加速度

        # 确保重力项正确应用
        gravity_vec = np.array([0, 0, -self.g])  # 确保重力向下

        safe_ftot = np.clip(ftot, 0.1, 0.5)

        thrust_vec = R @ np.array([0, 0, safe_ftot])

        acceleration = thrust_vec / self.m + gravity_vec 

        # 添加空气阻力模型（防止无限上升）
        drag_coeff = 0.02  # 空气阻力系数
        velocity = state[3:6]
        drag_force = -drag_coeff * np.abs(velocity) * velocity
        acceleration += drag_force / self.m
        
        # 3. 欧拉角微分方程
        gamma, beta, alpha = rpy
        
        # 避免奇点 (当cos(beta)接近0时)
        cos_beta = np.cos(beta)
        if np.abs(cos_beta) < 1e-5:
            cos_beta = 1e-5 * np.sign(cos_beta) if cos_beta != 0 else 1e-5
        
        T = np.array([
            [1, np.sin(gamma)*np.tan(beta), np.cos(gamma)*np.tan(beta)],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma)/cos_beta, np.cos(gamma)/cos_beta]
        ])
        rpy_dot = T @ omega
        
        # 4. 角加速度
        omega_dot = self.J_inv @ tau

        # 添加详细诊断
        # if state[2] > 2.0:
        #     print(f"高度诊断 [t={t:.2f}s]: "
        #           f"Z={state[2]:.2f}m, Vz={state[5]:.2f}m/s, "
        #           f"推力={safe_ftot:.3f}, 加速度={acceleration[2]:.2f}m/s²")
        
        # 状态导数 - 添加小量限制变化率
        return np.concatenate([
            v, 
            np.clip(acceleration, -15, 15),
            np.clip(rpy_dot, -3, 3),
            np.clip(omega_dot, -15, 15)
        ])
    
    def step(self, state, u, dt=0.02):
        """带有错误处理的离散时间步进"""
        from scipy.integrate import solve_ivp

        try:
            # 限制输入
            u = np.clip(u, [0.1, -0.5, -0.5], [0.5, 0.5, 0.5])
            
            # 使用欧拉积分（更稳定）
            state_dot = self.dynamics(state, 0, u)
            new_state = state + state_dot * dt
            
            # 限制欧拉角在合理范围内
            new_state[6:9] = np.clip(new_state[6:9], -self.angle_limit, self.angle_limit)
            
            # 限制角速度
            new_state[9:12] = np.clip(new_state[9:12], -8, 8)
            
            # 限制位置和速度
            new_state[0:3] = np.clip(new_state[0:3], [-5, -5, 0], [5, 5, 4])
            new_state[3:6] = np.clip(new_state[3:6], [-3, -3, -3], [3, 3, 3])
            
            return new_state
            
        except Exception as e:
            print(f"积分错误: {str(e)}")
            return state


class DeePC:
    """改进的Data-Enabled Predictive Controller"""
    def __init__(self, Td=80, Tini=3, Tf=8, 
                 lambda_g=1.0, lambda_s=100.0, p=3):
        # 优化后的参数 - 更保守的设置
        self.Td = Td      
        self.Tini = Tini  
        self.Tf = Tf      
        self.lambda_g = lambda_g  # 大幅降低正则化
        self.lambda_s = lambda_s  # 降低正则化
        self.p = p        
        self.H = np.empty((0, 0))
        self.H_std = None
        
        self.m = 3        # 输入维度
        self.N = Tf       
        
        # 放宽约束参数 - 关键修改
        self.u_min = np.array([0.15, -0.4, -0.4])  
        self.u_max = np.array([0.45, 0.4, 0.4])
        self.y_min = np.array([-3, -3, 0.2])  
        self.y_max = np.array([3, 3, 3.5])

        # 平衡的成本矩阵权重
        self.Q = np.diag([10, 10, 15])   # 位置跟踪权重
        self.R = np.diag([1, 0.1, 0.1]) # 控制权重
        
        # 添加松弛变量权重
        self.slack_weight = 1000.0
    

    def build_hankel(self, ud, yd):
        """构建Hankel矩阵 - 改进版本"""
        L = self.Tini + self.Tf
        cols = self.Td - L + 1
        
        if cols <= 0:
            raise ValueError(f"数据长度不足: Td={self.Td}, L={L}")
        
        # 输入Hankel矩阵
        Hu = np.zeros((L * self.m, cols))
        for i in range(self.m):
            for j in range(cols):
                Hu[i*L:(i+1)*L, j] = ud[j:j+L, i]
        
        # 输出Hankel矩阵
        Hy = np.zeros((L * self.p, cols))
        for i in range(self.p):
            for j in range(cols):
                Hy[i*L:(i+1)*L, j] = yd[j:j+L, i]
        
        H = np.vstack([Hu, Hy])
        
        # 简单的数据标准化
        H_std = np.std(H, axis=1, keepdims=True)
        H_std[H_std < 1e-6] = 1.0
        
        return H, H_std
    
    def emergency_recovery(self, current_state, target_pos):
        """紧急恢复控制器"""
        # 高度优先恢复策略
        Kp_z = 0.5  # 高度比例增益
        Kp_xy = 0.2  # 水平位置比例增益
        
        # 计算位置误差
        error = target_pos - current_state[0:3]
        
        # 高度控制（优先级最高）
        if current_state[2] > target_pos[2] + 1.0:  # 过高
            ftot = 0.2747 - min(0.1, Kp_z * abs(error[2]))
        elif current_state[2] < target_pos[2] - 0.5:  # 过低
            ftot = 0.2747 + min(0.1, Kp_z * abs(error[2]))
        else:  # 高度可接受
            ftot = 0.2747
        
        # 水平位置控制
        omega_x = np.clip(Kp_xy * error[0], -0.2, 0.2)
        omega_y = np.clip(Kp_xy * error[1], -0.2, 0.2)
        
        # 构建控制输入
        u_opt = np.array([ftot, omega_x, omega_y])
        
        # 确保在约束范围内
        u_opt = np.clip(u_opt, self.u_min, self.u_max)
        
        return u_opt, np.zeros((self.Tf, 3)), 0.0
    

    def solve(self, H, uini, yini, ref_traj, current_state):
        """改进的求解函数 - 更鲁棒的优化"""
        try:
            if H.size == 0:
                print("Hankel矩阵为空!")
                return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), 0.0
            
            # 分解Hankel矩阵
            n_u = self.m
            n_y = self.p
            Up = H[:self.Tini*n_u, :]
            Yp = H[self.Tini*n_u:(self.Tini*n_u + self.Tini*n_y), :]
            Uf = H[(self.Tini*n_u + self.Tini*n_y):(self.Tini*n_u + self.Tini*n_y + self.Tf*n_u), :]
            Yf = H[(self.Tini*n_u + self.Tini*n_y + self.Tf*n_u):, :]
            
            if Up.shape[1] == 0 or Yp.shape[1] == 0:
                print("分解后的矩阵维度错误!")
                return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), 0.0
            
            # 优化变量
            g = cp.Variable(H.shape[1])
            u = cp.Variable((self.Tf, n_u))
            y = cp.Variable((self.Tf, n_y))
            
            # 松弛变量
            sigma_y = cp.Variable(self.Tini * n_y)
            slack_u = cp.Variable((self.Tf, n_u), nonneg=True)
            slack_y = cp.Variable((self.Tf, n_y), nonneg=True)
            
            # 基础约束
            constraints = [
                Up @ g == uini.flatten(),
                Yp @ g == yini.flatten() + sigma_y,
                Uf @ g == u.flatten(),
                Yf @ g == y.flatten()
            ]
            
            # 软约束 - 输入约束
            for i in range(self.Tf):
                constraints += [
                    u[i, 0] >= self.u_min[0] - slack_u[i, 0],
                    u[i, 0] <= self.u_max[0] + slack_u[i, 0],
                    u[i, 1] >= self.u_min[1] - slack_u[i, 1],
                    u[i, 1] <= self.u_max[1] + slack_u[i, 1],
                    u[i, 2] >= self.u_min[2] - slack_u[i, 2],
                    u[i, 2] <= self.u_max[2] + slack_u[i, 2]
                ]
            
            # 软约束 - 输出约束
            for i in range(self.Tf):
                constraints += [
                    y[i, 0] >= self.y_min[0] - slack_y[i, 0],
                    y[i, 0] <= self.y_max[0] + slack_y[i, 0],
                    y[i, 1] >= self.y_min[1] - slack_y[i, 1],
                    y[i, 1] <= self.y_max[1] + slack_y[i, 1],
                    y[i, 2] >= self.y_min[2] - slack_y[i, 2],
                    y[i, 2] <= self.y_max[2] + slack_y[i, 2]
                ]
            
            # 目标函数
            cost = 0
            
            # 跟踪误差
            for k in range(self.Tf):
                cost += cp.quad_form(y[k, :3] - ref_traj[k], self.Q)
                cost += cp.quad_form(u[k], self.R)
            
            # 正则化项
            cost += self.lambda_s * cp.sum_squares(sigma_y)
            cost += self.lambda_g * cp.sum_squares(g)
            
            # 松弛变量惩罚
            cost += self.slack_weight * cp.sum(slack_u)
            cost += self.slack_weight * cp.sum(slack_y)
            
            # 平滑性约束
            for k in range(self.Tf-1):
                cost += 0.1 * cp.sum_squares(u[k+1] - u[k])
            
            # 创建问题
            problem = cp.Problem(cp.Minimize(cost), constraints)
            
            # 求解器选择
            solvers_to_try = [
                ('MOSEK', {'solver': cp.MOSEK, 'verbose': False, 'max_iters': 500}),
                ('ECOS', {'solver': cp.ECOS, 'verbose': False, 'max_iters': 500}),
                ('OSQP', {'solver': cp.OSQP, 'verbose': False, 'max_iter': 1000, 
                         'eps_abs': 1e-3, 'eps_rel': 1e-3}),
                ('SCS', {'solver': cp.SCS, 'verbose': False, 'max_iters': 1000, 'eps': 1e-3}),
            ]
            
            start_time = time.time()
            
            for solver_name, solver_params in solvers_to_try:
                if solver_name not in cp.installed_solvers():
                    continue
                    
                try:
                    result = problem.solve(**solver_params)
                    
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solve_time = time.time() - start_time
                        
                        if u.value is not None and y.value is not None:
                            u_opt = np.clip(u.value[0], self.u_min, self.u_max)
                            return u_opt, y.value, solve_time
                        
                except Exception as e:
                    continue
            
            # 后备PID控制器
            print("优化失败，使用PID后备控制")
            return self.pid_fallback(current_state, ref_traj[0])
            
        except Exception as e:
            print(f"求解过程异常: {str(e)}")
            return self.pid_fallback(current_state, ref_traj[0])
        
    def pid_fallback(self, current_state, target_pos):
        """PID后备控制器"""
        # PID参数
        Kp = np.array([0.8, 0.8, 1.2])  # 位置比例增益
        Kd = np.array([0.3, 0.3, 0.5])  # 位置微分增益
        
        # 位置误差
        pos_error = target_pos - current_state[0:3]
        vel_error = -current_state[3:6]  # 期望速度为0
        
        # 控制输出
        hover_thrust = 0.2747
        u_pid = np.array([
            hover_thrust + Kp[2] * pos_error[2] + Kd[2] * vel_error[2],
            Kp[0] * pos_error[0] + Kd[0] * vel_error[0],
            Kp[1] * pos_error[1] + Kd[1] * vel_error[1]
        ])
        
        # 限制输出
        u_pid = np.clip(u_pid, self.u_min, self.u_max)
        
        return u_pid, np.zeros((self.Tf, 3)), 0.0
        

def prbs_signal(n, bits=5):
    """生成PRBS激励信号 (论文3.1节)"""
    mls = max_len_seq(bits, length=n)[0]
    return 0.3 *(2*mls - 1)  # 转换为[-1, 1]

def generate_figure8(t, x_scale=1.0, y_scale=0.6, z_height=1.5, period=8.0):
    """生成8字形参考轨迹"""
    omega = 2 * np.pi / period
    
    x = x_scale * np.sin(omega * t)
    y = y_scale * np.sin(2 * omega * t)
    z = z_height * np.ones_like(t)
    
    return np.vstack([x, y, z]).T


# 主仿真函数
def deepc_simulation():

    # ====================== 可调参数配置区 ======================
    # 参数组选择 (取消注释需要测试的组)
    # param_group = 0  # 基准参数
    # param_group = 1  # 基准参数
    # param_group = 2  # 加强位置跟踪
    # param_group = 3  # 抑制震荡
    # param_group = 4  # z轴精密控制
    # param_group = 5  # 平衡响应速度和平稳性
    # param_group = 6  # 误差控制在1-2米范围内
    # param_group = 7 # 缩短Td和Tf减少计算复杂度
    param_group = 8


    
    # 参数组定义
    param_sets = {
        # 0: {'Td': 120, 'Tini': 3, 'Tf': 8, 'lambda_g': 10, 'lambda_s': 1e4, 'p': 3,
        #     'Q_diag': [100, 100, 100], 'R_diag': [20, 1, 1]},

        # 1: {'Td': 150, 'Tini': 4, 'Tf': 12, 'lambda_g': 25, 'lambda_s': 5e5, 'p': 3,
        #     'Q_diag': [25, 25, 25], 'R_diag': [80, 2, 2]},

        # 2: {'Td': 180, 'Tini': 5, 'Tf': 15, 'lambda_g': 30, 'lambda_s': 8e5, 'p': 3,
        #     'Q_diag': [40, 40, 40], 'R_diag': [60, 5, 5]},
        # 3: {'Td': 200, 'Tini': 6, 'Tf': 20, 'lambda_g': 80, 'lambda_s': 3e5, 'p': 3,
        #     'Q_diag': [30, 30, 30], 'R_diag': [100, 8, 8]},
        # 4: {'Td': 170, 'Tini': 5, 'Tf': 18, 'lambda_g': 40, 'lambda_s': 6e5, 'p': 3,
        #     'Q_diag': [20, 20, 50], 'R_diag': [70, 4, 4]},
        # 5: {'Td': 190, 'Tini': 5, 'Tf': 16, 'lambda_g': 50, 'lambda_s': 7e5, 'p': 3,
        #     'Q_diag': [35, 35, 35], 'R_diag': [90, 6, 6]},
        # 6: {'Td': 300, 'Tini': 5, 'Tf': 20, 'lambda_g': 100, 'lambda_s': 1e6, 'p': 3,
        #     'Q_diag': [50, 50, 100], 'R_diag': [50, 5, 5]},
        # 7: {'Td': 200, 'Tini': 5, 'Tf': 10, 'lambda_g': 500, 'lambda_s': 5e6, 'p': 3,
        #     'Q_diag': [80, 80, 150], 'R_diag': [30, 3, 3]},
        8: {'Td': 80, 'Tini': 3, 'Tf': 8, 'lambda_g': 0.1, 'lambda_s': 10, 'p': 3}
            # 'Q_diag': [80, 80, 150], 'R_diag': [30, 3, 3]},


        }
    
    # 应用参数
    params = param_sets[param_group]
    print(f"应用参数：\n{params}")
    print("=" * 60)
    Td = params['Td']
    Tini = params['Tini']
    Tf = params['Tf']
    lambda_g = params['lambda_g']
    lambda_s = params['lambda_s']
    p = params['p']
    # Q_diag = params['Q_diag']
    # R_diag = params['R_diag']
    # ========================================================


    # 初始化系统和控制器 - 使用优化参数
    quad = Quadcopter()
    # deepc = DeePC(p=3, Td=150, Tini=4, Tf=12, lambda_g=25, lambda_s=5e5)  # 进一步优化参数
    deepc = DeePC(p=p, Td=Td, Tini=Tini, Tf=Tf, lambda_g=lambda_g, lambda_s=lambda_s) #配置参数
    
    
    # 初始状态
    state = np.zeros(12)
    # state[0] = 0.0  # x = 0
    # state[1] = 0.0  # y = 0
    state[2] = 1.2  # z = 1.2 (匹配轨迹起点高度)
    # state[2] = 1.0

    # # 设置成本矩阵
    # deepc.Q = np.diag(Q_diag)
    # deepc.R = np.diag(R_diag)
    
    # 1. 改进的数据收集阶段
    print(" 开始数据收集...")
    T_data = deepc.Td
    ud = np.zeros((T_data, 3))
    yd = np.zeros((T_data, 3))

    # 使用更温和的PRBS激励信号
    prbs1 = prbs_signal(T_data, bits=4)
    prbs2 = prbs_signal(T_data, bits=4)
    prbs3 = prbs_signal(T_data, bits=4)

    u_hover = np.array([0.2747, 0, 0])

    for i in range(T_data):
        # 更温和的扰动
        u = u_hover.copy()
        u[0] += 0.03 * prbs1[i]   # 降低推力扰动
        u[1] += 0.1 * prbs2[i]    # 降低角速度扰动
        u[2] += 0.1 * prbs3[i]
        
        # 限制输入范围
        # u = np.clip(u, [0.15, -0.5, -0.5], [0.45, 0.5, 0.5])

        # 限制输入范围(621)
        u = np.clip(u, deepc.u_min, deepc.u_max)
        
        # 应用控制并更新状态
        state = quad.step(state, u)
        
        # 检查状态是否合理
        if np.any(np.abs(state[6:9]) > np.pi/3):  # 角度过大
            print(f"警告: 第{i}步角度过大，重置状态")
            state[6:9] = np.clip(state[6:9], -np.pi/4, np.pi/4)
        
        # 存储数据
        ud[i] = u
        yd[i] = state[0:3] #+ np.random.normal(0, 0.001, 3)  # 稍微增加噪声
        if i % 20 == 0:
            print(f"数据收集进度: {i}/{T_data}")
    
    # 2. 构建Hankel矩阵
    print("构建Hankel矩阵...")
    try:
        deepc.H, deepc.H_std = deepc.build_hankel(ud, yd)
        print(f"Hankel矩阵形状：{deepc.H.shape}")
    except Exception as e:
        print(f"Hankel矩阵构建失败: {e}")
        return None, None, None, None
    
    # 3. 轨迹跟踪
    sim_time = 12.0
    dt = 0.02
    steps = int(sim_time / dt)
    time_arr = np.arange(0, sim_time, dt)
    
    # 生成8字形参考轨迹
    ref_traj = generate_figure8(time_arr, x_scale=1.2, y_scale=0.8, z_height=1.5, period=10.0)
    
    # 重置状态到轨迹起点
    state = np.zeros(12)
    state[0:3] = ref_traj[0]
    
    # 历史数据存储
    state_history = np.zeros((steps, 12))
    u_history = np.zeros((steps, 3))
    solve_times = []
    
    # 初始条件窗口
    uini = np.tile(u_hover, (deepc.Tini, 1))
    yini = np.tile(state[0:3], (deepc.Tini, 1))
    
    print("开始主控制循环...")
    
    # 4. 主控制循环
    for i in range(steps):
        y_meas = state[0:3]
        
        if i < deepc.Tini:
            # 初始阶段使用PID控制
            u_opt = deepc.pid_fallback(state, ref_traj[i])[0]
            uini[i] = u_opt
            yini[i] = y_meas
        else:
            # 滚动更新历史窗口
            uini = np.roll(uini, -1, axis=0)
            yini = np.roll(yini, -1, axis=0)
            uini[-1] = u_history[i-1]
            yini[-1] = y_meas
            
            # 获取参考轨迹
            start_idx = i
            end_idx = min(i + deepc.Tf, len(ref_traj))
            current_ref = ref_traj[start_idx:end_idx]
            
            if len(current_ref) < deepc.Tf:
                # 扩展最后一个点
                padding = np.tile(ref_traj[-1], (deepc.Tf - len(current_ref), 1))
                current_ref = np.vstack([current_ref, padding])
            
            # DeePC求解
            u_opt, y_pred, solve_time = deepc.solve(deepc.H, uini, yini, current_ref, state)
            solve_times.append(solve_time)
        
        # 应用控制输入
        state = quad.step(state, u_opt)
        
        # 存储数据
        state_history[i] = state
        u_history[i] = u_opt
        
        # 进度打印
        if i % 50 == 0:
            pos_error = np.linalg.norm(state[0:3] - ref_traj[i])
            print(f"步骤 {i}/{steps}, 位置误差: {pos_error:.3f}m")
    
    print("仿真完成!")
    
    # 性能统计
    position_errors = [np.linalg.norm(state_history[i, 0:3] - ref_traj[i]) for i in range(steps)]
    
    print(f"\n性能统计:")
    print(f"平均位置误差: {np.mean(position_errors):.4f} m")
    print(f"最大位置误差: {np.max(position_errors):.4f} m")
    print(f"最终位置误差: {position_errors[-1]:.4f} m")
    
    if solve_times:
        print(f"平均求解时间: {np.mean(solve_times)*1000:.2f} ms")
    
    return time_arr, state_history, ref_traj, np.array(solve_times)

    
   

def plot_results(time_arr, state_history, ref_traj, solve_times):
    """绘制仿真结果"""
    if time_arr is None or state_history is None:
        print("没有数据可绘制")
        return
        
    # 提取位置数据
    pos = state_history[:, 0:3]
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 1. 3D轨迹
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='实际轨迹')
    ax1.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 'r--', linewidth=2, label='参考轨迹')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹跟踪')
    ax1.legend()
    ax1.grid(True)
    
    # 2. XY轨迹
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, label='实际轨迹')
    ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', linewidth=2, label='参考轨迹')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY轨迹')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)
    
    # 3. X位置跟踪
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(time_arr, pos[:, 0], 'b-', linewidth=2, label='实际X')
    ax3.plot(time_arr, ref_traj[:, 0], 'r--', linewidth=2, label='参考X')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('X位置 (m)')
    ax3.set_title('X位置跟踪')
    ax3.legend()
    ax3.grid(True)
    
    # 4. Y位置跟踪
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(time_arr, pos[:, 1], 'g-', linewidth=2, label='实际Y')
    ax4.plot(time_arr, ref_traj[:, 1], 'm--', linewidth=2, label='参考Y')
    ax4.set_xlabel('时间 (s)')
    ax4.set_ylabel('Y位置 (m)')
    ax4.set_title('Y位置跟踪')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Z位置跟踪
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(time_arr, pos[:, 2], 'c-', linewidth=2, label='实际Z')
    ax5.plot(time_arr, ref_traj[:, 2], 'k--', linewidth=2, label='参考Z')
    ax5.set_xlabel('时间 (s)')
    ax5.set_ylabel('Z位置 (m)')
    ax5.set_title('Z位置跟踪')
    ax5.legend()
    ax5.grid(True)
    
    # 6. 跟踪误差
    ax6 = plt.subplot(2, 3, 6)
    errors = [np.linalg.norm(pos[i] - ref_traj[i]) for i in range(len(time_arr))]
    ax6.plot(time_arr, errors, 'r-', linewidth=2)
    ax6.set_xlabel('时间 (s)')
    ax6.set_ylabel('位置误差 (m)')
    ax6.set_title('位置跟踪误差')
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()




import mosek
import os
os.environ['MOSEKLM_LICENSE_FILE'] = './mosek.lic'

def check_license():
    """检查MOSEK许可证状态"""
    try:
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                # 尝试求解一个简单问题
                task.appendvars(1)
                task.putvarbound(0, mosek.boundkey.fr, -1, 1)
                task.putcj(0, 1)
                task.putobjsense(mosek.objsense.minimize)
                task.optimize()
                print("✅ MOSEK许可证验证成功！")
                # print(f"许可证版本: {task.getversion()}")
    except mosek.Error as e:
        print(f"❌ 许可证验证失败: {e}")
        print("可能原因：")
        print("1. 许可证文件路径错误")
        print("2. 许可证已过期")
        print("3. 网络许可服务器不可达")



# 主函数
def main():
    print("启动DeePC轨迹跟踪仿真...")
    time_arr, state_history, ref_traj, solve_times = deepc_simulation()
    
    if time_arr is not None:
        print("绘制结果...")
        plot_results(time_arr, state_history, ref_traj, solve_times)
    else:
        print("仿真失败!")

if __name__ == "__main__":
    main()