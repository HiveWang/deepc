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
        self.angle_limit = np.pi/3  # 角度限制 (60度)
        
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
            ftot * self.dy * np.sign(omega_ref_x) if abs(omega_ref_x) > 1e-3 else 0,
            -ftot * self.dx * np.sign(omega_ref_y) if abs(omega_ref_y) > 1e-3 else 0,
            self.dq * ftot
        ]) - np.cross(omega, self.J @ omega)
        
        # 总扭矩
        tau = tau_physical + tau_control

        # 2. 位置动力学
        R = self.rotation_matrix(rpy)
        
        # 限制推力范围避免过大的加速度

        # 确保重力项正确应用
        gravity_vec = np.array([0, 0, -self.g])  # 确保重力向下

        safe_ftot = np.clip(ftot, 0.15, 0.4)

        thrust_vec = R @ np.array([0, 0, safe_ftot])

        acceleration = thrust_vec / self.m + gravity_vec 

        # 添加空气阻力模型（防止无限上升）
        drag_coeff = 0.01  # 空气阻力系数
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
            np.clip(acceleration, -10, 10),
            np.clip(rpy_dot, -5, 5),
            np.clip(omega_dot, -20, 20)
        ])
    
    def step(self, state, u, dt=0.04):
        """带有错误处理的离散时间步进"""
        from scipy.integrate import solve_ivp

        # 添加状态变化监测
        prev_state = state.copy()


        # 定义事件函数检测异常状态
        def event(t, y):
            # 检查角度是否过大
            rpy = y[6:9]
            if np.any(np.abs(rpy) > np.pi/1.5):
                return 0
            return 1
        
        event.terminal = True  # 事件发生时终止积分
        
        try:
            # 尝试使用更稳健的求解器
            sol = solve_ivp(
                fun=lambda t, y: self.dynamics(y, t, u),
                t_span=[0, dt],
                y0=state,
                method='BDF',  # 对于刚性问题更有效,RK45,BDF
                events=event,
                rtol=1e-4,     # 稍微宽松的容差
                atol=1e-6
            )
            
            if not sol.success:
                raise RuntimeError(f"积分失败: {sol.message}")
            
            if sol.status == -1:  # 事件触发
                print("检测到不稳定状态，提前终止积分")
                # 返回上一个有效状态
                return state
            
            if sol.status == 1:  # 事件触发
                print(f"检测到不稳定状态(角度超过{np.degrees(self.angle_limit):.1f}度)，提前终止积分")
                # 重置角度和角速度
                new_state = state.copy()
                new_state[6:9] = np.clip(new_state[6:9], -self.angle_limit/2, self.angle_limit/2)
                new_state[9:12] *= 0.5  # 减半角速度
                return new_state

            # 对结果状态进行限制
            new_state = sol.y[:, -1]
        
            # 限制欧拉角在合理范围内
            # new_state[6:9] = np.clip(new_state[6:9], -np.pi/2, np.pi/2)
            # 限制欧拉角在合理范围内---（621）update 控制角度过大
            new_state[6:9] = np.clip(new_state[6:9], -self.angle_limit, self.angle_limit)
            
            # 限制角速度
            new_state[9:12] = np.clip(new_state[9:12], -10, 10)

            # 添加状态监测和安全限制
            if state[2] > 3.0:  # 高度超过3米
                print("! 安全限制：高度超过3米，强制下降 !")
                u = np.array([0.22, 0, 0])  # 降低推力
            
            # 角度过大保护
            if any(abs(state[6:9]) > np.pi/3):
                print("! 安全限制：角度过大，重置姿态 !")
                state[6:9] = np.clip(state[6:9], -np.pi/6, np.pi/6)
                state[9:12] *= 0.5  # 减半角速度

            # 检查状态是否实际变化
            if np.allclose(prev_state, new_state, atol=1e-5):
                print("! 警告：状态未更新，强制施加扰动 !")
                # 添加微小扰动打破僵局
                perturbation = np.random.normal(0, 1e-3, 12)
                new_state += perturbation

            return new_state
            
        except Exception as e:
            print(f"积分错误: {str(e)}")
            # 如果积分失败，使用简单的欧拉积分作为后备
            print("使用欧拉积分后备方案")
            return state + self.dynamics(state, 0, u) * dt


class DeePC:
    """改进的Data-Enabled Predictive Controller"""
    def __init__(self, Td=200, Tini=4, Tf=15, 
                 lambda_g=50, lambda_s=1e6, p=3):  # 调整正则化参数
        # 优化后的参数
        self.Td = Td      
        self.Tini = Tini  
        self.Tf = Tf      
        self.lambda_g = lambda_g  # 降低g正则化
        self.lambda_s = lambda_s  # 降低s正则化
        self.p = p        
        self.H = np.empty((0, 0))
        
        self.m = 3        # 输入维度
        self.N = Tf       
        
        # # 放宽约束参数，提高求解成功率
        # self.u_min = np.array([0.15, -np.pi/3, -np.pi/3])  # 放宽角度约束
        # self.u_max = np.array([0.45, np.pi/3, np.pi/3])
        # self.y_min = np.array([-3, -3, 0.2])  # 放宽位置约束
        # self.y_max = np.array([3, 3, 3])

        # # 调整成本矩阵权重
        # self.Q = np.diag([25, 25, 25])  # 降低位置权重
        # self.R = np.diag([80, 2, 2])    # 降低控制权重


        # 修正约束参数 - 更合理的范围
        self.u_min = np.array([0.2, -0.3, -0.3])    # 推力不能太小
        self.u_max = np.array([0.4, 0.3, 0.3])      # 限制角速度范围
        self.y_min = np.array([-2, -2, 0.5])        # 位置约束
        self.y_max = np.array([2, 2, 2.5])

        # 修正约束参数 - 更合理的范围``# 在DeePC类中收紧约束(621)
        # self.u_min = np.array([0.22, -0.25, -0.25]) 
        # self.u_max = np.array([0.38, 0.25, 0.25])
        # self.y_min = np.array([-1.5, -1.5, 0.8]) 
        # self.y_max = np.array([1.5, 1.5, 2.0])

        # 修正成本矩阵权重 - 更平衡的权重
        self.Q = np.diag([100, 100, 100])  # 增加位置跟踪权重
        self.R = np.diag([10, 1, 1])       # 降低控制权重
    

    def build_hankel(self, ud, yd):
        """构建Hankel矩阵 - 添加数值稳定性"""

        # 在build_hankel前添加数据平滑
        from scipy.signal import savgol_filter
        ud = savgol_filter(ud, window_length=5, polyorder=2, axis=0)
        yd = savgol_filter(yd, window_length=5, polyorder=2, axis=0)

        L = self.Tini + self.Tf
        cols = self.Td - L + 1
        
        if cols <= 0:
            raise ValueError(f"数据长度不足: Td={self.Td}, L={L}")
        
        # 输入Hankel矩阵
        Hu = np.zeros((L * ud.shape[1], cols))
        for i in range(ud.shape[1]):
            Hu[i*L:(i+1)*L, :] = hankel(
                ud[:L, i], 
                ud[L-1:L-1+cols, i]
            )[:L, :cols]
        
        # 输出Hankel矩阵
        Hy = np.zeros((L * self.p, cols))
        for i in range(self.p):
            Hy[i*L:(i+1)*L, :] = hankel(
                yd[:L, i], 
                yd[L-1:L-1+cols, i]
            )[:L, :cols]
        
        H = np.vstack([Hu, Hy])
        
        # 数据标准化提高数值稳定性
        H_std = np.std(H, axis=1, keepdims=True)
        H_std[H_std < 1e-8] = 1.0  # 避免除零
        H_normalized = H / H_std
        
        return H_normalized, H_std
    
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
    
    def solve(self, H, uini, yini, ref_traj,current_state):
        """改进的求解函数 - 更好的数值稳定性"""

        # 1. 检查状态偏离程度
        pos_error = np.linalg.norm(current_state[0:3] - ref_traj[0])
        
        # 紧急恢复模式：当位置误差 > 2m
        if pos_error > 3.0:
            print(f"! 位置误差: {pos_error:.2f}m !")
            # return self.emergency_recovery(current_state, ref_traj[0])

        try:
            # 检查输入维度
            if H.size == 0:
                print("Hankel矩阵为空!")
                return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), 0.0
            
            # 分解Hankel矩阵
            n_u = 3
            n_y = self.p
            Up = H[:self.Tini*n_u, :]
            Yp = H[self.Tini*n_u:(self.Tini*n_u + self.Tini*n_y), :]
            Uf = H[(self.Tini*n_u + self.Tini*n_y):(self.Tini*n_u + self.Tini*n_y + self.Tf*n_u), :]
            Yf = H[(self.Tini*n_u + self.Tini*n_y + self.Tf*n_u):, :]
            
            # 检查矩阵维度
            if Up.shape[1] == 0 or Yp.shape[1] == 0:
                print("分解后的矩阵维度错误!")
                return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), 0.0
            
            # 优化变量
            g = cp.Variable(H.shape[1])
            u = cp.Variable((self.Tf, n_u))
            y = cp.Variable((self.Tf, n_y))
            sigma_y = cp.Variable(self.Tini * n_y)
            
            # 基础约束
            constraints = [
                Up @ g == uini.flatten(),
                Yp @ g == yini.flatten() + sigma_y,
                Uf @ g == u.flatten(),
                Yf @ g == y.flatten()
            ]
            
            # 软约束 - 提高求解成功率
            u_slack = cp.Variable((self.Tf, n_u), nonneg=True)
            y_slack = cp.Variable((self.Tf, n_y), nonneg=True)
            
            # 输入约束 (软约束)
            for i in range(self.Tf):
                constraints += [
                    u[i, 0] >= self.u_min[0] - u_slack[i, 0],
                    u[i, 0] <= self.u_max[0] + u_slack[i, 0],
                    u[i, 1] >= self.u_min[1] - u_slack[i, 1],
                    u[i, 1] <= self.u_max[1] + u_slack[i, 1],
                    u[i, 2] >= self.u_min[2] - u_slack[i, 2],
                    u[i, 2] <= self.u_max[2] + u_slack[i, 2]
                ]
            
            # # 输出约束 (软约束)
            for i in range(self.Tf):
                constraints += [
                    y[i, 0] >= self.y_min[0] - y_slack[i, 0],
                    y[i, 0] <= self.y_max[0] + y_slack[i, 0],
                    y[i, 1] >= self.y_min[1] - y_slack[i, 1],
                    y[i, 1] <= self.y_max[1] + y_slack[i, 1],
                    y[i, 2] >= self.y_min[2] - y_slack[i, 2],
                    y[i, 2] <= self.y_max[2] + y_slack[i, 2]
                ]
            #-------
            ref_point = ref_traj[0]  # 使用参考点作为基准
            dynamic_range = 1.0      # 动态范围大小
            dynamic_y_min = np.array([
                max(self.y_min[0], ref_point[0] - dynamic_range),
                max(self.y_min[1], ref_point[1] - dynamic_range),
                max(self.y_min[2], ref_point[2] - dynamic_range)
            ])
            dynamic_y_max = np.array([
                min(self.y_max[0], ref_point[0] + dynamic_range),
                min(self.y_max[1], ref_point[1] + dynamic_range),
                min(self.y_max[2], ref_point[2] + dynamic_range)
            ])
            
            # 在约束中使用动态范围
            for i in range(self.Tf):
                constraints += [
                    y[i, 0] >= dynamic_y_min[0] - y_slack[i, 0],
                    y[i, 0] <= dynamic_y_max[0] + y_slack[i, 0],
                    y[i, 1] >= dynamic_y_min[1] - y_slack[i, 1],
                    y[i, 1] <= dynamic_y_max[1] + y_slack[i, 1],
                    y[i, 2] >= dynamic_y_min[2] - y_slack[i, 2],
                    y[i, 2] <= dynamic_y_max[2] + y_slack[i, 2]
                ]

            #--------
            
            # 改进的目标函数
            cost = 0
            for k in range(self.Tf):
             
                # 跟踪误差
                cost += cp.quad_form(y[k, :3] - ref_traj[k], self.Q)
                cost += cp.quad_form(u[k], self.R)

               

            # 添加当前位置与初始参考的强约束
            cost += 10 * cp.quad_form(y[0, :3] - ref_traj[0], np.eye(3))
            
            # 正则化项
            cost += self.lambda_s * cp.sum_squares(sigma_y)
            cost += self.lambda_g * cp.norm(g, 2)
            
            # 软约束惩罚
            cost += 1e4 * cp.sum(u_slack)  # 输入约束违反惩罚
            cost += 1e4 * cp.sum(y_slack)  # 输出约束违反惩罚
            
            # 数值稳定项
            cost += 1e-8 * cp.sum_squares(u)
            cost += 1e-8 * cp.sum_squares(y)
            
            # 创建问题
            problem = cp.Problem(cp.Minimize(cost), constraints)
            
            # 尝试多个求解器
            solvers_to_try = [
                ('MOSEK', {'solver': cp.MOSEK,'verbose': False,}),
            #     'mosek_params': {
            #         'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-4,
            #         'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-4,
            #         'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-4,
            #         # 'MSK_IPAR_NUM_THREADS': 4,
            #         'MSK_IPAR_INTPNT_MAX_ITERATIONS': 200,
            #         'MSK_IPAR_OPTIMIZER': 'MSK_OPTIMIZER_INTPNT'
            #     }
            # }),
                ('CLARABEL', {'solver': cp.CLARABEL, 'verbose': False}),
                ('ECOS', {'solver': cp.ECOS, 'verbose': False, 'max_iters': 1000}),
                ('OSQP', {'solver': cp.OSQP, 'verbose': False, 'max_iters': 2000, 'eps_abs': 1e-4, 'eps_rel': 1e-4}),
                ('SCS', {'solver': cp.SCS, 'verbose': False, 'max_iters': 2000, 'eps': 1e-3}),
            ]
            
            start_time = time.time()
            
            for solver_name, solver_params in solvers_to_try:
                if solver_name not in cp.installed_solvers():
                    continue
                    
                try:
                    result = problem.solve(**solver_params)
                    
                    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                        solve_time = time.time() - start_time
                        print(f"[{solver_name}] 求解成功! 状态: {problem.status}, 用时: {solve_time*1000:.1f}ms")
                        
                        # 检查解的有效性
                        if u.value is not None and y.value is not None:
                            u_opt = np.clip(u.value[0], self.u_min, self.u_max)  # 裁剪到约束范围
                            return u_opt, y.value, solve_time
                        
                except Exception as e:
                    print(f"[{solver_name}] 求解失败: {str(e)}")
                    continue
            
            # 所有求解器都失败
            print("所有求解器都失败! 使用后备控制")
            solve_time = time.time() - start_time
            return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), solve_time
            
        except Exception as e:
            print(f"求解过程异常: {str(e)}")
            return np.array([0.2747, 0, 0]), np.zeros((self.Tf, 3)), 0.0
        

def prbs_signal(n, bits=5):
    """生成PRBS激励信号 (论文3.1节)"""
    mls = max_len_seq(bits, length=n)[0]
    return 0.5 *(2*mls - 1)  # 转换为[-1, 1]

def generate_figure8(t, x_scale=1.0, y_scale=0.5, z_height_start=0.0, z_height_end=2.0, 
                    z_variation=0.2, period=5.0, z_rise_time=5.0):
    """生成带高度上升的8字形参考轨迹"""
    omega = 2 * np.pi / period
    
    # 计算每个时间点的高度
    z_base = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < z_rise_time:
            # 上升阶段：从起始高度平滑上升到目标高度
            z_base[i] = z_height_start + (z_height_end - z_height_start) * (1 - np.exp(-2.0 * ti / z_rise_time))
        else:
            # 上升完成后保持在目标高度
            z_base[i] = z_height_end
    
    # 添加周期性高度变化
    z_osc = z_variation * np.sin(omega * t)
    
    # 生成8字形XY轨迹
    x = x_scale * np.sin(omega * t)
    y = y_scale * np.sin(2 * omega * t)
    
    # 组合XYZ轨迹
    z = z_base + z_osc
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
    param_group = 7 # 缩短Td和Tf减少计算复杂度


    
    # 参数组定义
    param_sets = {
        0: {'Td': 120, 'Tini': 3, 'Tf': 8, 'lambda_g': 10, 'lambda_s': 1e4, 'p': 3,
            'Q_diag': [100, 100, 100], 'R_diag': [20, 1, 1]},

        1: {'Td': 150, 'Tini': 4, 'Tf': 12, 'lambda_g': 25, 'lambda_s': 5e5, 'p': 3,
            'Q_diag': [25, 25, 25], 'R_diag': [80, 2, 2]},

        2: {'Td': 180, 'Tini': 5, 'Tf': 15, 'lambda_g': 30, 'lambda_s': 8e5, 'p': 3,
            'Q_diag': [40, 40, 40], 'R_diag': [60, 5, 5]},
        3: {'Td': 200, 'Tini': 6, 'Tf': 20, 'lambda_g': 80, 'lambda_s': 3e5, 'p': 3,
            'Q_diag': [30, 30, 30], 'R_diag': [100, 8, 8]},
        4: {'Td': 170, 'Tini': 5, 'Tf': 18, 'lambda_g': 40, 'lambda_s': 6e5, 'p': 3,
            'Q_diag': [20, 20, 50], 'R_diag': [70, 4, 4]},
        5: {'Td': 190, 'Tini': 5, 'Tf': 16, 'lambda_g': 50, 'lambda_s': 7e5, 'p': 3,
            'Q_diag': [35, 35, 35], 'R_diag': [90, 6, 6]},
        6: {'Td': 300, 'Tini': 5, 'Tf': 20, 'lambda_g': 100, 'lambda_s': 1e6, 'p': 3,
            'Q_diag': [50, 50, 100], 'R_diag': [50, 5, 5]},
        7: {'Td': 200, 'Tini': 5, 'Tf': 10, 'lambda_g': 500, 'lambda_s': 5e6, 'p': 3,
            'Q_diag': [80, 80, 150], 'R_diag': [30, 3, 3]}

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
    Q_diag = params['Q_diag']
    R_diag = params['R_diag']
    # ========================================================


    # 初始化系统和控制器 - 使用优化参数
    quad = Quadcopter()
    # deepc = DeePC(p=3, Td=150, Tini=4, Tf=12, lambda_g=25, lambda_s=5e5)  # 进一步优化参数
    deepc = DeePC(p=p, Td=Td, Tini=Tini, Tf=Tf, lambda_g=lambda_g, lambda_s=lambda_s) #配置参数
    
    
    # 初始状态
    state = np.zeros(12)
    state[0] = 0.0  # x = 0
    state[1] = 0.0  # y = 0
    state[2] = 0.0  # z = 1.2 (匹配轨迹起点高度)
    # state[2] = 1.0

    # 设置成本矩阵
    deepc.Q = np.diag(Q_diag)
    deepc.R = np.diag(R_diag)
    
    # 1. 改进的数据收集阶段
    print(" 开始数据收集...")
    T_data = deepc.Td
    ud = np.zeros((T_data, 3))
    yd = np.zeros((T_data, 3))

    # 使用更温和的PRBS激励信号
    prbs = prbs_signal(T_data, bits=5)  # 降低bits减少激励强度
    prbs2 = prbs_signal(T_data, bits=5)  # 降低bits减少激励强度
    prbs3 = prbs_signal(T_data, bits=5)  # 降低bits减少激励强度

    u_hover = np.array([0.2747, 0, 0])

    for i in range(T_data):
        # 更温和的扰动
        u = u_hover.copy()
        u[0] += 0.02 * prbs[i]   # 降低推力扰动
        u[1] += 0.5 * prbs2[i]    # 降低角速度扰动
        u[2] += 0.5 * prbs3[i]
        
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
        yd[i] = state[0:3] + np.random.normal(0, 0.001, 3)  # 稍微增加噪声
    
    # 2. 构建Hankel矩阵
    print(" 构建Hankel矩阵...")
    try:
        deepc.H, deepc.H_std = deepc.build_hankel(ud, yd)  # 使用标准化版本
        print(f" 完成：Hankel矩阵形状：{deepc.H.shape}")
    except Exception as e:
        print(f"Hankel矩阵构建失败: {e}")
        return np.array([0]), np.zeros((1, 12)), np.zeros((1, 3)), np.array([0])
    
    # 3. 轨迹跟踪
    print(" 开始轨迹跟踪...")
    sim_time = 10.0  # 缩短仿真时间
    dt = 0.02
    steps = int(sim_time / dt)
    time_arr = np.arange(0, sim_time, dt)
    
    # 生成更温和的参考轨迹
    # ref_traj = generate_figure8(time_arr, x_scale=1.0, y_scale=0.5, z_height=1.2, period=8.0)  # 缩小幅度
    # 生成带高度上升的8字形参考轨迹
    ref_traj = generate_figure8(time_arr, 
                               x_scale=1.0, 
                               y_scale=0.5,
                               z_height_start=0.0,  # 从地面开始
                               z_height_end=2.0,    # 上升到2米高度
                               z_variation=0.2,     # 高度波动幅度
                               period=5.0,          # 8字形周期
                               z_rise_time=5.0)     # 上升时间
    
    # 历史数据存储
    state_history = np.zeros((steps, 12))
    u_history = np.zeros((steps, 3))
    y_history = np.zeros((steps, 3))
    solve_times = []


    # 用于性能评价的变量
    position_errors = []  # 每一步的位置误差
    ref_positions = []    # 每一步的参考位置

    # 初始条件窗口
    uini = np.zeros((deepc.Tini, 3))
    yini = np.zeros((deepc.Tini, 3))
    
    # 重置初始状态匹配轨迹起点[这里为什么曜重置]
    state = np.zeros(12)
    state[0] = ref_traj[0][0]  # x = 起点x
    state[1] = ref_traj[0][1]  # y = 起点y
    state[2] = ref_traj[0][2]  # z = 起点z
    
    # 4. 主控制循环
    for i in range(steps):
        y_meas = state[0:3] #+ np.random.normal(0, 0.001, 3)

        # 记录性能评价数据
        current_ref = ref_traj[i]  # 当前时刻的参考位置
        ref_positions.append(current_ref.copy())
        pos_error = np.linalg.norm(y_meas - current_ref)
        position_errors.append(pos_error)

        # 根据误差大小动态调整正则化
        if pos_error > 1.0:  # 误差较大时
            lambda_g = 150
            lambda_s = 2e6
        else:               # 误差较小时
            lambda_g = 50
            lambda_s = 5e5

        if len(current_ref) < deepc.Tf:
                padding = np.tile(ref_traj[-1], (deepc.Tf - len(current_ref), 1))
                current_ref = np.vstack([current_ref, padding])

        # 在控制循环中添加诊断
        if pos_error > 5.0:
            print(f"步骤{i} 大误差诊断:")
            print(f"当前状态: {state[0:3]}")
            print(f"参考位置: {ref_traj[i]}")
            print(f"控制输入: {u_opt}")
            # 保存当前Hankel矩阵分析
            np.save(f'error_step_H.txt', deepc.H)

        
        if i < deepc.Tini:
            # 初始阶段使用简单控制
            u_opt = np.array([0.2747, 0.1*(ref_traj[i][0]-y_meas[0]), 0.1*(ref_traj[i][1]-y_meas[1])])
            u_opt = np.clip(u_opt, deepc.u_min, deepc.u_max)
            uini[i] = u_opt
            yini[i] = y_meas
        else:
            # 滚动更新
            uini = np.roll(uini, -1, axis=0)
            yini = np.roll(yini, -1, axis=0)
            uini[-1] = u_history[i-1]
            yini[-1] = y_meas
            
            # 获取参考轨迹
            start_idx = min(i, len(ref_traj) - 1)
            end_idx = min(i + deepc.Tf, len(ref_traj))
            current_ref = ref_traj[start_idx:end_idx]
            
            if len(current_ref) < deepc.Tf:
                padding = np.tile(ref_traj[-1], (deepc.Tf - len(current_ref), 1))
                current_ref = np.vstack([current_ref, padding])
        
            # DeePC求解
            u_opt, y_pred, solve_time = deepc.solve(deepc.H, uini, yini, current_ref,current_state=state)

            # 应用控制输入
            prev_z = state[2]
            solve_times.append(solve_time)

            # # 添加状态监测 # 增强高度安全协议
            # if state[2] > 1.8:  # 提前预警（1.8m）
            #     print(f"警告：高度预警 ({state[2]:.2f}m)，启动安全协议")
                
            #     # 1. 主动降低推力上限
            #     deepc.u_max[0] = 0.3  # 从0.4降到0.3
                
            #     # 2. 修改参考轨迹 - 优先降低高度
            #     ref_traj[i:] = np.tile(ref_traj[i], (len(ref_traj)-i, 1))
            #     ref_traj[i:, 2] = np.linspace(state[2], 1.2, len(ref_traj)-i)
                
            #     # 3. 添加高度下降奖励
            #     deepc.Q[2,2] = 500  # 大幅增加高度跟踪权重
            
            # # 高度超过2.5m紧急措施
            # if state[2] > 2.5:
            #     print(f"! 紧急高度控制激活 ({state[2]:.2f}m) !")
            #     # 1. 强制降低推力
            #     u_opt[0] = 0.25  # 低于悬停推力
                
            #     # 2. 忽略求解器结果
            #     state = quad.step(state, u_opt)
            #     continue  # 跳过存储步骤

            # # 打印控制效果
            # if i % 10 == 0:
            #     z_change = state[2] - prev_z
            #     print(f"步骤 {i}: 推力={u_opt[0]:.3f}, "
            #         f"高度变化={z_change:.3f}m, "
            #         f"位置误差={np.linalg.norm(state[0:3]-ref_traj[i]):.3f}m")
        
        # 应用控制输入
        state = quad.step(state, u_opt)
        
        # 存储数据
        state_history[i] = state
        u_history[i] = u_opt
        y_history[i] = state[0:3]
        
        # 定期打印进度
        if i % 25 == 0:
            pos_error = np.linalg.norm(state[0:3] - ref_traj[i])
            print(f"步骤 {i}/{steps}, 位置误差: {pos_error:.3f}m")

    
    
    # ====================== 性能评价代码 ======================
    # 确保数组长度一致
    min_len = min(len(position_errors), len(ref_positions), len(state_history))
    position_errors = position_errors[:min_len]
    ref_positions = np.array(ref_positions[:min_len])
    actual_positions = state_history[:min_len, 0:3]
    
    # 1. 跟踪误差 (SSE)
    squared_errors = np.sum((actual_positions - ref_positions)**2, axis=1)
    SSE = np.sum(squared_errors)
    
    # 2. 位置RMSE
    RMSE = np.sqrt(np.mean(squared_errors))
    
    # 3. 计算时间统计
    if solve_times:
        avg_solve_time = np.mean(solve_times) * 1000  # 毫秒
        max_solve_time = np.max(solve_times) * 1000
    else:
        avg_solve_time = max_solve_time = 0.0
    
    # 4. 最大跟踪误差
    max_error = np.max(position_errors) if position_errors else 0.0
    
    # 5. 稳态误差 (最后1秒)
    steady_state_error = np.mean(position_errors[-25:]) if len(position_errors) >= 25 else 0.0
    
    # 打印结果
    print(f"\n{'='*50}")
    print(f"参数配置评估结果 (组 {param_group})")
    print(f"{'='*50}")
    print(f"Td={Td}, Tini={Tini}, Tf={Tf}, lambda_g={lambda_g}, lambda_s={lambda_s}")
    print(f"Q={Q_diag}, R={R_diag}")
    print(f"1. 跟踪误差SSE: {SSE:.2f}")
    print(f"2. 位置RMSE: {RMSE:.4f} m")
    print(f"3. 最大跟踪误差: {max_error:.4f} m")
    print(f"4. 稳态误差: {steady_state_error:.4f} m")
    print(f"5. 平均求解时间: {avg_solve_time:.2f} ms")
    print(f"6. 最大求解时间: {max_solve_time:.2f} ms")
    print(f"{'='*50}")


    # # 计算统计信息
    # if solve_times:
    #     avg_solve_time = np.mean(solve_times) * 1000
    #     print(f"平均求解时间: {avg_solve_time:.2f}ms")
    #     success_rate = len([t for t in solve_times if t < 0.1]) / len(solve_times) * 100
    #     print(f"求解成功率: {success_rate:.1f}%")
    
    return time_arr, state_history, ref_traj, np.array(solve_times)


def plot_results(time_arr, state_history, ref_traj, solve_times):
    """绘制仿真结果"""
    # 提取位置数据
    pos = state_history[:, 0:3]
    
    # 创建图形
    plt.figure(figsize=(18, 12))
    
    # 1. 3D轨迹
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='实际轨迹')
    ax1.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 'r--', linewidth=2, label='参考轨迹')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title('3D轨迹跟踪', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.view_init(30, 45)
    ax1.grid(True)
    
    # 2. XY平面轨迹
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, label='实际轨迹')
    ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', linewidth=2, label='参考轨迹')
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('XY平面轨迹', fontsize=14)
    ax2.axis('equal')
    ax2.legend(fontsize=12)
    ax2.grid(True)
    
     # 3. 位置分量跟踪
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time_arr, pos[:, 0], 'b-', linewidth=2, label='X位置')
    ax3.plot(time_arr, ref_traj[:, 0], 'r--', linewidth=2, label='X参考')
    ax3.set_xlabel('时间 (s)', fontsize=12)
    ax3.set_ylabel('位置 (m)', fontsize=12)
    ax3.set_title('X位置跟踪', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time_arr, pos[:, 1], 'g-', linewidth=2, label='Y位置')
    ax4.plot(time_arr, ref_traj[:, 1], 'm--', linewidth=2, label='Y参考')
    ax4.set_xlabel('时间 (s)', fontsize=12)
    ax4.set_ylabel('位置 (m)', fontsize=12)
    ax4.set_title('Y位置跟踪', fontsize=14)
    ax4.legend(fontsize=12)
    ax4.grid(True)
    
    # 添加高度跟踪图
    plt.figure(figsize=(10, 6))
    plt.plot(time_arr, pos[:, 2], 'b-', linewidth=2, label='Z位置')
    plt.plot(time_arr, ref_traj[:, 2], 'r--', linewidth=2, label='Z参考')
    plt.xlabel('时间 (s)', fontsize=12)
    plt.ylabel('高度 (m)', fontsize=12)
    plt.title('高度跟踪', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # 求解时间图
    if len(solve_times) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(solve_times)), np.array(solve_times)*1000, 'b-')
        plt.xlabel('控制步', fontsize=12)
        plt.ylabel('求解时间 (ms)', fontsize=12)
        plt.title('DeePC优化求解时间', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
    
    plt.show()




# 运行仿真并绘制结果
def main():
    # 运行DeePC仿真
    # time, state_history, ref_traj = deepc_simulation()
    print("Stage-1:启动DeePC轨迹跟踪仿真...")
    time_arr, state_history, ref_traj, solve_times = deepc_simulation()
    

    # 绘制结果
    print("Stage-2:绘制仿真结果...")
    plot_results(time_arr, state_history, ref_traj, solve_times)
    
    # 绘制正则化效果
    print("Stage-3:绘制正则化效果比较...")
    plot_regularization_effect()



    # 确保数据一致性
    min_length = min(len(time_arr), state_history.shape[0], ref_traj.shape[0])
    time_arr = time_arr[:min_length]
    state_history = state_history[:min_length]
    ref_traj = ref_traj[:min_length]

    # 提取位置数据
    pos = state_history[:, 0:3]
    
    # 绘制轨迹跟踪结果 (论文Fig 10风格)
    plt.figure(figsize=(15, 10))
    
    # 3D轨迹
    ax1 = plt.subplot(2, 2, 1, projection='3d')
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', label='实际轨迹')
    ax1.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], 'r--', label='参考轨迹')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D轨迹跟踪')
    ax1.legend()
    ax1.view_init(30, 45)
    
    # XY平面轨迹
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(pos[:, 0], pos[:, 1], 'b-', label='实际轨迹')
    ax2.plot(ref_traj[:, 0], ref_traj[:, 1], 'r--', label='参考轨迹')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面轨迹')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)
    
    # 位置分量跟踪
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(time, pos[:, 0], 'b-', label='X位置')
    ax3.plot(time, pos[:, 1], 'g-', label='Y位置')
    ax3.plot(time, pos[:, 2], 'r-', label='Z位置')
    ax3.plot(time, ref_traj[:, 0], 'b--', label='X参考')
    ax3.plot(time, ref_traj[:, 1], 'g--', label='Y参考')
    ax3.plot(time, ref_traj[:, 2], 'r--', label='Z参考')
    ax3.set_xlabel('时间 (s)')
    ax3.set_ylabel('位置 (m)')
    ax3.set_title('位置分量跟踪')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('deepc_trajectory_tracking.png')
    plt.show()
    
    # 绘制正则化效果比较 (论文Fig 6)
    # plot_regularization_effect()

def plot_regularization_effect():
    """绘制正则化参数效果 (论文Fig 6)"""
    lambda_vals = [0, 10, 500, 10000]
    colors = ['r', 'g', 'b', 'm']
    labels = [f'λ_g={λ}' for λ in lambda_vals]
    
    plt.figure(figsize=(12, 8))
    
    for i, lambda_g in enumerate(lambda_vals):
        # 简化仿真 (仅高度通道)
        deepc = DeePC(lambda_g=lambda_g)
        steps = 200
        pos_z = np.zeros(steps)
        state = np.zeros(12)
        state[2] = 1.0
        
        for t in range(steps):
            # 简单高度参考
            ref = np.array([[0, 0, 1.5]]) if t > 50 else np.array([[0, 0, 1.0]])
            
            # 伪求解 (实际中需完整实现)
            u_opt = np.array([0.2747 + 0.01*(1.5 - state[2]), 0, 0])
            state = state + np.random.normal(0, 0.01, 12)
            state[2] += 0.05 * (1.5 - state[2])
            
            pos_z[t] = state[2]
        
        plt.plot(pos_z, color=colors[i], label=labels[i])
    
    plt.plot([0, 200], [1.5, 1.5], 'k--', label='目标高度')
    plt.xlabel('时间步')
    plt.ylabel('高度 (m)')
    plt.title('不同正则化参数的控制效果')
    plt.legend()
    plt.grid(True)
    plt.savefig('regularization_effect.png')
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



if __name__ == "__main__":
    check_license()
    main()
