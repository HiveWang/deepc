from matplotlib.font_manager import FontProperties

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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



# 定义轨迹生成函数
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

# 创建时间序列 (0到10秒，5000个点 - 高密度采样确保连续性)
t = np.linspace(0, 10, 5000)
traj = generate_figure8(t, z_rise_time=5.0)  # 使用明确的z_rise_time

# 1. 验证Z坐标连续性
dt = t[1] - t[0]  # 时间步长
dz = np.gradient(traj[:, 2], dt)  # Z方向速度

fig_verify, axs = plt.subplots(2, 1, figsize=(10, 8))

# Z坐标随时间变化
axs[0].plot(t, traj[:, 2], 'b-')
axs[0].set_title('Z坐标随时间变化')
axs[0].set_xlabel('时间 (s)')
axs[0].set_ylabel('Z高度')
axs[0].grid(True)

# 标记上升阶段结束点
rise_end_index = np.argmin(np.abs(t - 5.0))  # z_rise_time=5.0
axs[0].plot(t[rise_end_index], traj[rise_end_index, 2], 'ro', label=f'上升结束点 (t=5.0s)')
axs[0].legend()

# Z方向速度随时间变化
axs[1].plot(t, dz, 'g-')
axs[1].set_title('Z方向速度随时间变化')
axs[1].set_xlabel('时间 (s)')
axs[1].set_ylabel('dZ/dt (速度)')
axs[1].grid(True)

# 标记最大速度点
max_dz_index = np.argmax(dz)
axs[1].plot(t[max_dz_index], dz[max_dz_index], 'ro', label=f'最大速度点 (t={t[max_dz_index]:.2f}s)')
axs[1].legend()

plt.tight_layout()

# 2. 静态轨迹图（带速度向量）
fig_static = plt.figure(figsize=(12, 8))
ax_static = fig_static.add_subplot(111, projection='3d')

# 创建自定义颜色映射（随时间变化）
colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
ax_static.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=colors, s=2, alpha=0.7)

# 添加起点和终点标记
ax_static.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='起点')
ax_static.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='X', label='终点')

# 计算归一化速度向量
dx = np.gradient(traj[:, 0], dt)
dy = np.gradient(traj[:, 1], dt)
dz = np.gradient(traj[:, 2], dt)

# 归一化处理
speed = np.sqrt(dx**2 + dy**2 + dz**2)
max_speed = np.max(speed)
dx_norm = dx / max_speed * 0.3
dy_norm = dy / max_speed * 0.3
dz_norm = dz / max_speed * 0.3

# 每隔50个点绘制一个速度向量
for i in range(0, len(t), 50):
    ax_static.quiver(traj[i, 0], traj[i, 1], traj[i, 2],
                    dx_norm[i], dy_norm[i], dz_norm[i],
                    color='blue', length=0.3, arrow_length_ratio=0.5, alpha=0.8)

ax_static.set_title('3D 8字形轨迹（带速度向量）', fontsize=14)
ax_static.set_xlabel('X轴', fontsize=12)
ax_static.set_ylabel('Y轴', fontsize=12)
ax_static.set_zlabel('Z轴', fontsize=12)
ax_static.legend()
ax_static.grid(True)
fig_static.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax_static, label='时间进度')

plt.tight_layout()
plt.show()