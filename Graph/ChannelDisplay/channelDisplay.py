import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.switch_backend('agg')
def channel_Display(num,save_path='./graph_result'):
    def generate_random_ellipse(ax, center, width, height, angle):
        """生成一个椭圆并绘制在给定的轴上，使用深颜色"""
        ellipse = patches.Ellipse(center, width, height, angle=angle, color='grey', alpha=0.5)  # 修改颜色和透明度
        ax.add_patch(ellipse)
    def is_overlap(center, width, height, existing_ellipses):
        """检查新椭圆是否与已有椭圆重叠"""
        for (c, w, h) in existing_ellipses:
            dist = np.linalg.norm(np.array(center) - np.array(c))
            if dist < (width / 2 + w / 2) and dist < (height / 2 + h / 2):
                return True
        return False
    def generate_size():
        """生成椭圆大小：有更多很大的和很小的椭圆"""
        choice = random.random()
        if choice < 0.5:  # 20% 生成非常小的椭圆
            width = random.uniform(0.3, 1.2)
            height = random.uniform(0.6, 1.5)
        else:
            width = random.uniform(0.6, 1.4)
            height = random.uniform(0.3, 1.2)

        return width, height
    def create_fso_channel(num_ellipses):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        # ax.set_title('FSO Channel with Random Obstacles (Varied Sizes)')
        # 移除坐标轴
        ax.axis('off')  # 关闭横纵坐标轴显示
        existing_ellipses = []
        for _ in range(num_ellipses):
            width, height = generate_size()  # 调用生成椭圆大小函数
            center = (random.uniform(0, 10), random.uniform(0, 6))
            angle = random.uniform(0, 360)
            if not is_overlap(center, width, height, existing_ellipses):
                generate_random_ellipse(ax, center, width, height, angle)
                existing_ellipses.append((center, width, height))
        plt.savefig(save_path+"channelCondition.png", format="png", dpi=600, bbox_inches='tight', pad_inches=0,transparent=True)
        plt.show()
    create_fso_channel(num)