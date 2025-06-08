import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def test_visualizations():
    """간단한 테스트용 시각화 생성"""
    
    print("테스트 시각화 생성 중...")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 1. 박스플롯 테스트
    plt.figure(figsize=(12, 8))
    
    # 4개 모델의 더미 데이터
    cruise_data = np.random.normal(3.8, 0.2, 50)
    scratch_data = np.random.normal(4.5, 0.3, 50) 
    lunar_data = np.random.normal(4.8, 0.25, 50)
    mountain_data = np.random.normal(4.6, 0.28, 50)
    
    data = [cruise_data, scratch_data, lunar_data, mountain_data]
    labels = ['Cruise Mode\n(Baseline)', 'SAC From Scratch\n(100k steps)', 
              'SAC LunarLander\nTransfer (50k)', 'SAC MountainCar\nTransfer (50k)']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 박스플롯 생성
    box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 평균값 표시
    means = [np.mean(d) for d in data]
    for i, mean in enumerate(means):
        plt.text(i+1, mean + 0.05, f'{mean:.2f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=11)
    
    plt.title('Energy Efficiency Comparison Test', fontweight='bold', fontsize=16)
    plt.ylabel('Energy Efficiency (km/kWh)', fontsize=12)
    plt.axhline(y=4.2, color='red', linestyle='--', alpha=0.8, label='Target (4.2)')
    plt.axhline(y=5.0, color='green', linestyle='--', alpha=0.8, label='Goal (5.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('test_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 막대 그래프 테스트
    plt.figure(figsize=(10, 6))
    
    model_names = ['Cruise', 'Scratch', 'LunarLander', 'MountainCar']
    performance = [3.8, 4.5, 4.8, 4.6]
    improvements = [0, 18.4, 26.3, 21.1]  # vs cruise 개선율
    
    bars = plt.bar(model_names, performance, color=colors, alpha=0.8)
    
    # 개선율 텍스트 추가
    for bar, perf, imp in zip(bars, performance, improvements):
        height = bar.get_height()
        if imp > 0:
            status = '✅' if imp >= 20 else '⚠️'
            text = f'{perf:.1f}\n{status}{imp:+.1f}%'
        else:
            text = f'{perf:.1f}\n(Baseline)'
        
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                text, ha='center', va='bottom', fontweight='bold')
    
    plt.title('Performance Ranking Test', fontweight='bold', fontsize=14)
    plt.ylabel('Performance Score', fontsize=12)
    plt.axhline(y=4.2, color='red', linestyle='--', alpha=0.8, label='Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_barplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 히트맵 테스트
    plt.figure(figsize=(8, 6))
    
    # p-value 매트릭스 (더미)
    p_matrix = np.array([
        [1.0, 0.001, 0.003, 0.005],
        [0.001, 1.0, 0.062, 0.341],
        [0.003, 0.062, 1.0, 0.351],
        [0.005, 0.341, 0.351, 1.0]
    ])
    
    # 대각선 마스크
    mask = np.eye(4, dtype=bool)
    
    # 히트맵 생성
    sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f',
               xticklabels=model_names, yticklabels=model_names,
               cmap='RdYlBu_r', center=0.05,
               cbar_kws={'label': 'p-value'})
    
    plt.title('Statistical Significance Matrix Test', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. 라인 차트 테스트 (학습 곡선)
    plt.figure(figsize=(12, 6))
    
    episodes = np.arange(1, 101)
    
    # 서로 다른 학습 패턴
    scratch_rewards = 0.3 + 0.5 * (1 - 0.98**episodes) + np.random.normal(0, 0.02, 100)
    lunar_rewards = 0.5 + 0.4 * (1 - 0.96**episodes) + np.random.normal(0, 0.015, 100)
    mountain_rewards = 0.45 + 0.35 * (1 - 0.97**episodes) + np.random.normal(0, 0.018, 100)
    
    plt.plot(episodes, scratch_rewards, label='SAC From Scratch', color='#4ECDC4', linewidth=2)
    plt.plot(episodes, lunar_rewards, label='SAC LunarLander Transfer', color='#45B7D1', linewidth=2)
    plt.plot(episodes, mountain_rewards, label='SAC MountainCar Transfer', color='#96CEB4', linewidth=2)
    
    plt.title('Learning Curves Comparison Test', fontweight='bold', fontsize=14)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 서브플롯 테스트
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Multi-Panel Visualization Test', fontsize=16, fontweight='bold')
    
    # 서브플롯 1: 산점도
    x = np.random.normal(0, 1, 100)
    y = 2*x + np.random.normal(0, 0.5, 100)
    axes[0, 0].scatter(x, y, alpha=0.6, color='#FF6B6B')
    axes[0, 0].set_title('Scatter Plot Test')
    axes[0, 0].set_xlabel('X values')
    axes[0, 0].set_ylabel('Y values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 서브플롯 2: 히스토그램
    data = np.random.gamma(2, 2, 1000)
    axes[0, 1].hist(data, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
    axes[0, 1].set_title('Histogram Test')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 서브플롯 3: 파이 차트
    sizes = [30, 25, 25, 20]
    labels = ['Model A', 'Model B', 'Model C', 'Model D']
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    axes[1, 0].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Pie Chart Test')
    
    # 서브플롯 4: 막대그래프 + 에러바
    categories = ['A', 'B', 'C', 'D']
    values = [4.2, 4.5, 4.8, 4.6]
    errors = [0.2, 0.3, 0.25, 0.28]
    axes[1, 1].bar(categories, values, yerr=errors, capsize=5, 
                   color=colors, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Bar Chart with Error Bars Test')
    axes[1, 1].set_ylabel('Performance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_subplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("모든 테스트 시각화 생성 완료!")
    print("생성된 파일:")
    print("  - test_boxplot.png")
    print("  - test_barplot.png") 
    print("  - test_heatmap.png")
    print("  - test_learning_curves.png")
    print("  - test_subplots.png")

# 실행
if __name__ == "__main__":
    test_visualizations()