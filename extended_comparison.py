# 확장된 전이학습 비교 실험: MountainCar + LunarLander + 순수학습
# 현재 실험 완료 후 실행할 추가 비교 코드

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedTransferLearningComparison:
    """확장된 전이학습 비교 (4개 모델 동시 비교)"""
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}
        
        # 비교할 모델들
        self.models_to_compare = {
            'cruise_mode': {'type': 'baseline', 'description': '크루즈 모드 (PID 제어)'},
            'sac_scratch': {'type': 'scratch', 'description': 'SAC 순수학습 (100k 스텝)'},
            'sac_lunarlander': {'type': 'transfer', 'repo': 'sb3/sac-LunarLanderContinuous-v2', 'steps': 50000},
            'sac_mountaincar': {'type': 'transfer', 'repo': 'sb3/sac-MountainCarContinuous-v0', 'steps': 50000}
        }
        
        logger.info(" 확장된 전이학습 비교 실험 초기화")
        logger.info("비교 모델: 크루즈, 순수학습, LunarLander 전이, MountainCar 전이")
    
    def load_previous_results(self):
        """이전 실험 결과 로드"""
        
        logger.info("📂 이전 실험 결과 로드 중...")
        
        try:
            # 현재 실행 중인 실험 결과 찾기
            import glob
            result_files = glob.glob('statistical_experiment_results_*.json')
            
            if result_files:
                latest_file = max(result_files)
                with open(latest_file, 'r') as f:
                    previous_results = json.load(f)
                
                logger.info(f" 이전 결과 로드: {latest_file}")
                
                # 기존 결과 저장
                if 'performance' in previous_results:
                    self.results['cruise_mode'] = previous_results['performance'].get('cruise_mode')
                    self.results['sac_scratch'] = previous_results['performance'].get('sac_scratch')
                    self.results['sac_lunarlander'] = previous_results['performance'].get('sac_lunarlander_transfer')
                
                return True
            else:
                logger.warning(" 이전 실험 결과를 찾을 수 없음")
                return False
                
        except Exception as e:
            logger.error(f"이전 결과 로드 실패: {e}")
            return False
    
    def create_ev_environment(self):
        """전기차 환경 생성 (증강된 데이터 사용)"""
        
        try:
            from sagemaker_training import EVEnergyEnvironmentPreprocessed
            env = EVEnergyEnvironmentPreprocessed(data_dir="./")
            logger.info(" 전기차 환경 생성 완료")
            return env
        except Exception as e:
            logger.error(f"환경 생성 실패: {e}")
            return None
    
    def train_mountaincar_transfer_model(self, env):
        """MountainCar 전이학습 모델 훈련"""
        
        logger.info("🏔️ MountainCar 전이학습 시작...")
        
        try:
            from huggingface_sb3 import load_from_hub
            from stable_baselines3 import SAC
            import torch
            
            # 1. MountainCar 사전학습 모델 다운로드
            logger.info("📥 MountainCar 모델 다운로드...")
            checkpoint = load_from_hub(
                repo_id="sb3/sac-MountainCarContinuous-v0",
                filename="sac-MountainCarContinuous-v0.zip"
            )
            
            # 2. 사전학습 모델 로드 및 분석
            pretrained_model = SAC.load(checkpoint)
            logger.info(f" MountainCar 모델 로드 완료")
            logger.info(f"   관측 공간: {pretrained_model.observation_space.shape[0]}차원")
            logger.info(f"   행동 공간: {pretrained_model.action_space.shape[0]}차원")
            
            # 3. 전기차 환경용 새 모델 생성
            transfer_config = {
                'learning_rate': 1e-4,  # 낮은 학습률 (미세조정)
                'buffer_size': 50000,
                'batch_size': 64,
                'tau': 0.01,
                'gamma': 0.95,
                'policy_kwargs': {
                    'net_arch': [128, 128],
                    'activation_fn': torch.nn.ReLU
                    # dropout 제거됨
                },
                'verbose': 1
            }
            
            target_model = SAC('MlpPolicy', env, **transfer_config)
            
            # 4. 가중치 전이 시도
            transfer_success = self.transfer_mountaincar_weights(pretrained_model, target_model)
            
            if transfer_success:
                logger.info(" MountainCar 가중치 전이 성공")
            else:
                logger.info(" 가중치 전이 실패, 랜덤 초기화로 진행")
            
            # 5. 미세조정 훈련
            logger.info("MountainCar 전이학습 미세조정 (50k 스텝)...")
            target_model.learn(
                total_timesteps=50000,
                progress_bar=True
            )
            
            # 6. 모델 저장
            model_path = f"./models/sac_mountaincar_transfer_{self.experiment_id}.zip"
            target_model.save(model_path)
            logger.info(f"💾 MountainCar 모델 저장: {model_path}")
            
            return target_model, {
                'transfer_success': transfer_success,
                'model_path': model_path,
                'training_steps': 50000
            }
            
        except Exception as e:
            logger.error(f"MountainCar 전이학습 실패: {e}")
            return None, {'transfer_success': False, 'error': str(e)}
    
    def transfer_mountaincar_weights(self, source_model, target_model):
        """MountainCar 가중치 전이 (더 정교한 방법)"""
        
        try:
            source_params = source_model.policy.state_dict()
            target_params = target_model.policy.state_dict()
            
            transferred_layers = 0
            total_layers = len(target_params)
            
            logger.info("🔄 가중치 전이 시도 중...")
            
            for target_key, target_tensor in target_params.items():
                # 호환 가능한 레이어 찾기
                for source_key, source_tensor in source_params.items():
                    if self.is_compatible_layer(source_key, target_key, source_tensor, target_tensor):
                        
                        # 차원이 정확히 맞는 경우
                        if source_tensor.shape == target_tensor.shape:
                            target_params[target_key] = source_tensor.clone()
                            transferred_layers += 1
                            logger.info(f"    전이: {source_key} → {target_key} {source_tensor.shape}")
                            break
                        
                        # 부분 호환 (첫 번째 차원만 맞는 경우)
                        elif (len(source_tensor.shape) == len(target_tensor.shape) and 
                              source_tensor.shape[0] <= target_tensor.shape[0]):
                            
                            # 가능한 부분만 전이
                            if len(source_tensor.shape) == 2:
                                target_params[target_key][:source_tensor.shape[0], :source_tensor.shape[1]] = source_tensor
                            elif len(source_tensor.shape) == 1:
                                target_params[target_key][:source_tensor.shape[0]] = source_tensor
                            
                            transferred_layers += 1
                            logger.info(f"   🔸 부분 전이: {source_key} → {target_key}")
                            break
            
            # 업데이트된 가중치 로드
            target_model.policy.load_state_dict(target_params)
            
            transfer_rate = transferred_layers / total_layers
            logger.info(f"전이 완료: {transferred_layers}/{total_layers} ({transfer_rate:.1%})")
            
            return transfer_rate > 0.05  # 5% 이상 전이되면 성공
            
        except Exception as e:
            logger.warning(f" 가중치 전이 중 오류: {e}")
            return False
    
    def is_compatible_layer(self, source_key, target_key, source_tensor, target_tensor):
        """레이어 호환성 확인 (MountainCar 특화)"""
        
        # 유사한 레이어 패턴
        compatible_patterns = [
            ('actor.mu', 'actor.mu'),           # Actor 출력
            ('critic.qf', 'critic.qf'),         # Critic Q-function
            ('.0.weight', '.0.weight'),         # 첫 번째 레이어
            ('.2.weight', '.2.weight'),         # 두 번째 레이어
            ('.0.bias', '.0.bias'),             # 첫 번째 바이어스
            ('.2.bias', '.2.bias'),             # 두 번째 바이어스
        ]
        
        for source_pattern, target_pattern in compatible_patterns:
            if source_pattern in source_key and target_pattern in target_key:
                return True
        
        return False
    
    def evaluate_mountaincar_model(self, model, env):
        """MountainCar 전이학습 모델 평가"""
        
        logger.info(" MountainCar 모델 평가 중...")
        
        rewards = []
        efficiency_values = []
        
        for episode in range(30):  # 통계적 신뢰성을 위한 30회
            obs, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            rewards.append(total_reward)
            metrics = env.get_episode_metrics()
            efficiency_values.append(metrics['energy_efficiency'])
        
        # 통계적 분석
        results = {
            'episodes': len(rewards),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards, ddof=1),
            'mean_efficiency': np.mean(efficiency_values),
            'std_efficiency': np.std(efficiency_values, ddof=1),
            'confidence_interval_95': stats.t.interval(
                0.95, len(efficiency_values)-1,
                loc=np.mean(efficiency_values),
                scale=stats.sem(efficiency_values)
            ),
            'raw_values': {
                'rewards': rewards,
                'efficiencies': efficiency_values
            }
        }
        
        logger.info(f"🏔️ MountainCar 결과:")
        logger.info(f"   평균 효율: {results['mean_efficiency']:.3f} ± {results['std_efficiency']:.3f} km/kWh")
        logger.info(f"   95% 신뢰구간: [{results['confidence_interval_95'][0]:.3f}, {results['confidence_interval_95'][1]:.3f}]")
        
        return results
    
    def perform_extended_statistical_comparison(self):
        """4개 모델 확장 통계 비교"""
        
        logger.info("📈 확장된 통계 분석 (4개 모델)...")
        
        # 에너지 효율 값 추출
        model_efficiencies = {}
        
        for model_name, result in self.results.items():
            if result and 'mean_efficiency' in result:
                model_efficiencies[model_name] = result['raw_values']['efficiencies']
            elif result and 'energy_efficiency' in result:
                model_efficiencies[model_name] = result['energy_efficiency']['values']
        
        if len(model_efficiencies) < 2:
            logger.error("비교할 모델이 부족합니다")
            return None
        
        # 통계 분석
        analysis = {
            'model_means': {},
            'pairwise_comparisons': {},
            'anova_test': None,
            'ranking': [],
            'improvement_rates': {}
        }
        
        # 모델별 평균 계산
        for model, values in model_efficiencies.items():
            analysis['model_means'][model] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'n': len(values)
            }
        
        # 순위 매기기
        ranking = sorted(analysis['model_means'].items(), 
                        key=lambda x: x[1]['mean'], reverse=True)
        analysis['ranking'] = [(name, data['mean']) for name, data in ranking]
        
        # 쌍별 비교 (모든 조합)
        model_names = list(model_efficiencies.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # 중복 방지
                    values1 = model_efficiencies[model1]
                    values2 = model_efficiencies[model2]
                    
                    # t-검정
                    t_stat, t_p = stats.ttest_ind(values1, values2)
                    
                    # Mann-Whitney U 검정 (비모수)
                    u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    
                    # 효과크기 (Cohen's d)
                    cohens_d = self.calculate_cohens_d(values1, values2)
                    
                    comparison_key = f"{model1}_vs_{model2}"
                    analysis['pairwise_comparisons'][comparison_key] = {
                        't_test': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < 0.05},
                        'mann_whitney': {'statistic': u_stat, 'p_value': u_p, 'significant': u_p < 0.05},
                        'effect_size': cohens_d,
                        'mean_difference': np.mean(values1) - np.mean(values2)
                    }
        
        # ANOVA (3개 이상 모델이 있는 경우)
        if len(model_efficiencies) >= 3:
            anova_values = list(model_efficiencies.values())
            f_stat, anova_p = stats.f_oneway(*anova_values)
            analysis['anova_test'] = {
                'f_statistic': f_stat,
                'p_value': anova_p,
                'significant': anova_p < 0.05
            }
        
        # 크루즈 모드 대비 개선율 계산
        if 'cruise_mode' in analysis['model_means']:
            cruise_mean = analysis['model_means']['cruise_mode']['mean']
            
            for model, data in analysis['model_means'].items():
                if model != 'cruise_mode':
                    improvement = ((data['mean'] - cruise_mean) / cruise_mean) * 100
                    analysis['improvement_rates'][model] = improvement
        
        return analysis
    
    def calculate_cohens_d(self, group1, group2):
        """Cohen's d 효과크기 계산"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def create_comprehensive_visualizations(self, statistical_analysis):
        """sagemaker_test.py 수준의 종합 시각화 생성"""
        
        logger.info(" 종합 시각화 생성 중 (sagemaker_test.py 수준)...")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec
        
        # 스타일 설정
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 모델 표시명 매핑
        model_display_names = {
            'cruise_mode': '크루즈 모드',
            'sac_scratch': 'SAC 순수학습',
            'sac_lunarlander': 'SAC LunarLander 전이',
            'sac_mountaincar': 'SAC MountainCar 전이'
        }
        
        # 1. 메인 성능 비교 차트 (2x2 레이아웃)
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        
        fig.suptitle('4개 모델 종합 성능 비교\nMountainCar vs LunarLander vs 순수학습 vs 크루즈', 
                    fontsize=20, fontweight='bold', y=0.95)
        
        # 1-1. 에너지 효율 박스플롯
        ax1 = fig.add_subplot(gs[0, 0])
        efficiency_data = []
        labels = []
        
        for model_name in statistical_analysis['ranking']:
            model_key = model_name[0]
            if model_key in self.results and self.results[model_key]:
                if 'raw_values' in self.results[model_key]:
                    efficiency_data.append(self.results[model_key]['raw_values']['efficiencies'])
                elif 'energy_efficiency' in self.results[model_key]:
                    efficiency_data.append(self.results[model_key]['energy_efficiency']['values'])
                labels.append(model_display_names.get(model_key, model_key))
        
        if efficiency_data:
            box_plot = ax1.boxplot(efficiency_data, labels=labels, patch_artist=True)
            colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
            for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax1.set_title('에너지 효율 분포 비교', fontweight='bold', fontsize=14)
        ax1.set_ylabel('에너지 효율 (km/kWh)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 평균값 표시
        for i, data in enumerate(efficiency_data):
            mean_val = np.mean(data)
            ax1.text(i+1, mean_val + 0.05, f'{mean_val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 1-2. 성능 순위 막대차트
        ax2 = fig.add_subplot(gs[0, 1])
        models = [model_display_names.get(name[0], name[0]) for name in statistical_analysis['ranking']]
        efficiencies = [eff for _, eff in statistical_analysis['ranking']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = ax2.bar(models, efficiencies, color=colors[:len(models)], alpha=0.8)
        ax2.set_title('성능 순위 (에너지 효율)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('에너지 효율 (km/kWh)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 순위 표시
        for i, (bar, eff) in enumerate(zip(bars, efficiencies)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{i+1}위\n{eff:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # 1-3. 개선율 비교
        ax3 = fig.add_subplot(gs[1, 0])
        if statistical_analysis['improvement_rates']:
            improvement_models = []
            improvement_values = []
            
            for model, improvement in statistical_analysis['improvement_rates'].items():
                improvement_models.append(model_display_names.get(model, model))
                improvement_values.append(improvement)
            
            colors = ['green' if x >= 20 else 'orange' if x >= 10 else 'red' 
                     for x in improvement_values]
            
            bars = ax3.bar(improvement_models, improvement_values, color=colors, alpha=0.8)
            ax3.axhline(y=20, color='red', linestyle='--', linewidth=2, label='목표 20% 개선')
            ax3.set_title('크루즈 모드 대비 개선율', fontweight='bold', fontsize=14)
            ax3.set_ylabel('개선율 (%)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 개선율 값 표시
            for bar, improvement in zip(bars, improvement_values):
                height = bar.get_height()
                status = '' if improvement >= 20 else '' if improvement >= 10 else '❌'
                ax3.text(bar.get_x() + bar.get_width()/2., 
                        height + (1 if height > 0 else -3),
                        f'{status}\n{improvement:.1f}%', 
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold', fontsize=10)
        
        # 1-4. 통계적 유의성 히트맵
        ax4 = fig.add_subplot(gs[1, 1])
        
        # p-value 매트릭스 생성
        model_names = list(statistical_analysis['model_means'].keys())
        n_models = len(model_names)
        p_matrix = np.ones((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i != j:
                    comparison_key = f"{model1}_vs_{model2}"
                    reverse_key = f"{model2}_vs_{model1}"
                    
                    if comparison_key in statistical_analysis['pairwise_comparisons']:
                        p_val = statistical_analysis['pairwise_comparisons'][comparison_key]['t_test']['p_value']
                        p_matrix[i, j] = p_val
                    elif reverse_key in statistical_analysis['pairwise_comparisons']:
                        p_val = statistical_analysis['pairwise_comparisons'][reverse_key]['t_test']['p_value']
                        p_matrix[i, j] = p_val
        
        # 히트맵 생성
        mask = np.eye(n_models, dtype=bool)
        display_names = [model_display_names.get(name, name) for name in model_names]
        
        sns.heatmap(p_matrix, mask=mask, annot=True, fmt='.3f', 
                   xticklabels=display_names, yticklabels=display_names,
                   cmap='RdYlBu_r', center=0.05, ax=ax4,
                   cbar_kws={'label': 'p-value'})
        ax4.set_title('통계적 유의성 (p-values)', fontweight='bold', fontsize=14)
        
        # 1-5. 효과크기 비교
        ax5 = fig.add_subplot(gs[2, 0])
        
        effect_sizes = []
        comparison_labels = []
        
        for key, comparison in statistical_analysis['pairwise_comparisons'].items():
            if 'vs' in key:
                model1, model2 = key.split('_vs_')
                label = f"{model_display_names.get(model1, model1)}\nvs\n{model_display_names.get(model2, model2)}"
                comparison_labels.append(label)
                effect_sizes.append(abs(comparison['effect_size']))
        
        if effect_sizes:
            colors = ['green' if x >= 0.8 else 'orange' if x >= 0.5 else 'yellow' if x >= 0.2 else 'red' 
                     for x in effect_sizes]
            
            bars = ax5.bar(range(len(effect_sizes)), effect_sizes, color=colors, alpha=0.8)
            ax5.set_xticks(range(len(comparison_labels)))
            ax5.set_xticklabels(comparison_labels, fontsize=8)
            ax5.set_title('효과크기 (Cohen\'s d)', fontweight='bold', fontsize=14)
            ax5.set_ylabel('효과크기', fontsize=12)
            ax5.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='소 효과 (0.2)')
            ax5.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='중 효과 (0.5)')
            ax5.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='대 효과 (0.8)')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # 효과크기 값 표시
            for bar, effect in zip(bars, effect_sizes):
                height = bar.get_height()
                if effect >= 0.8:
                    effect_desc = '대'
                elif effect >= 0.5:
                    effect_desc = '중'
                elif effect >= 0.2:
                    effect_desc = '소'
                else:
                    effect_desc = '미미'
                
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{effect:.2f}\n({effect_desc})', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
        
        # 1-6. 가설 검증 결과
        ax6 = fig.add_subplot(gs[2, 1])
        
        # 가설 검증 항목들
        hypotheses = []
        results = []
        
        # H1: 전이학습 우수성
        if ('sac_mountaincar' in statistical_analysis['model_means'] and 
            'sac_lunarlander' in statistical_analysis['model_means']):
            mountain_eff = statistical_analysis['model_means']['sac_mountaincar']['mean']
            lunar_eff = statistical_analysis['model_means']['sac_lunarlander']['mean']
            
            hypotheses.append('H1: MountainCar\n> LunarLander')
            results.append(1 if mountain_eff > lunar_eff else 0)
        
        # H2: 20% 개선 달성
        for model in ['sac_scratch', 'sac_lunarlander', 'sac_mountaincar']:
            if model in statistical_analysis['improvement_rates']:
                improvement = statistical_analysis['improvement_rates'][model]
                model_name = model_display_names.get(model, model)
                hypotheses.append(f'H2: {model_name}\n≥20% 개선')
                results.append(1 if improvement >= 20 else 0)
        
        if hypotheses:
            colors = ['green' if x == 1 else 'red' for x in results]
            bars = ax6.bar(hypotheses, results, color=colors, alpha=0.8)
            ax6.set_title('가설 검증 결과', fontweight='bold', fontsize=14)
            ax6.set_ylabel('달성 여부 (1=성공, 0=실패)', fontsize=12)
            ax6.set_ylim(0, 1.2)
            ax6.tick_params(axis='x', rotation=45)
            ax6.grid(True, alpha=0.3)
            
            # 결과 텍스트 표시
            for bar, result in zip(bars, results):
                text = ' 달성' if result else '미달성'
                ax6.text(bar.get_x() + bar.get_width()/2., 0.5, text, 
                        ha='center', va='center', fontweight='bold', fontsize=10,
                        color='white')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/comprehensive_4model_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 상세 통계 분석 차트
        self.create_detailed_statistical_charts(statistical_analysis)
        
        # 3. 전이학습 효과 분석 차트
        self.create_transfer_learning_analysis_charts(statistical_analysis)
        
        logger.info(" 종합 시각화 완료")
    
    def create_detailed_statistical_charts(self, statistical_analysis):
        """상세 통계 분석 차트"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('상세 통계 분석 결과', fontsize=16, fontweight='bold')
        
        model_display_names = {
            'cruise_mode': '크루즈 모드',
            'sac_scratch': 'SAC 순수학습',
            'sac_lunarlander': 'SAC LunarLander 전이',
            'sac_mountaincar': 'SAC MountainCar 전이'
        }
        
        # 2-1. 신뢰구간 비교
        models = []
        means = []
        ci_lowers = []
        ci_uppers = []
        
        for model_name, result in self.results.items():
            if result and 'confidence_interval_95' in result:
                models.append(model_display_names.get(model_name, model_name))
                means.append(result['mean_efficiency'])
                ci_lower, ci_upper = result['confidence_interval_95']
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
        
        if models:
            x_pos = np.arange(len(models))
            axes[0, 0].errorbar(x_pos, means, 
                              yerr=[np.array(means) - np.array(ci_lowers),
                                    np.array(ci_uppers) - np.array(means)],
                              fmt='o', capsize=5, capthick=2, markersize=8)
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(models, rotation=45)
            axes[0, 0].set_title('95% 신뢰구간 비교', fontweight='bold')
            axes[0, 0].set_ylabel('에너지 효율 (km/kWh)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2-2. 분산 비교
        models = []
        variances = []
        
        for model_name, data in statistical_analysis['model_means'].items():
            models.append(model_display_names.get(model_name, model_name))
            variances.append(data['std']**2)
        
        if models:
            bars = axes[0, 1].bar(models, variances, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('분산 비교 (안정성)', fontweight='bold')
            axes[0, 1].set_ylabel('분산')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
            
            for bar, var in zip(bars, variances):
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{var:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2-3. 정규성 검정 결과 (만약 있다면)
        axes[0, 2].text(0.5, 0.5, 'Normality Tests\n(if available)', 
                       ha='center', va='center', fontsize=12, 
                       transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('정규성 검정', fontweight='bold')
        
        # 2-4. ANOVA 결과
        if statistical_analysis['anova_test']:
            anova = statistical_analysis['anova_test']
            
            labels = ['F-통계량', 'p-값']
            values = [anova['f_statistic'], anova['p_value']]
            
            bars = axes[0, 3].bar(labels, values, color=['orange', 'lightgreen'], alpha=0.7)
            axes[0, 3].set_title('일원배치 분산분석 (ANOVA)', fontweight='bold')
            axes[0, 3].grid(True, alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                axes[0, 3].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            # 유의성 표시
            significance = '유의함' if anova['significant'] else '비유의함'
            axes[0, 3].text(0.5, 0.8, f'결과: {significance}', 
                           ha='center', va='center', fontsize=12, fontweight='bold',
                           transform=axes[0, 3].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/detailed_statistical_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_transfer_learning_analysis_charts(self, statistical_analysis):
        """전이학습 효과 분석 차트"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('전이학습 효과 분석', fontsize=16, fontweight='bold')
        
        # 3-1. 전이학습 vs 순수학습 비교
        transfer_models = []
        transfer_efficiencies = []
        
        if 'sac_scratch' in statistical_analysis['model_means']:
            scratch_eff = statistical_analysis['model_means']['sac_scratch']['mean']
            
            for model in ['sac_lunarlander', 'sac_mountaincar']:
                if model in statistical_analysis['model_means']:
                    transfer_models.append('LunarLander' if 'lunar' in model else 'MountainCar')
                    transfer_efficiencies.append(statistical_analysis['model_means'][model]['mean'])
            
            if transfer_models:
                colors = ['lightblue' if eff > scratch_eff else 'lightcoral' for eff in transfer_efficiencies]
                bars = axes[0, 0].bar(transfer_models, transfer_efficiencies, 
                                    color=colors, alpha=0.8)
                axes[0, 0].axhline(y=scratch_eff, color='red', linestyle='--', 
                                 label=f'순수학습: {scratch_eff:.3f}')
                axes[0, 0].set_title('전이학습 모델 vs 순수학습', fontweight='bold')
                axes[0, 0].set_ylabel('에너지 효율 (km/kWh)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                for bar, eff in zip(bars, transfer_efficiencies):
                    height = bar.get_height()
                    improvement = ((eff - scratch_eff) / scratch_eff) * 100
                    status = '↑' if improvement > 0 else '↓'
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                   f'{eff:.3f}\n{status}{abs(improvement):.1f}%', 
                                   ha='center', va='bottom', fontweight='bold')
        
        # 3-2. 도메인 유사성 분석
        domain_similarity = {
            'LunarLander': {'물리법칙': 8, '제어방식': 9, '환경복잡도': 6, '목표유사성': 5},
            'MountainCar': {'물리법칙': 9, '제어방식': 8, '환경복잡도': 4, '목표유사성': 8}
        }
        
        categories = list(domain_similarity['LunarLander'].keys())
        lunar_scores = list(domain_similarity['LunarLander'].values())
        mountain_scores = list(domain_similarity['MountainCar'].values())
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[0, 1].bar(x - width/2, lunar_scores, width, label='LunarLander', alpha=0.8)
        bars2 = axes[0, 1].bar(x + width/2, mountain_scores, width, label='MountainCar', alpha=0.8)
        
        axes[0, 1].set_title('도메인 유사성 분석 (주관적 평가)', fontweight='bold')
        axes[0, 1].set_ylabel('유사성 점수 (1-10)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3-3. 학습 효율성 (스텝당 성능)
        training_data = {
            'SAC 순수학습': {'스텝': 100000, '효율': 0},
            'SAC LunarLander 전이': {'스텝': 50000, '효율': 0},
            'SAC MountainCar 전이': {'스텝': 50000, '효율': 0}
        }
        
        # 효율 데이터 채우기
        for model_key, model_name in [('sac_scratch', 'SAC 순수학습'), 
                                     ('sac_lunarlander', 'SAC LunarLander 전이'),
                                     ('sac_mountaincar', 'SAC MountainCar 전이')]:
            if model_key in statistical_analysis['model_means']:
                training_data[model_name]['효율'] = statistical_analysis['model_means'][model_key]['mean']
        
        models = list(training_data.keys())
        steps = [training_data[model]['스텝'] for model in models]
        efficiencies = [training_data[model]['효율'] for model in models]
        efficiency_per_step = [eff/step*1000 if step > 0 else 0 for eff, step in zip(efficiencies, steps)]
        
        bars = axes[1, 0].bar(models, efficiency_per_step, 
                             color=['orange', 'lightblue', 'lightgreen'], alpha=0.8)
        axes[1, 0].set_title('학습 효율성 (1000스텝당 성능)', fontweight='bold')
        axes[1, 0].set_ylabel('효율/1000스텝')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, eps in zip(bars, efficiency_per_step):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                           f'{eps:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3-4. 전이학습 성공률 분석
        transfer_success_data = {
            'LunarLander': {'가중치_전이율': 6.2, '최종_성능': 0, '개선여부': False},
            'MountainCar': {'가중치_전이율': 0, '최종_성능': 0, '개선여부': False}  # 실제 결과로 업데이트
        }
        
        # 실제 데이터로 업데이트
        if 'sac_lunarlander' in statistical_analysis['model_means']:
            transfer_success_data['LunarLander']['최종_성능'] = statistical_analysis['model_means']['sac_lunarlander']['mean']
            if 'sac_scratch' in statistical_analysis['model_means']:
                scratch_perf = statistical_analysis['model_means']['sac_scratch']['mean']
                transfer_success_data['LunarLander']['개선여부'] = transfer_success_data['LunarLander']['최종_성능'] > scratch_perf
        
        if 'sac_mountaincar' in statistical_analysis['model_means']:
            transfer_success_data['MountainCar']['최종_성능'] = statistical_analysis['model_means']['sac_mountaincar']['mean']
            if 'sac_scratch' in statistical_analysis['model_means']:
                scratch_perf = statistical_analysis['model_means']['sac_scratch']['mean']
                transfer_success_data['MountainCar']['개선여부'] = transfer_success_data['MountainCar']['최종_성능'] > scratch_perf
        
        # 성공률 막대차트
        models = list(transfer_success_data.keys())
        success_scores = []
        
        for model, data in transfer_success_data.items():
            # 종합 성공 점수 (가중치 전이 + 성능 개선)
            weight_score = min(data['가중치_전이율'] / 10, 1.0)  # 10% 기준으로 정규화
            performance_score = 1.0 if data['개선여부'] else 0.0
            total_score = (weight_score * 0.3 + performance_score * 0.7) * 100
            success_scores.append(total_score)
        
        colors = ['green' if score >= 70 else 'orange' if score >= 40 else 'red' for score in success_scores]
        bars = axes[1, 1].bar(models, success_scores, color=colors, alpha=0.8)
        axes[1, 1].set_title('전이학습 종합 성공률', fontweight='bold')
        axes[1, 1].set_ylabel('성공률 (%)')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, score in zip(bars, success_scores):
            height = bar.get_height()
            status = '성공' if score >= 70 else '부분성공' if score >= 40 else '실패'
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{score:.1f}%\n({status})', ha='center', va='bottom', 
                           fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/transfer_learning_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_publication_ready_tables(self, statistical_analysis):
        """논문용 LaTeX 테이블 생성"""
        
        logger.info("📋 논문용 테이블 생성...")
        
        model_display_names = {
            'cruise_mode': 'Cruise Mode',
            'sac_scratch': 'SAC From Scratch',
            'sac_lunarlander': 'SAC + LunarLander Transfer',
            'sac_mountaincar': 'SAC + MountainCar Transfer'
        }
        
        # 1. 메인 성능 비교 테이블
        performance_table = """\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Four Models}
\\label{tab:performance_comparison}
\\begin{tabular}{lcccc}
\\toprule
Model & Energy Efficiency & 95\\% CI & Improvement & Rank \\\\
& (km/kWh) & (km/kWh) & (\\%) & \\\\
\\midrule
"""
        
        for i, (model_name, efficiency) in enumerate(statistical_analysis['ranking'], 1):
            model_display = model_display_names.get(model_name, model_name)
            
            # 신뢰구간 정보
            if model_name in self.results and 'confidence_interval_95' in self.results[model_name]:
                ci = self.results[model_name]['confidence_interval_95']
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                ci_str = "N/A"
            
            # 개선율
            if model_name in statistical_analysis['improvement_rates']:
                improvement = statistical_analysis['improvement_rates'][model_name]
                improvement_str = f"{improvement:+.1f}"
            else:
                improvement_str = "Baseline"
            
            performance_table += f"{model_display} & {efficiency:.3f} & {ci_str} & {improvement_str} & {i} \\\\\n"
        
        performance_table += """\\bottomrule
\\end{tabular}
\\end{table}

"""
        
        # 2. 통계적 검정 결과 테이블
        statistical_table = """\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance Tests (Pairwise Comparisons)}
\\label{tab:statistical_tests}
\\begin{tabular}{lcccr}
\\toprule
Comparison & t-statistic & p-value & Cohen's d & Significance \\\\
\\midrule
"""
        
        for comparison_name, comparison_data in statistical_analysis['pairwise_comparisons'].items():
            model1, model2 = comparison_name.split('_vs_')
            model1_display = model_display_names.get(model1, model1)
            model2_display = model_display_names.get(model2, model2)
            
            t_stat = comparison_data['t_test']['statistic']
            p_val = comparison_data['t_test']['p_value']
            cohens_d = comparison_data['effect_size']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            comparison_label = f"{model1_display} vs {model2_display}"
            statistical_table += f"{comparison_label} & {t_stat:.3f} & {p_val:.4f} & {cohens_d:.3f} & {significance} \\\\\n"
        
        statistical_table += """\\bottomrule
\\end{tabular}
\\note{*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant}
\\end{table}

"""
        
        # 3. 전이학습 효과 분석 테이블
        transfer_table = """\\begin{table}[htbp]
\\centering
\\caption{Transfer Learning Effectiveness Analysis}
\\label{tab:transfer_learning}
\\begin{tabular}{lcccc}
\\toprule
Transfer Model & Training Steps & Weight Transfer & Final Performance & vs Scratch \\\\
& & Rate (\\%) & (km/kWh) & Improvement \\\\
\\midrule
"""
        
        transfer_models = [
            ('sac_lunarlander', 'LunarLander', 50000, 6.2),
            ('sac_mountaincar', 'MountainCar', 50000, 0)  # 실제 전이율로 업데이트 필요
        ]
        
        scratch_performance = statistical_analysis['model_means'].get('sac_scratch', {}).get('mean', 0)
        
        for model_key, model_name, steps, transfer_rate in transfer_models:
            if model_key in statistical_analysis['model_means']:
                performance = statistical_analysis['model_means'][model_key]['mean']
                vs_scratch = ((performance - scratch_performance) / scratch_performance) * 100 if scratch_performance > 0 else 0
                vs_scratch_str = f"{vs_scratch:+.1f}\\%"
                
                transfer_table += f"{model_name} & {steps:,} & {transfer_rate:.1f} & {performance:.3f} & {vs_scratch_str} \\\\\n"
        
        transfer_table += """\\bottomrule
\\end{tabular}
\\end{table}

"""
        
        # 4. ANOVA 결과 테이블
        anova_table = ""
        if statistical_analysis['anova_test']:
            anova = statistical_analysis['anova_test']
            anova_table = f"""\\begin{table}[htbp]
\\centering
\\caption{{One-way ANOVA Results}}
\\label{{tab:anova}}
\\begin{{tabular}}{{lcc}}
\\toprule
Source & F-statistic & p-value \\\\
\\midrule
Between Groups & {anova['f_statistic']:.3f} & {anova['p_value']:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\note{{Null hypothesis: All group means are equal}}
\\end{{table}}

"""
        
        # 전체 LaTeX 테이블 저장
        full_latex = performance_table + statistical_table + transfer_table + anova_table
        
        with open(f'{self.results_dir}/publication_tables.tex', 'w', encoding='utf-8') as f:
            f.write(full_latex)
        
        # CSV 형태로도 저장 (스프레드시트 호환)
        self.create_csv_tables(statistical_analysis)
        
        logger.info(" 논문용 테이블 생성 완료")
        logger.info(f"  - LaTeX: {self.results_dir}/publication_tables.tex")
        logger.info(f"  - CSV: {self.results_dir}/results_tables.csv")
    
    def create_csv_tables(self, statistical_analysis):
        """CSV 형태 테이블 생성"""
        
        import csv
        
        # 성능 비교 CSV
        with open(f'{self.results_dir}/performance_comparison.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Energy_Efficiency_kmkWh', 'Std_Dev', 'CI_Lower', 'CI_Upper', 'Improvement_Percent', 'Rank'])
            
            for i, (model_name, efficiency) in enumerate(statistical_analysis['ranking'], 1):
                model_data = statistical_analysis['model_means'][model_name]
                
                # 신뢰구간
                if model_name in self.results and 'confidence_interval_95' in self.results[model_name]:
                    ci_lower, ci_upper = self.results[model_name]['confidence_interval_95']
                else:
                    ci_lower, ci_upper = '', ''
                
                # 개선율
                improvement = statistical_analysis['improvement_rates'].get(model_name, 0)
                
                writer.writerow([
                    model_name, 
                    f"{efficiency:.4f}", 
                    f"{model_data['std']:.4f}",
                    f"{ci_lower:.4f}" if ci_lower != '' else '',
                    f"{ci_upper:.4f}" if ci_upper != '' else '',
                    f"{improvement:.2f}" if improvement != 0 else '',
                    i
                ])
        
        # 통계 검정 CSV
        with open(f'{self.results_dir}/statistical_tests.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Comparison', 'T_Statistic', 'P_Value', 'Cohens_D', 'Significant', 'Effect_Size_Category'])
            
            for comparison_name, comparison_data in statistical_analysis['pairwise_comparisons'].items():
                t_stat = comparison_data['t_test']['statistic']
                p_val = comparison_data['t_test']['p_value']
                cohens_d = comparison_data['effect_size']
                significant = 'Yes' if comparison_data['t_test']['significant'] else 'No'
                
                # 효과크기 분류
                abs_d = abs(cohens_d)
                if abs_d >= 0.8:
                    effect_category = 'Large'
                elif abs_d >= 0.5:
                    effect_category = 'Medium'
                elif abs_d >= 0.2:
                    effect_category = 'Small'
                else:
                    effect_category = 'Negligible'
                
                writer.writerow([
                    comparison_name.replace('_vs_', ' vs '),
                    f"{t_stat:.4f}",
                    f"{p_val:.6f}",
                    f"{cohens_d:.4f}",
                    significant,
                    effect_category
                ])
    
    def run_extended_comparison(self):
        """확장된 비교 실험 전체 실행"""
        
        logger.info("확장된 전이학습 비교 실험 시작!")
        logger.info("=" * 60)
        
        # 결과 디렉토리 생성
        self.results_dir = f"extended_results_{self.experiment_id}"
        import os
        os.makedirs(self.results_dir, exist_ok=True)
        
        try:
            # 1. 이전 결과 로드
            if not self.load_previous_results():
                logger.error("이전 실험 결과가 필요합니다. 먼저 기본 실험을 완료하세요.")
                return None
            
            # 2. 환경 생성
            env = self.create_ev_environment()
            if not env:
                return None
            
            # 3. MountainCar 전이학습 실행
            mountaincar_model, mountaincar_info = self.train_mountaincar_transfer_model(env)
            
            if mountaincar_model:
                # 4. MountainCar 모델 평가
                mountaincar_results = self.evaluate_mountaincar_model(mountaincar_model, env)
                self.results['sac_mountaincar'] = mountaincar_results
            else:
                logger.error("MountainCar 모델 훈련 실패")
                return None
            
            # 5. 4개 모델 통계 비교
            statistical_analysis = self.perform_extended_statistical_comparison()
            
            if not statistical_analysis:
                return None
            
            # 6. 종합 시각화 생성 (sagemaker_test.py 수준)
            self.create_comprehensive_visualizations(statistical_analysis)
            
            # 7. 논문용 테이블 생성
            self.create_publication_ready_tables(statistical_analysis)
            
            # 8. 상세 보고서 생성
            report = self.generate_extended_report(statistical_analysis)
            
            # 9. 결과 저장
            final_results = {
                'experiment_id': self.experiment_id,
                'models_compared': list(self.results.keys()),
                'individual_results': self.results,
                'statistical_analysis': statistical_analysis,
                'mountaincar_transfer_info': mountaincar_info,
                'visualization_files': {
                    'comprehensive_comparison': f'{self.results_dir}/comprehensive_4model_comparison.png',
                    'detailed_statistical': f'{self.results_dir}/detailed_statistical_analysis.png',
                    'transfer_learning_analysis': f'{self.results_dir}/transfer_learning_analysis.png'
                },
                'publication_files': {
                    'latex_tables': f'{self.results_dir}/publication_tables.tex',
                    'csv_tables': f'{self.results_dir}/performance_comparison.csv',
                    'markdown_report': f'extended_comparison_report_{self.experiment_id}.md'
                }
            }
            
            with open(f'{self.results_dir}/complete_results.json', 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            # 10. 최종 요약 출력
            logger.info(" 확장된 비교 실험 완료!")
            logger.info("\n 최종 순위:")
            
            for i, (model_name, efficiency) in enumerate(statistical_analysis['ranking'], 1):
                model_display = {
                    'cruise_mode': '크루즈 모드',
                    'sac_scratch': 'SAC 순수학습',
                    'sac_lunarlander': 'SAC LunarLander 전이',
                    'sac_mountaincar': 'SAC MountainCar 전이'
                }.get(model_name, model_name)
                
                logger.info(f"  {i}. {model_display}: {efficiency:.3f} km/kWh")
            
            # 전이학습 비교 결과
            if ('sac_mountaincar' in statistical_analysis['model_means'] and 
                'sac_lunarlander' in statistical_analysis['model_means']):
                
                mountain_eff = statistical_analysis['model_means']['sac_mountaincar']['mean']
                lunar_eff = statistical_analysis['model_means']['sac_lunarlander']['mean']
                
                if mountain_eff > lunar_eff:
                    logger.info(f"\n🏆 최적 전이학습: MountainCar ({mountain_eff:.3f} > {lunar_eff:.3f})")
                else:
                    logger.info(f"\n🏆 최적 전이학습: LunarLander ({lunar_eff:.3f} > {mountain_eff:.3f})")
            
            logger.info(f"\n 생성된 파일들:")
            logger.info(f"   시각화: {self.results_dir}/comprehensive_4model_comparison.png")
            logger.info(f"  📈 통계분석: {self.results_dir}/detailed_statistical_analysis.png") 
            logger.info(f"  🔄 전이학습: {self.results_dir}/transfer_learning_analysis.png")
            logger.info(f"  📋 LaTeX 테이블: {self.results_dir}/publication_tables.tex")
            logger.info(f"  📄 보고서: extended_comparison_report_{self.experiment_id}.md")
            logger.info(f"  💾 전체결과: {self.results_dir}/complete_results.json")
            
            return final_results
            
        except Exception as e:
            logger.error(f"확장된 비교 실험 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_extended_report(self, statistical_analysis):
        """확장된 비교 보고서 생성 (시각화 포함)"""
        
        logger.info("📋 확장된 비교 보고서 생성...")
        
        report = f"""# 확장된 전이학습 비교 실험 보고서
## MountainCar vs LunarLander vs 순수학습 vs 크루즈 모드

### 🔬 실험 개요
- **실험 ID**: {self.experiment_id}
- **비교 모델**: 4개 (크루즈, 순수, LunarLander 전이, MountainCar 전이)
- **통계적 검정력**: 각 모델당 30 에피소드
- **신뢰수준**: 95%
- **데이터**: 통계적으로 검증된 증강 데이터 (972행 훈련, 216행 테스트)

### 🏆 성능 순위 (에너지 효율 기준)
"""
        
        for i, (model_name, efficiency) in enumerate(statistical_analysis['ranking'], 1):
            model_display = {
                'cruise_mode': '크루즈 모드',
                'sac_scratch': 'SAC 순수학습',
                'sac_lunarlander': 'SAC LunarLander 전이',
                'sac_mountaincar': 'SAC MountainCar 전이'
            }.get(model_name, model_name)
            
            # 신뢰구간 추가
            if model_name in self.results and 'confidence_interval_95' in self.results[model_name]:
                ci = self.results[model_name]['confidence_interval_95']
                ci_str = f" (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
            else:
                ci_str = ""
            
            report += f"{i}. **{model_display}**: {efficiency:.3f} km/kWh{ci_str}\n"
        
        report += "\n###  상세 성능 결과\n"
        
        for model_name, data in statistical_analysis['model_means'].items():
            model_display = {
                'cruise_mode': '크루즈 모드',
                'sac_scratch': 'SAC 순수학습', 
                'sac_lunarlander': 'SAC LunarLander 전이',
                'sac_mountaincar': 'SAC MountainCar 전이'
            }.get(model_name, model_name)
            
            report += f"- **{model_display}**: {data['mean']:.3f} ± {data['std']:.3f} km/kWh (n={data['n']})\n"
        
        report += "\n### 크루즈 모드 대비 개선율\n"
        
        for model, improvement in statistical_analysis['improvement_rates'].items():
            model_display = {
                'sac_scratch': 'SAC 순수학습',
                'sac_lunarlander': 'SAC LunarLander 전이', 
                'sac_mountaincar': 'SAC MountainCar 전이'
            }.get(model, model)
            
            status = ' 목표 달성' if improvement >= 20 else '목표 미달성'
            report += f"- **{model_display}**: {improvement:+.1f}% ({status})\n"
        
        report += "\n### 📈 전이학습 모델 비교\n"
        
        # LunarLander vs MountainCar 직접 비교
        mountain_vs_lunar_key = None
        for key in statistical_analysis['pairwise_comparisons']:
            if 'mountaincar' in key and 'lunarlander' in key:
                mountain_vs_lunar_key = key
                break
        
        if mountain_vs_lunar_key:
            comparison = statistical_analysis['pairwise_comparisons'][mountain_vs_lunar_key]
            
            winner = "MountainCar" if comparison['mean_difference'] < 0 else "LunarLander"
            significance = "통계적으로 유의" if comparison['t_test']['significant'] else "통계적으로 비유의"
            effect_size = abs(comparison['effect_size'])
            
            if effect_size >= 0.8:
                effect_desc = "대 효과"
            elif effect_size >= 0.5:
                effect_desc = "중 효과"
            elif effect_size >= 0.2:
                effect_desc = "소 효과"
            else:
                effect_desc = "무시할 수 있는 효과"
            
            report += f"- **최적 전이학습 모델**: {winner}\n"
            report += f"- **통계적 유의성**: {significance} (p = {comparison['t_test']['p_value']:.4f})\n"
            report += f"- **효과크기**: {effect_desc} (Cohen's d = {comparison['effect_size']:.3f})\n"
        
        # 학습 효율성 분석
        report += "\n### ⚡ 학습 효율성 분석\n"
        
        if 'sac_scratch' in statistical_analysis['model_means']:
            scratch_perf = statistical_analysis['model_means']['sac_scratch']['mean']
            
            report += f"- **순수학습 (100k 스텝)**: {scratch_perf:.3f} km/kWh\n"
            
            for model_key, steps in [('sac_lunarlander', 50000), ('sac_mountaincar', 50000)]:
                if model_key in statistical_analysis['model_means']:
                    model_perf = statistical_analysis['model_means'][model_key]['mean']
                    efficiency_per_step = (model_perf / steps) * 1000
                    model_name = 'LunarLander' if 'lunar' in model_key else 'MountainCar'
                    
                    report += f"- **{model_name} 전이 ({steps//1000}k 스텝)**: {model_perf:.3f} km/kWh "
                    report += f"(효율: {efficiency_per_step:.4f}/1k스텝)\n"
        
        report += f"\n###  통계적 검증 결과\n"
        
        # ANOVA 결과
        if statistical_analysis['anova_test']:
            anova = statistical_analysis['anova_test']
            significance = "유의함" if anova['significant'] else "비유의함"
            report += f"- **일원배치 분산분석**: F = {anova['f_statistic']:.3f}, "
            report += f"p = {anova['p_value']:.4f} ({significance})\n"
        
        # 주요 쌍별 비교
        key_comparisons = [
            ('sac_mountaincar_vs_sac_lunarlander', 'MountainCar vs LunarLander'),
            ('sac_scratch_vs_cruise_mode', '순수학습 vs 크루즈'),
        ]
        
        for comp_key, comp_name in key_comparisons:
            if comp_key in statistical_analysis['pairwise_comparisons']:
                comp = statistical_analysis['pairwise_comparisons'][comp_key]
                sig_status = "유의" if comp['t_test']['significant'] else "비유의"
                report += f"- **{comp_name}**: t = {comp['t_test']['statistic']:.3f}, "
                report += f"p = {comp['t_test']['p_value']:.4f} ({sig_status})\n"
        
        report += f"\n### 🎨 생성된 시각화\n"
        report += f"- **종합 성능 비교**: `{self.results_dir}/comprehensive_4model_comparison.png`\n"
        report += f"- **상세 통계 분석**: `{self.results_dir}/detailed_statistical_analysis.png`\n"
        report += f"- **전이학습 분석**: `{self.results_dir}/transfer_learning_analysis.png`\n"
        
        report += f"\n### 📋 논문용 자료\n"
        report += f"- **LaTeX 테이블**: `{self.results_dir}/publication_tables.tex`\n"
        report += f"- **CSV 데이터**: `{self.results_dir}/performance_comparison.csv`\n"
    
    def run_extended_comparison(self):
        """확장된 비교 실험 전체 실행"""
        
        logger.info("확장된 전이학습 비교 실험 시작!")
        logger.info("=" * 60)
        
        try:
            # 1. 이전 결과 로드
            if not self.load_previous_results():
                logger.error("이전 실험 결과가 필요합니다. 먼저 기본 실험을 완료하세요.")
                return None
            
            # 2. 환경 생성
            env = self.create_ev_environment()
            if not env:
                return None
            
            # 3. MountainCar 전이학습 실행
            mountaincar_model, mountaincar_info = self.train_mountaincar_transfer_model(env)
            
            if mountaincar_model:
                # 4. MountainCar 모델 평가
                mountaincar_results = self.evaluate_mountaincar_model(mountaincar_model, env)
                self.results['sac_mountaincar'] = mountaincar_results
            else:
                logger.error("MountainCar 모델 훈련 실패")
                return None
            
            # 5. 4개 모델 통계 비교
            statistical_analysis = self.perform_extended_statistical_comparison()
            
            if not statistical_analysis:
                return None
            
            # 6. 결과 저장
            final_results = {
                'experiment_id': self.experiment_id,
                'models_compared': list(self.results.keys()),
                'individual_results': self.results,
                'statistical_analysis': statistical_analysis,
                'mountaincar_transfer_info': mountaincar_info
            }
            
            with open(f'extended_comparison_results_{self.experiment_id}.json', 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            # 7. 보고서 생성
            report = self.generate_extended_report(statistical_analysis)
            
            # 8. 요약 출력
            logger.info(" 확장된 비교 실험 완료!")
            logger.info("\n 최종 순위:")
            
            for i, (model_name, efficiency) in enumerate(statistical_analysis['ranking'], 1):
                model_display = {
                    'cruise_mode': '크루즈 모드',
                    'sac_scratch': 'SAC 순수학습',
                    'sac_lunarlander': 'SAC LunarLander 전이',
                    'sac_mountaincar': 'SAC MountainCar 전이'
                }.get(model_name, model_name)
                
                logger.info(f"  {i}. {model_display}: {efficiency:.3f} km/kWh")
            
            return final_results
            
        except Exception as e:
            logger.error(f"확장된 비교 실험 실패: {e}")
            import traceback
            traceback.print_exc()
            return None


# 실행 함수
def run_extended_transfer_comparison():
    """확장된 전이학습 비교 실행"""
    
    print(" 확장된 전이학습 비교 실험 시작!")
    print(" 비교 모델: 크루즈 + 순수학습 + LunarLander 전이 + MountainCar 전이")
    print("목표: 최적의 전이학습 모델 식별 및 종합 성능 분석")
    print("=" * 70)
    
    comparison = ExtendedTransferLearningComparison()
    results = comparison.run_extended_comparison()
    
    if results:
        print("\n 확장된 비교 실험 성공!")
        print(" 4개 모델 종합 분석 완료")
        print(" 최적 전이학습 모델 식별")
        print(" 데이터 확보")
        print("\n 생성된 파일:")
        print(f"  - extended_comparison_results_{comparison.experiment_id}.json")
        print(f"  - extended_comparison_report_{comparison.experiment_id}.md")
    else:
        print("\n확장된 비교 실험 실패")
        print("기본 실험 완료 후 재시도하세요")
    
    return results


if __name__ == "__main__":
    # 즉시 실행
    results = run_extended_transfer_comparison()
