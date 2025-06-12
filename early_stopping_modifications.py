# 1. PerformanceMetricsCallback 클래스 수정 - 수렴 검사 기능 추가

class PerformanceMetricsCallback(BaseCallback):
    """실시간 성능 추적 및 조기 종료"""
    
    def __init__(self, eval_env, eval_freq=2000, verbose=0, 
                 convergence_threshold=0.9, patience=10, min_episodes=50):
        super(PerformanceMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        
        # 조기 종료 관련 파라미터
        self.convergence_threshold = convergence_threshold  # 90% 수렴 기준
        self.patience = patience  # 연속으로 개선되지 않는 에피소드 수
        self.min_episodes = min_episodes  # 최소 학습 에피소드
        
        # 수렴 추적 변수
        self.best_performance = -np.inf
        self.performance_history = []
        self.episodes_without_improvement = 0
        self.converged = False
        
        self.metrics_history = {
            'episode': [],
            'timestep': [],
            'energy_efficiency': [],
            'soc_decrease_rate': [],
            'speed_tracking_rate': [],
            'target_speed_proximity': [],
            'comfort_score': [],
            'safety_violations': [],
            'convergence_reward': [],
            'learning_stability': [],
            'performance_score': []  # 종합 성능 점수
        }
        self.episode_rewards = []
        self.last_episode_count = 0
    
    def _calculate_performance_score(self, metrics):
        """종합 성능 점수 계산 (0~1 범위)"""
        # 에너지 효율성 (40% 가중치)
        efficiency_score = min(metrics.get('energy_efficiency', 1.0) / 10.0, 1.0) * 0.4
        
        # 속도 추종률 (30% 가중치)  
        speed_score = (metrics.get('speed_tracking_rate', 0) / 100.0) * 0.3
        
        # SOC 관리 (20% 가중치) - 낮을수록 좋음
        soc_rate = metrics.get('soc_decrease_rate', 100)
        soc_score = max(0, (100 - soc_rate) / 100.0) * 0.2
        
        # 안전성 (10% 가중치)
        safety_violations = metrics.get('safety_violations', 10)
        safety_score = max(0, (10 - safety_violations) / 10.0) * 0.1
        
        total_score = efficiency_score + speed_score + soc_score + safety_score
        return min(1.0, max(0.0, total_score))
    
    def _check_convergence(self):
        """수렴 조건 검사"""
        if len(self.performance_history) < self.min_episodes:
            return False
        
        # 최근 성능의 안정성 검사
        recent_scores = self.performance_history[-10:]  # 최근 10개 평가
        if len(recent_scores) < 10:
            return False
        
        # 표준편차가 작고 평균 성능이 높으면 수렴으로 판단
        mean_performance = np.mean(recent_scores)
        std_performance = np.std(recent_scores)
        
        # 수렴 조건:
        # 1. 평균 성능이 threshold 이상
        # 2. 성능의 변동성이 작음 (std < 0.05)
        # 3. 최근 성능 개선이 없음
        performance_stable = std_performance < 0.05
        performance_high = mean_performance >= self.convergence_threshold
        
        if performance_stable and performance_high:
            logger.info(f"수렴 감지: 평균성능={mean_performance:.3f}, 안정성={std_performance:.4f}")
            return True
        
        return False
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            try:
                # 환경에서 메트릭 수집
                if hasattr(self.training_env, 'get_attr'):
                    current_metrics_list = self.training_env.get_attr('get_current_metrics')
                    episode_counts = self.training_env.get_attr('episode_count')
                    
                    if current_metrics_list and episode_counts:
                        current_metrics = current_metrics_list[0]() if callable(current_metrics_list[0]) else current_metrics_list[0]
                        current_episode = episode_counts[0]
                    else:
                        current_metrics = {'energy_efficiency': 4.0}
                        current_episode = 0
                else:
                    if hasattr(self.training_env, 'get_current_metrics'):
                        current_metrics = self.training_env.get_current_metrics()
                    else:
                        current_metrics = {'energy_efficiency': 4.0}
                    
                    if hasattr(self.training_env, 'episode_count'):
                        current_episode = self.training_env.episode_count
                    else:
                        current_episode = 0
                
                # 성능 점수 계산
                performance_score = self._calculate_performance_score(current_metrics)
                self.performance_history.append(performance_score)
                
                # 효율값 로그 출력
                efficiency = current_metrics.get('energy_efficiency', 4.0)
                logger.info(f"Step {self.n_calls}: 성능점수={performance_score:.3f}, 효율={efficiency:.3f} km/kWh")
                
                # 새 에피소드 완료시만 기록
                if current_episode > self.last_episode_count:
                    self.last_episode_count = current_episode
                    
                    # 성능 개선 추적
                    if performance_score > self.best_performance:
                        self.best_performance = performance_score
                        self.episodes_without_improvement = 0
                        logger.info(f" 새로운 최고 성능: {performance_score:.3f}")
                    else:
                        self.episodes_without_improvement += 1
                    
                    # 메트릭 기록
                    self.metrics_history['timestep'].append(self.n_calls)
                    self.metrics_history['episode'].append(current_episode)
                    self.metrics_history['energy_efficiency'].append(efficiency)
                    self.metrics_history['performance_score'].append(performance_score)
                    self.metrics_history['soc_decrease_rate'].append(
                        current_metrics.get('soc_decrease_rate', 15.0)
                    )
                    self.metrics_history['speed_tracking_rate'].append(
                        current_metrics.get('speed_tracking_rate', 85.0)
                    )
                    
                    # 수렴 검사
                    if self._check_convergence():
                        logger.info(" 90% 수렴 달성! 학습을 조기 종료합니다.")
                        self.converged = True
                        return False  # 학습 중단
                    
                    # Patience 체크
                    if self.episodes_without_improvement >= self.patience:
                        logger.info(f" {self.patience}회 연속 개선 없음. 조기 종료 검토...")
                        if len(self.performance_history) >= self.min_episodes:
                            recent_performance = np.mean(self.performance_history[-5:])
                            if recent_performance >= 0.8:  # 80% 이상이면 종료
                                logger.info(" 충분한 성능 달성으로 조기 종료합니다.")
                                return False
                        
            except Exception as e:
                logger.warning(f"Metrics collection failed at step {self.n_calls}: {e}")
                
        return True


# 2. train_sac_model 함수 수정 - 조기 종료 콜백 적용

def train_sac_model(model_name, is_transfer_learning=False, total_timesteps=100000, 
                   data_dir="./data", save_dir="./models", 
                   enable_early_stopping=True, convergence_threshold=0.9):
    """SAC 모델 훈련 - 조기 종료 기능 포함"""
    
    logger.info(f" SAC 모델 훈련 시작: {model_name}")
    logger.info(f"전이학습: {is_transfer_learning}")
    logger.info(f"총 스텝: {total_timesteps}")
    logger.info(f"조기 종료: {enable_early_stopping} (임계값: {convergence_threshold})")
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # 환경 생성
    env = EVEnergyEnvironmentPreprocessed(data_dir=data_dir)
    
    # SageMaker 최적화된 SAC 설정
    sac_config = {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'verbose': 1,
        'device': 'cuda' if hasattr(torch, 'cuda') and torch.cuda.is_available() else 'cpu'
    }
    
    # 모델 생성 (기존 코드와 동일)
    if is_transfer_learning:
        logger.info("전이학습 모델 로드 시도...")
        
        if install_transfer_learning_dependencies():
            try:
                from huggingface_sb3 import load_from_hub
                
                pretrained_models = [
                    ("sb3/sac-LunarLanderContinuous-v2", "sac-LunarLanderContinuous-v2.zip"),
                    ("sb3/sac-BipedalWalker-v3", "sac-BipedalWalker-v3.zip"),
                    ("sb3/sac-Pendulum-v1", "sac-Pendulum-v1.zip")
                ]
                
                model = None
                for repo_id, filename in pretrained_models:
                    try:
                        logger.info(f"🔍 {repo_id} 모델 로드 시도...")
                        checkpoint = load_from_hub(repo_id=repo_id, filename=filename)
                        temp_model = SAC.load(checkpoint)
                        
                        model = SAC(policy=temp_model.policy_class, env=env, **sac_config)
                        
                        try:
                            model.policy.load_state_dict(temp_model.policy.state_dict(), strict=False)
                            logger.info(f" {repo_id} 모델에서 전이학습 성공!")
                            break
                        except:
                            logger.warning(f" {repo_id} 파라미터 복사 실패, 다음 모델 시도")
                            continue
                            
                    except Exception as e:
                        logger.warning(f" {repo_id} 로드 실패: {e}")
                        continue
                
                if model is None:
                    logger.warning(" 전이학습 모델 로드 실패, 순수 학습으로 진행")
                    model = SAC('MlpPolicy', env, **sac_config)
                    
            except ImportError:
                logger.warning(" huggingface_sb3 설치 실패, 순수 학습으로 진행")
                model = SAC('MlpPolicy', env, **sac_config)
            except Exception as e:
                logger.warning(f" 전이학습 실패: {e}, 순수 학습으로 진행")
                model = SAC('MlpPolicy', env, **sac_config)
        else:
            logger.warning(" 전이학습 라이브러리 설치 실패, 순수 학습으로 진행")
            model = SAC('MlpPolicy', env, **sac_config)
    else:
        logger.info("순수 학습 모델 생성")
        model = SAC('MlpPolicy', env, **sac_config)
    
    # 조기 종료 콜백 설정
    if enable_early_stopping:
        eval_callback = PerformanceMetricsCallback(
            eval_env=env,
            eval_freq=2000,
            convergence_threshold=convergence_threshold,
            patience=10,
            min_episodes=20,  # 최소 20 에피소드는 학습
            verbose=1
        )
        logger.info(f" 조기 종료 활성화: 임계값={convergence_threshold}, patience=10")
    else:
        eval_callback = PerformanceMetricsCallback(eval_env=env, eval_freq=2000, verbose=1)
        logger.info(" 고정 timesteps로 학습")
    
    # 훈련 시작
    logger.info(f" 학습 시작 - 최대: {total_timesteps} 스텝")
    start_time = datetime.now()
    
    actual_timesteps_trained = 0
    early_stopped = False
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=100,
            progress_bar=True
        )
        
        # 실제 학습된 스텝 수 확인
        actual_timesteps_trained = model.num_timesteps
        early_stopped = getattr(eval_callback, 'converged', False)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        if early_stopped:
            logger.info(f" 조기 종료로 학습 완료! 실제 스텝: {actual_timesteps_trained}/{total_timesteps}")
            logger.info(f" 소요시간: {training_time:.1f}초 (예상 시간 대비 단축)")
        else:
            logger.info(f" 전체 학습 완료! 소요시간: {training_time:.1f}초")
        
    except KeyboardInterrupt:
        logger.info(" 사용자 중단")
    except Exception as e:
        logger.error(f" 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 모델 저장
    model_path = f"{save_dir}/{model_name}.zip"
    model.save(model_path)
    logger.info(f" 모델 저장: {model_path}")
    
    # 성능 평가 (기존 코드와 동일)
    logger.info(" 최종 성능 평가 중...")
    eval_results, eval_episodes = evaluate_policy(
        model, env, n_eval_episodes=50, return_episode_rewards=True
    )
    
    # 메트릭 수집
    final_metrics = []
    for _ in range(50):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        final_metrics.append(env.get_episode_metrics())
    
    # 결과 정리 - 조기 종료 정보 추가
    results = {
        'model_name': model_name,
        'is_transfer_learning': is_transfer_learning,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'actual_timesteps': actual_timesteps_trained,  # 실제 학습 스텝
        'early_stopped': early_stopped,  # 조기 종료 여부
        'convergence_threshold': convergence_threshold,
        'time_efficiency': (total_timesteps - actual_timesteps_trained) / total_timesteps if early_stopped else 0,
        'eval_mean_reward': np.mean(eval_results),
        'eval_std_reward': np.std(eval_results),
        'metrics': {
            'energy_efficiency': {
                'mean': np.mean([m['energy_efficiency'] for m in final_metrics]),
                'std': np.std([m['energy_efficiency'] for m in final_metrics]),
                'values': [m['energy_efficiency'] for m in final_metrics]
            },
            'speed_tracking_rate': {
                'mean': np.mean([m['speed_tracking_rate'] for m in final_metrics]),
                'std': np.std([m['speed_tracking_rate'] for m in final_metrics]),
                'values': [m['speed_tracking_rate'] for m in final_metrics]
            },
            'soc_decrease_rate': {
                'mean': np.mean([m['soc_decrease_rate'] for m in final_metrics]),
                'std': np.std([m['soc_decrease_rate'] for m in final_metrics]),
                'values': [m['soc_decrease_rate'] for m in final_metrics]
            }
        },
        'learning_history': eval_callback.metrics_history,
        'convergence_info': {
            'performance_history': getattr(eval_callback, 'performance_history', []),
            'best_performance': getattr(eval_callback, 'best_performance', 0),
            'episodes_without_improvement': getattr(eval_callback, 'episodes_without_improvement', 0)
        }
    }
    
    # 결과 저장 (기존 코드와 동일하게 JSON 직렬화)
    results_path = f"./results/{model_name}_results.json"
    with open(results_path, 'w') as f:
        def serialize_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            else:
                return obj
        
        json.dump(serialize_for_json(results), f, indent=2)
    
    logger.info(f" 결과 저장: {results_path}")
    
    if early_stopped:
        time_saved = (total_timesteps - actual_timesteps_trained) / total_timesteps * 100
        logger.info(f" 조기 종료로 {time_saved:.1f}% 시간 절약!")
    
    return model, results


# 3. main 함수에 조기 종료 옵션 추가

def main():
    """메인 실행 함수 - 조기 종료 기능 포함"""
    parser = argparse.ArgumentParser(description='SAC 전기차 에너지 효율 최적화 - 조기 종료 기능')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='데이터 디렉토리 경로')
    parser.add_argument('--timesteps_scratch', type=int, default=100000,
                       help='순수학습 최대 스텝 수')
    parser.add_argument('--timesteps_transfer', type=int, default=50000,
                       help='전이학습 최대 스텝 수')
    parser.add_argument('--mode', type=str, choices=['compare', 'train_scratch', 'train_transfer'], 
                       default='compare', help='실행 모드')
    parser.add_argument('--aws_instance', type=str, default='ml.m5.xlarge',
                       help='AWS SageMaker 인스턴스 타입')
    
    # 조기 종료 관련 옵션 추가
    parser.add_argument('--enable_early_stopping', action='store_true', default=True,
                       help='조기 종료 활성화 (기본: True)')
    parser.add_argument('--convergence_threshold', type=float, default=0.9,
                       help='수렴 임계값 (기본: 0.9 = 90%)')
    parser.add_argument('--patience', type=int, default=10,
                       help='개선 없이 기다릴 에피소드 수 (기본: 10)')
    
    args = parser.parse_args()
    
    # SageMaker 환경 정보 로깅
    logger.info(" SageMaker 최적화 SAC 실험 시작 (조기 종료 기능 포함)")
    logger.info(f"인스턴스 타입: {args.aws_instance}")
    logger.info(f"데이터 디렉토리: {args.data_dir}")
    logger.info(f"실행 모드: {args.mode}")
    logger.info(f"조기 종료: {args.enable_early_stopping} (임계값: {args.convergence_threshold})")
    
    if args.mode == 'compare':
        # compare_models_and_baseline 함수도 수정해야 함 (train_sac_model 호출 부분)
        results = compare_models_and_baseline_with_early_stopping(
            enable_early_stopping=args.enable_early_stopping,
            convergence_threshold=args.convergence_threshold
        )
        
    elif args.mode == 'train_scratch':
        model, results = train_sac_model(
            model_name="sac_from_scratch",
            is_transfer_learning=False,
            total_timesteps=args.timesteps_scratch,
            data_dir=args.data_dir,
            enable_early_stopping=args.enable_early_stopping,
            convergence_threshold=args.convergence_threshold
        )
        
    elif args.mode == 'train_transfer':
        model, results = train_sac_model(
            model_name="sac_with_transfer", 
            is_transfer_learning=True,
            total_timesteps=args.timesteps_transfer,
            data_dir=args.data_dir,
            enable_early_stopping=args.enable_early_stopping,
            convergence_threshold=args.convergence_threshold
        )
    
    logger.info(" SageMaker 실험 완료")


# 4. compare_models_and_baseline 함수도 수정 필요

def compare_models_and_baseline_with_early_stopping(enable_early_stopping=True, convergence_threshold=0.9):
    """모델 성능 비교 및 분석 - 조기 종료 기능 포함"""
    
    logger.info("=" * 60)
    logger.info("SAC 전기차 에너지 효율 최적화 실험 시작 (조기 종료 기능 포함)")
    logger.info("=" * 60)
    
    # 기존 코드와 동일한 기준선 평가...
    
    # SAC 모델 훈련 부분만 수정
    sac_scratch_model, sac_scratch_results = train_sac_model(
        model_name="sac_from_scratch",
        is_transfer_learning=False,
        total_timesteps=100000,
        data_dir="./data",
        enable_early_stopping=enable_early_stopping,
        convergence_threshold=convergence_threshold
    )
    
    sac_transfer_model, sac_transfer_results = train_sac_model(
        model_name="sac_with_transfer",
        is_transfer_learning=True,
        total_timesteps=50000,
        data_dir="./data",
        enable_early_stopping=enable_early_stopping,
        convergence_threshold=convergence_threshold
    )
    
    # 나머지는 기존 코드와 동일...
    
    return comparison_results