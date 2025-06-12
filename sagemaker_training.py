import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import glob
from datetime import datetime
import logging
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# shimmy 임포트
try:
    import shimmy
    SHIMMY_AVAILABLE = True
except ImportError:
    SHIMMY_AVAILABLE = False
    print("shimmy가 설치되지 않음 - pip install shimmy 실행 필요")
    

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMetricsCallback(BaseCallback):
    """실시간 성능 추적"""
    
    def __init__(self, eval_env, eval_freq=2000, verbose=0):
        super(PerformanceMetricsCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
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
            'learning_stability': []
        }
        self.episode_rewards = []
        self.last_episode_count = 0
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            try:
                # 환경이 벡터화된 환경인지 확인
                if hasattr(self.training_env, 'get_attr'):
                    # 벡터화된 환경의 경우
                    current_metrics_list = self.training_env.get_attr('get_current_metrics')
                    episode_counts = self.training_env.get_attr('episode_count')
                    
                    # 첫 번째 환경의 메트릭 사용
                    if current_metrics_list and episode_counts:
                        current_metrics = current_metrics_list[0]() if callable(current_metrics_list[0]) else current_metrics_list[0]
                        current_episode = episode_counts[0]
                    else:
                        current_metrics = {'energy_efficiency': 4.0}
                        current_episode = 0
                        
                else:
                    # 단일 환경의 경우
                    if hasattr(self.training_env, 'get_current_metrics'):
                        current_metrics = self.training_env.get_current_metrics()
                    else:
                        current_metrics = {'energy_efficiency': 4.0}
                    
                    if hasattr(self.training_env, 'episode_count'):
                        current_episode = self.training_env.episode_count
                    else:
                        current_episode = 0
                
                # 실제 효율값 로그 출력
                efficiency = current_metrics.get('energy_efficiency', 4.0)
                logger.info(f"Step {self.n_calls}: Energy Efficiency = {efficiency:.3f} km/kWh")
                
                # 새 에피소드 완료시만 기록
                if current_episode > self.last_episode_count:
                    self.last_episode_count = current_episode
                    
                    self.metrics_history['timestep'].append(self.n_calls)
                    self.metrics_history['episode'].append(current_episode)
                    self.metrics_history['energy_efficiency'].append(efficiency)
                    self.metrics_history['soc_decrease_rate'].append(
                        current_metrics.get('soc_decrease_rate', 15.0)
                    )
                    self.metrics_history['speed_tracking_rate'].append(
                        current_metrics.get('speed_tracking_rate', 85.0)
                    )
                    
            except Exception as e:
                logger.warning(f"Metrics collection failed at step {self.n_calls}: {e}")
                
        return True


class EVEnergyEnvironmentPreprocessed(gym.Env):
    """전처리된 데이터 활용 전기차 에너지 효율 최적화 환경"""
    
    def __init__(self, data_dir, config=None):
        super(EVEnergyEnvironmentPreprocessed, self).__init__()
        
        # 환경 설정
        self.data_dir = data_dir
        self.config = config or {}
        
        # 상태 공간: 28차원 (확인됨)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        
        # 행동 공간: 가속도 [-3.0, 3.0] m/s²
        self.action_space = spaces.Box(
            low=-3.0, high=3.0, shape=(1,), dtype=np.float32
        )
        
        # 아이오닉5 차량 제원 (실제 스펙)
        self.vehicle_specs = {
            'mass': 2050,  # kg
            'battery_capacity': 77.4,  # kWh
            'motor_max_torque': 350,  # Nm
            'drag_coefficient': 0.28,
            'frontal_area': 2.8,  # m²
            'wheel_radius': 0.35,  # m
            'final_drive_ratio': 7.4,
            'motor_efficiency': 0.95,
            'battery_efficiency': 0.95,
            'regen_efficiency': 0.80
        }
        
        # 도로 저항 계수 (논문 Table 1 기준)
        self.road_resistance = {
            'f0': 53.90,  # N
            'f1': 0.21,   # N⋅s/m
            'f2': 0.02    # N⋅s²/m²
        }
        
        # 성능 추적 변수
        self.episode_data = {
            'speeds': [],
            'accelerations': [],
            'energy_consumption': [],
            'soc_changes': [],
            'rewards': [],
            'rush_periods': [],
            'traffic_conditions': [],
            'distances': [],
            'elevation_changes': []
        }
        
        # 현재 상태
        self.current_data_idx = 0
        self.current_speed = 30.0  # km/h
        self.current_soc = 0.8
        self.step_count = 0
        self.total_distance = 0.0  # km
        self.total_energy_consumed = 0.0  # kWh
        
        # 목표 속도 (평일 서울 도심 출퇴근 평균 속도)
        self.target_speed = 30.0  # km/h
        
        # 에피소드 카운터
        self.episode_count = 0
        
        self._load_preprocessed_data()
        self.reset()

    def get_current_metrics(self):
        """현재 메트릭 반환"""
        # 실제 효율 계산 (학습 진행 확인 가능하도록 수정됨)
        energy_efficiency = self._calculate_current_efficiency()
        
        # SOC 감소율
        initial_soc = 0.8
        soc_decrease_rate = ((initial_soc - self.current_soc) / initial_soc) * 100
        
        # 속도 추종률
        if len(self.episode_data['speeds']) > 0:
            speeds = np.array(self.episode_data['speeds'])
            in_range = np.abs(speeds - self.target_speed) <= 5
            speed_tracking_rate = np.mean(in_range) * 100
            target_speed_proximity = np.mean(np.abs(speeds - self.target_speed))
        else:
            speed_tracking_rate = max(0, 100 - abs(self.current_speed - self.target_speed) * 5)
            target_speed_proximity = abs(self.current_speed - self.target_speed)
        
        return {
            'energy_efficiency': energy_efficiency,
            'soc_decrease_rate': soc_decrease_rate,
            'speed_tracking_rate': speed_tracking_rate,
            'target_speed_proximity': target_speed_proximity,
            'total_distance': self.total_distance,
            'total_energy_consumed': self.total_energy_consumed
        }

    def _parse_state_vector(self, vector_str):
        """상태 벡터 문자열을 리스트로 변환 - 견고하게 수정"""
        try:
            if isinstance(vector_str, str):
                # 따옴표 제거
                clean_str = vector_str.strip('"\'')
                # 대괄호 내부 추출
                if '[' in clean_str and ']' in clean_str:
                    start = clean_str.find('[')
                    end = clean_str.find(']') + 1
                    vector_part = clean_str[start:end]
                    # 대괄호 제거하고 파싱
                    content = vector_part.strip('[]')
                    return [float(x.strip()) for x in content.split(',')]
                else:
                    # 대괄호 없이 쉼표로 구분된 경우
                    return [float(x.strip()) for x in clean_str.split(',')]
            elif isinstance(vector_str, list):
                return vector_str
            else:
                logger.warning(f"알 수 없는 상태 벡터 형식: {type(vector_str)}")
                return [0.0] * 28
        except Exception as e:
            logger.warning(f"상태 벡터 파싱 오류: {e}")
            return [0.0] * 28

    def _get_default_normalization(self):
        """기본 정규화 범위"""
        return {
            'morning_rush': {'traffic_volume': {'min': 1446, 'max': 1764}},
            'evening_rush': {'traffic_volume': {'min': 1896, 'max': 2130}},
            'weather': {
                'temperature': {'min': -17.2, 'max': 35.8},
                'humidity': {'min': 15.0, 'max': 100.0},
                'wind_speed': {'min': 0.0, 'max': 8.1},
                'precipitation': {'min': 0.0, 'max': 34.7}
            }
        }
    
    def _load_preprocessed_data(self):
        """전처리된 데이터 로드 - SageMaker 경로 최적화"""
        try:
            # SageMaker 환경에서의 데이터 경로
            train_patterns = [
                f"{self.data_dir}/train_statistically_valid_*.csv",
                f"{self.data_dir}/train_statistically_valid_*.csv", 
                f"{self.data_dir}/*train*.csv",
                f"{self.data_dir}/*.csv"
            ]
            
            test_patterns = [
                f"{self.data_dir}/test_statistically_valid_*.csv",
                f"{self.data_dir}/test_statistically_valid_*.csv",
                f"{self.data_dir}/*test*.csv"
            ]
            
            norm_patterns = [
                f"{self.data_dir}/rush_normalization_corrected_*.json",
                f"{self.data_dir}/rush_normalization*.json"
            ]
            
            # 훈련 데이터 찾기
            train_files = []
            for pattern in train_patterns:
                files = glob.glob(pattern)
                if files:
                    train_files = files
                    logger.info(f"훈련 데이터 발견: {pattern}")
                    break
            
            # 테스트 데이터 찾기  
            test_files = []
            for pattern in test_patterns:
                files = glob.glob(pattern)
                if files:
                    test_files = files
                    logger.info(f" 테스트 데이터 발견: {pattern}")
                    break
                    
            # 정규화 파일 찾기
            norm_files = []
            for pattern in norm_patterns:
                files = glob.glob(pattern)
                if files:
                    norm_files = files
                    logger.info(f" 정규화 데이터 발견: {pattern}")
                    break
            
            if train_files:
                self.train_data = pd.read_csv(train_files[0])
                logger.info(f" 훈련 데이터 로드 완료: {len(self.train_data)}행 × {len(self.train_data.columns)}열")
                
                if 'state_vector_rush' in self.train_data.columns:
                    logger.info(" state_vector_rush 컬럼 존재 - 파싱 시작")
                    self.train_data['state_vector_rush'] = self.train_data['state_vector_rush'].apply(
                        self._parse_state_vector
                    )
                    
                    # 차원 검증
                    sample_state = self.train_data['state_vector_rush'].iloc[0]
                    actual_dim = len(sample_state)
                    logger.info(f" 상태 벡터 차원 확인: {actual_dim}차원")
                    
                    if actual_dim != 28:
                        logger.warning(f" 예상 차원(28)과 실제 차원({actual_dim})이 다름")
                        # 관찰 공간 동적 조정
                        self.observation_space = spaces.Box(
                            low=-np.inf, high=np.inf, shape=(actual_dim,), dtype=np.float32
                        )
                    
                    logger.info(" state_vector_rush 파싱 완료")
                else:
                    logger.warning("state_vector_rush 컬럼이 없음, 기본 상태벡터 생성")
                    self._create_basic_state_vectors()
                
            else:
                logger.error(" 훈련 데이터 파일을 찾을 수 없음")
                self._create_dummy_data()
                return
            
            if test_files:
                self.test_data = pd.read_csv(test_files[0])
                logger.info(f" 테스트 데이터 로드 완료: {len(self.test_data)}행")
                
                if 'state_vector_rush' in self.test_data.columns:
                    self.test_data['state_vector_rush'] = self.test_data['state_vector_rush'].apply(
                        self._parse_state_vector
                    )
            
            # 정규화 범위 로드
            if norm_files:
                with open(norm_files[0], 'r') as f:
                    self.normalization_ranges = json.load(f)
                logger.info(" 정규화 범위 로드 완료")
            else:
                self.normalization_ranges = self._get_default_normalization()
                logger.info("기본 정규화 범위 사용")
                
            # 데이터 요약 출력
            logger.info("=" * 50)
            logger.info("데이터 로드 완료 요약:")
            logger.info(f"  훈련 데이터: {len(self.train_data) if hasattr(self, 'train_data') else 0}행")
            logger.info(f"  테스트 데이터: {len(self.test_data) if hasattr(self, 'test_data') else 0}행")
            logger.info(f"  주요 컬럼: {list(self.train_data.columns)[:10] if hasattr(self, 'train_data') else 'None'}...")
            logger.info("=" * 50)
                
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            self._create_dummy_data()
    
    def _create_basic_state_vectors(self):
        """기본 상태 벡터 생성"""
        logger.info("기본 상태 벡터 생성 중...")
        
        def create_state_vector(row):
            return [
                np.random.uniform(0, 1),  # 시간
                np.random.uniform(0, 1),  # 월
                np.random.uniform(0, 1),  # 교통량
                np.random.uniform(-1, 1), np.random.uniform(-1, 1),  # 시간 주기성
                np.random.uniform(-1, 1), np.random.uniform(-1, 1),  # 월 주기성
                np.random.randint(0, 2), np.random.randint(0, 2),     # 방향
                0, 0, 1,  # 더미
                np.random.randint(0, 2), np.random.randint(0, 2),     # 출퇴근
                np.random.randint(0, 2), np.random.randint(0, 2),     # 시간대
                np.random.uniform(0, 1),  # 온도
                np.random.uniform(0, 1),  # 습도
                np.random.uniform(0, 1),  # 풍속
                np.random.uniform(0, 1),  # 강수량
                np.random.uniform(-0.1, 0.1),  # 경사도
                1,  # visibility
                np.random.randint(0, 2),  # 강수 여부
                1.0,  # difficulty
                np.random.randint(0, 2), np.random.randint(0, 2),  # 출퇴근 세부
                0.8,  # SOC
                0.5,  # 속도
                0.0   # 추가
            ]
        
        self.train_data['state_vector_rush'] = [create_state_vector(None) for _ in range(len(self.train_data))]
    
    def _create_dummy_data(self):
        """더미 데이터 생성"""
        logger.info("더미 데이터 생성 중...")
        
        dummy_data = []
        for i in range(1000):
            state_vector = [np.random.uniform(-1, 1) for _ in range(28)]
            
            dummy_data.append({
                'state_vector_rush': state_vector,
                'rush_separated_reward': np.random.uniform(0.3, 0.9),
                'rush_period_detailed': np.random.choice(['morning_rush', 'evening_rush']),
                'traffic_volume': np.random.uniform(1000, 2000),
                'hour': np.random.randint(7, 21),
                'month': np.random.randint(1, 13),
                'temperature': np.random.uniform(-5, 30),
                'road_avg_gradient': np.random.uniform(-0.05, 0.05)
            })
        
        self.train_data = pd.DataFrame(dummy_data)
    
    def step(self, action):
        acceleration = action[0]
        
        # 현재 데이터 행 가져오기
        if self.current_data_idx < len(self.train_data):
            current_row = self.train_data.iloc[self.current_data_idx]
        else:
            current_row = self.train_data.sample(1).iloc[0]
        
        # 물리 모델 시뮬레이션
        old_speed_kmh = self.current_speed
        old_speed_ms = old_speed_kmh / 3.6
        
        # 가속도 적용 (dt = 1초)
        new_speed_ms = max(0, old_speed_ms + acceleration * 1.0)
        new_speed_kmh = new_speed_ms * 3.6
        
        # 이동 거리 계산
        avg_speed_ms = (old_speed_ms + new_speed_ms) / 2
        distance_step = avg_speed_ms * 1.0 / 1000  # km
        self.total_distance += distance_step
        
        # 에너지 소비 계산
        energy_consumption = self._calculate_energy_consumption(old_speed_ms, new_speed_ms, acceleration)
        
        # SOC 업데이트
        soc_decrease = energy_consumption / self.vehicle_specs['battery_capacity']
        new_soc = max(0, self.current_soc - soc_decrease)
        
        # 상태 업데이트
        self.current_speed = new_speed_kmh
        self.current_soc = new_soc
        self.total_energy_consumed += energy_consumption
        
        # 에피소드 데이터 수집
        self.episode_data['speeds'].append(new_speed_kmh)
        self.episode_data['accelerations'].append(acceleration)
        self.episode_data['energy_consumption'].append(energy_consumption)
        self.episode_data['soc_changes'].append(soc_decrease)
        
        # 상태 벡터 설정
        if isinstance(current_row['state_vector_rush'], list):
            self.state = np.array(current_row['state_vector_rush'], dtype=np.float32)
        else:
            self.state = np.random.uniform(-1, 1, 28).astype(np.float32)
        
        # 동적 요소 업데이트
        if len(self.state) >= 28:
            self.state[25] = new_soc  # SOC 업데이트
            self.state[26] = new_speed_kmh / 120.0  # 속도 정규화
        
        # 보상 계산
        reward = self._calculate_reward(acceleration, energy_consumption, new_speed_kmh, new_soc)
        self.episode_data['rewards'].append(reward)
        
        # 종료 조건
        self.step_count += 1
        self.current_data_idx += 1
        
        terminated = self.total_distance >= 7.816 or new_soc <= 0.05 or self.step_count >= 500
        truncated = False
        
        info = {
            'current_speed': self.current_speed,
            'current_soc': self.current_soc,
            'energy_efficiency': self._calculate_current_efficiency(),
            'step_count': self.step_count,
            'total_distance': self.total_distance,
            'total_energy_consumed': self.total_energy_consumed
        }
        
        return self.state.copy(), reward, terminated, truncated, info
    
    def _calculate_energy_consumption(self, old_speed_ms, new_speed_ms, acceleration):
        """에너지 소비 계산"""
        g = 9.81
        air_density = 1.225
        avg_speed_ms = (old_speed_ms + new_speed_ms) / 2
        
        # 저항력 계산
        rolling_resistance = (self.road_resistance['f0'] + 
                            self.road_resistance['f1'] * avg_speed_ms + 
                            self.road_resistance['f2'] * avg_speed_ms**2)
        
        drag_force = 0.5 * air_density * self.vehicle_specs['drag_coefficient'] * \
                    self.vehicle_specs['frontal_area'] * avg_speed_ms**2
        
        inertial_force = self.vehicle_specs['mass'] * acceleration
        
        total_force = rolling_resistance + drag_force + inertial_force
        power_required = total_force * avg_speed_ms
        
        # 회생 제동 고려
        if power_required < 0 and new_speed_ms > 5:
            power_required *= -self.vehicle_specs['regen_efficiency']
        
        # 효율 적용
        if power_required > 0:
            battery_power = power_required / (self.vehicle_specs['motor_efficiency'] * 
                                            self.vehicle_specs['battery_efficiency'])
        else:
            battery_power = power_required * self.vehicle_specs['regen_efficiency']
        
        energy_consumption = abs(battery_power) / 3600000  # kWh
        return max(0.0001, energy_consumption)
    
    def _calculate_reward(self, acceleration, energy_consumption, speed, soc):
        """보상 함수"""
        # 에너지 효율성 보상
        current_efficiency = self._calculate_current_efficiency()
        baseline_efficiency = 4.2
        efficiency_reward = (current_efficiency - baseline_efficiency) / baseline_efficiency * 0.6
        
        # 속도 추종 보상
        speed_error = abs(speed - self.target_speed)
        speed_reward = max(0, (10 - speed_error) / 10) * 0.3
        
        # 주행 안전성
        comfort_penalty = min(abs(acceleration) / 3.0, 1.0) * 0.1
        
        # SOC 관리
        soc_penalty = 0.2 if soc < 0.2 else 0
        
        total_reward = 0.7 + efficiency_reward + speed_reward - comfort_penalty - soc_penalty
        return max(0.1, min(1.0, total_reward))
    
    def _calculate_current_efficiency(self):
        """현재 에너지 효율 계산 - 고정값 문제 해결됨"""
        if self.total_energy_consumed > 0.0001 and self.total_distance > 0.0001:
            efficiency = self.total_distance / self.total_energy_consumed
            # 학습 진행을 확인할 수 있도록 넓은 범위 허용
            return max(0.1, min(50.0, efficiency))
        # 초기 상태에서는 낮은 값 반환 (학습 진행 확인 가능)
        return 1.0
    
    def get_episode_metrics(self):
        """에피소드 종료 시 성능 지표 계산"""
        if len(self.episode_data['speeds']) == 0:
            return {
                'energy_efficiency': self._calculate_current_efficiency(),
                'soc_decrease_rate': ((0.8 - self.current_soc) / 0.8) * 100,
                'speed_tracking_rate': max(0, 100 - abs(self.current_speed - self.target_speed) * 5),
                'target_speed_proximity': abs(self.current_speed - self.target_speed),
                'comfort_score': 7.0,
                'safety_violations': 0
            }
        
        # 메트릭 계산
        energy_efficiency = self._calculate_current_efficiency()
        
        initial_soc = 0.8
        soc_decrease_rate = ((initial_soc - self.current_soc) / initial_soc) * 100
        
        speeds = np.array(self.episode_data['speeds'])
        in_range = np.abs(speeds - self.target_speed) <= 5
        speed_tracking_rate = np.mean(in_range) * 100
        
        target_speed_proximity = np.mean(np.abs(speeds - self.target_speed))
        
        accelerations = np.array(self.episode_data['accelerations'])
        comfort_score = max(0, 10 - np.std(accelerations) * 5)
        
        safety_violations = np.sum(np.abs(accelerations) > 2.5)
        
        return {
            'energy_efficiency': energy_efficiency,
            'soc_decrease_rate': soc_decrease_rate,
            'speed_tracking_rate': speed_tracking_rate,
            'target_speed_proximity': target_speed_proximity,
            'comfort_score': comfort_score,
            'safety_violations': safety_violations
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
    
        # 에피소드 카운터 증가
        self.episode_count += 1
    
        # 차량 상태 초기화
        self.current_data_idx = 0
        self.current_speed = 60.0
        self.current_soc = 0.8
        self.step_count = 0
        self.total_distance = 0.0
        self.total_energy_consumed = 0.0
    
        # 에피소드 데이터 초기화
        self.episode_data = {
            'speeds': [],
            'accelerations': [],
            'energy_consumption': [],
            'soc_changes': [],
            'rewards': [],
            'rush_periods': [],
            'traffic_conditions': [],
            'distances': [],
            'elevation_changes': []
        }

        info = {
            'initial_soc': self.current_soc,
            'initial_speed': self.current_speed,
            'route_distance': 7.816
        }
    
        # 랜덤 데이터 샘플 선택
        if hasattr(self, 'train_data') and len(self.train_data) > 0:
            self.current_data_idx = np.random.randint(0, len(self.train_data))
            initial_row = self.train_data.iloc[self.current_data_idx]
        
            if isinstance(initial_row['state_vector_rush'], list):
                self.state = np.array(initial_row['state_vector_rush'], dtype=np.float32)
            else:
                self.state = np.random.uniform(-1, 1, 28).astype(np.float32)
        
            if len(self.state) >= 28:
                self.state[25] = self.current_soc
                self.state[26] = self.current_speed / 120.0
        else:
            self.state = np.random.uniform(-1, 1, 28).astype(np.float32)
    
        return self.state.copy(), info

# 크루즈 모드 기준선
class CruiseControlBaseline:
    """크루즈 모드 기준선 - 일정 속도 유지"""
    
    def __init__(self, target_speed=60.0):
        self.target_speed = target_speed
        self.kp = 0.8
        self.ki = 0.1
        self.kd = 0.05
        self.integral_error = 0
        self.previous_error = 0
    
    def predict(self, observation, deterministic=True):
        """PID 제어로 목표 속도 유지"""
        current_speed_normalized = observation[26] if len(observation) > 26 else 0.5
        current_speed = current_speed_normalized * 120.0
        
        error = self.target_speed - current_speed
        
        self.integral_error += error
        derivative_error = error - self.previous_error
        
        acceleration = (self.kp * error + 
                       self.ki * self.integral_error + 
                       self.kd * derivative_error)
        
        acceleration = np.clip(acceleration, -3.0, 3.0)
        self.previous_error = error
        
        return np.array([acceleration]), None


def evaluate_cruise_baseline(env, num_episodes=50):
    """크루즈 모드 기준선 평가 - SageMaker 설정"""
    cruise_controller = CruiseControlBaseline(target_speed=60.0)
    
    episode_metrics = []
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = cruise_controller.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        metrics = env.get_episode_metrics()
        episode_metrics.append(metrics)
        episode_rewards.append(episode_reward)
    
    # 평균 성능 계산
    avg_efficiency = np.mean([m['energy_efficiency'] for m in episode_metrics])
    avg_speed_tracking = np.mean([m['speed_tracking_rate'] for m in episode_metrics])
    avg_reward = np.mean(episode_rewards)
    
    return {
        'energy_efficiency': {'mean': avg_efficiency, 'values': [m['energy_efficiency'] for m in episode_metrics]},
        'speed_tracking_rate': {'mean': avg_speed_tracking, 'values': [m['speed_tracking_rate'] for m in episode_metrics]},
        'episode_reward': {'mean': avg_reward, 'values': episode_rewards}
    }, episode_metrics


def install_transfer_learning_dependencies():
    """전이학습 라이브러리 설치"""
    try:
        import subprocess
        import sys
        
        logger.info("🔧 전이학습 라이브러리 설치 중...")
        
        # huggingface_sb3 설치
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "huggingface-sb3", "--quiet"
        ])
        
        logger.info(" huggingface_sb3 설치 완료")
        return True
        
    except Exception as e:
        logger.warning(f" 전이학습 라이브러리 설치 실패: {e}")
        return False


def train_sac_model(model_name, is_transfer_learning=False, total_timesteps=100000, data_dir="./data", save_dir="./models"):
    """SAC 모델 훈련 - SageMaker 최적화"""
    
    logger.info(f" SAC 모델 훈련 시작: {model_name}")
    logger.info(f"전이학습: {is_transfer_learning}")
    logger.info(f" 총 스텝: {total_timesteps}")
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # 환경 생성
    env = EVEnergyEnvironmentPreprocessed(data_dir=data_dir)
    
    # SageMaker 최적화된 SAC 설정
    sac_config = {
        'learning_rate': 3e-4,
        'buffer_size': 100000,
        'batch_size': 256,  # 실험 설계서 권장값
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'verbose': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 모델 생성
    if is_transfer_learning:
        logger.info(" 전이학습 모델 로드 시도...")
        
        # 전이학습 라이브러리 설치
        if install_transfer_learning_dependencies():
            try:
                from huggingface_sb3 import load_from_hub
                
                # 사전학습 모델 후보들 (실험 설계서 기준)
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
                        
                        # 기존 모델 로드
                        temp_model = SAC.load(checkpoint)
                        
                        # 새 환경에 맞게 모델 재생성 (네트워크 구조 복사)
                        model = SAC(
                            policy=temp_model.policy_class,
                            env=env,
                            **sac_config
                        )
                        
                        # 가능한 파라미터 복사 (실험 설계서 도메인 적응 전략)
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
        logger.info("🆕 순수 학습 모델 생성")
        model = SAC('MlpPolicy', env, **sac_config)
    
    # 콜백 설정 (SageMaker 최적화)
    eval_callback = PerformanceMetricsCallback(
        eval_env=env,
        eval_freq=2000,  # SageMaker용 빈도 조정
        verbose=1
    )
    
    # 훈련 시작
    logger.info(f" 학습 시작 - 목표: {total_timesteps} 스텝")
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=100,
            progress_bar=True
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        logger.info(f" 학습 완료! 소요시간: {training_time:.1f}초")
        
    except KeyboardInterrupt:
        logger.info("사용자 중단")
    except Exception as e:
        logger.error(f" 학습 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    
    # 모델 저장
    model_path = f"{save_dir}/{model_name}.zip"
    model.save(model_path)
    logger.info(f" 모델 저장: {model_path}")
    
    # 성능 평가 (실험 설계서 기준)
    logger.info("최종 성능 평가 중...")
    eval_results, eval_episodes = evaluate_policy(
        model, env, n_eval_episodes=50, return_episode_rewards=True  # 50회로 증가
    )
    
    # 메트릭 수집
    final_metrics = []
    for _ in range(50):  # 50회로 증가
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        final_metrics.append(env.get_episode_metrics())
    
    # 결과 정리
    results = {
        'model_name': model_name,
        'is_transfer_learning': is_transfer_learning,
        'training_time': training_time,
        'total_timesteps': total_timesteps,
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
        'learning_history': eval_callback.metrics_history
    }
    
    # 결과 저장
    results_path = f"./results/{model_name}_results.json"
    with open(results_path, 'w') as f:
        # JSON 직렬화를 위해 numpy 배열을 리스트로 변환
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        json_results[key][k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, np.ndarray):
                                json_results[key][k][kk] = vv.tolist()
                            else:
                                json_results[key][k][kk] = vv
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    logger.info(f"결과 저장: {results_path}")
    
    return model, results


def compare_models_and_baseline():
    """모델 성능 비교 및 분석 - SageMaker 설정"""
    
    logger.info("=" * 60)
    logger.info("SAC 전기차 에너지 효율 최적화 실험 시작 (SageMaker 최적화)")
    logger.info("=" * 60)
    
    # 데이터 디렉토리 확인
    data_dir = "./data"
    if not os.path.exists(data_dir):
        logger.error(f" 데이터 디렉토리 없음: {data_dir}")
        logger.info(" 다음 중 하나를 생성하세요:")
        logger.info("  - ./data/rush_separated_train_corrected_*.csv")
        logger.info("  - ./data/rush_separated_test_corrected_*.csv")
        logger.info("  - ./data/rush_normalization_corrected_*.json")
        return
    
    # 환경 생성 (기준선 평가용)
    env = EVEnergyEnvironmentPreprocessed(data_dir=data_dir)
    
    # 1. 크루즈 모드 기준선 평가
    logger.info("1단계: 크루즈 모드 기준선 평가")
    cruise_results, cruise_episodes = evaluate_cruise_baseline(env, num_episodes=50)
    
    logger.info("크루즈 모드 결과:")
    logger.info(f"  에너지 효율: {cruise_results['energy_efficiency']['mean']:.3f} km/kWh")
    logger.info(f"  속도 추종률: {cruise_results['speed_tracking_rate']['mean']:.1f}%")
    logger.info(f"  평균 보상: {cruise_results['episode_reward']['mean']:.3f}")
    
    # 2. SAC 순수 학습 (실험 설계서 기준)
    logger.info("\n 2단계: SAC 순수 학습")
    sac_scratch_model, sac_scratch_results = train_sac_model(
        model_name="sac_from_scratch",
        is_transfer_learning=False,
        total_timesteps=100000,  # 순수 학습용 스텝 수
        data_dir=data_dir
    )
    
    if sac_scratch_results:
        logger.info("SAC 순수 학습 결과:")
        logger.info(f"  에너지 효율: {sac_scratch_results['metrics']['energy_efficiency']['mean']:.3f} km/kWh")
        logger.info(f"  속도 추종률: {sac_scratch_results['metrics']['speed_tracking_rate']['mean']:.1f}%")
        logger.info(f"  평균 보상: {sac_scratch_results['eval_mean_reward']:.3f}")
        logger.info(f"  학습 시간: {sac_scratch_results['training_time']:.1f}초")
    
    # 3. SAC 전이 학습 (실험 설계서 기준)
    logger.info("\n 3단계: SAC 전이 학습")
    sac_transfer_model, sac_transfer_results = train_sac_model(
        model_name="sac_with_transfer",
        is_transfer_learning=True,
        total_timesteps=50000,  # 전이 학습용 스텝 수 (50% 단축)
        data_dir=data_dir
    )
    
    if sac_transfer_results:
        logger.info("SAC 전이 학습 결과:")
        logger.info(f"  에너지 효율: {sac_transfer_results['metrics']['energy_efficiency']['mean']:.3f} km/kWh")
        logger.info(f"  속도 추종률: {sac_transfer_results['metrics']['speed_tracking_rate']['mean']:.1f}%")
        logger.info(f"  평균 보상: {sac_transfer_results['eval_mean_reward']:.3f}")
        logger.info(f"  학습 시간: {sac_transfer_results['training_time']:.1f}초")
    
    # 4. 결과 비교 및 분석
    logger.info("\n4단계: 결과 비교 및 분석")
    
    comparison_results = {
        'cruise_mode': cruise_results,
        'sac_scratch': sac_scratch_results,
        'sac_transfer': sac_transfer_results
    }
    
    # 실험 설계서 기준 비교 표 생성
    print("\n" + "="*80)
    print("성능 비교 요약 (실험 설계서 기준)")
    print("="*80)
    print(f"{'모델':<20} {'에너지효율':<12} {'속도추종률':<12} {'평균보상':<10} {'학습시간':<10}")
    print("-"*80)
    
    # 크루즈 모드
    print(f"{'크루즈 모드':<20} {cruise_results['energy_efficiency']['mean']:<12.3f} "
          f"{cruise_results['speed_tracking_rate']['mean']:<12.1f} "
          f"{cruise_results['episode_reward']['mean']:<10.3f} {'N/A':<10}")
    
    # SAC 순수 학습
    if sac_scratch_results:
        print(f"{'SAC 순수학습':<20} {sac_scratch_results['metrics']['energy_efficiency']['mean']:<12.3f} "
              f"{sac_scratch_results['metrics']['speed_tracking_rate']['mean']:<12.1f} "
              f"{sac_scratch_results['eval_mean_reward']:<10.3f} {sac_scratch_results['training_time']:<10.1f}")
    
    # SAC 전이 학습
    if sac_transfer_results:
        print(f"{'SAC 전이학습':<20} {sac_transfer_results['metrics']['energy_efficiency']['mean']:<12.3f} "
              f"{sac_transfer_results['metrics']['speed_tracking_rate']['mean']:<12.1f} "
              f"{sac_transfer_results['eval_mean_reward']:<10.3f} {sac_transfer_results['training_time']:<10.1f}")
    
    print("="*80)
    
    # 실험 설계서 가설 검증
    if sac_scratch_results and sac_transfer_results:
        cruise_efficiency = cruise_results['energy_efficiency']['mean']
        scratch_efficiency = sac_scratch_results['metrics']['energy_efficiency']['mean']
        transfer_efficiency = sac_transfer_results['metrics']['energy_efficiency']['mean']
        
        scratch_improvement = ((scratch_efficiency - cruise_efficiency) / cruise_efficiency) * 100
        transfer_improvement = ((transfer_efficiency - cruise_efficiency) / cruise_efficiency) * 100
        
        # 수렴 시간 비교
        scratch_time = sac_scratch_results['training_time']
        transfer_time = sac_transfer_results['training_time']
        time_reduction = ((scratch_time - transfer_time) / scratch_time) * 100
        
        print(f"\n 실험 설계서 가설 검증:")
        print(f"  H1 - 전이학습 > 순수학습 > 크루즈: {' 성공' if transfer_efficiency > scratch_efficiency > cruise_efficiency else ' 실패'}")
        print(f"  H2 - 전이학습 수렴 시간 단축: {time_reduction:+.1f}% ({' 성공' if time_reduction > 0 else ' 실패'})")
        print(f"  H3 - 20% 이상 효율 개선:")
        print(f"    순수학습: {scratch_improvement:+.1f}% ({' 달성' if scratch_improvement >= 20 else ' 미달'})")
        print(f"    전이학습: {transfer_improvement:+.1f}% ({' 달성' if transfer_improvement >= 20 else ' 미달'})")
        
        if transfer_efficiency > scratch_efficiency:
            transfer_advantage = ((transfer_efficiency - scratch_efficiency) / scratch_efficiency) * 100
            print(f"  전이학습 우위: {transfer_advantage:+.1f}%")
    
    # 최종 결과 저장
    final_comparison = {
        'experiment_date': datetime.now().isoformat(),
        'data_directory': data_dir,
        'sagemaker_optimized': True,
        'hypothesis_verification': {
            'H1_efficiency_ranking': transfer_efficiency > scratch_efficiency > cruise_efficiency if sac_scratch_results and sac_transfer_results else False,
            'H2_convergence_speedup': time_reduction > 0 if sac_scratch_results and sac_transfer_results else False,
            'H3_20percent_improvement': {
                'scratch_achieved': scratch_improvement >= 20 if sac_scratch_results else False,
                'transfer_achieved': transfer_improvement >= 20 if sac_transfer_results else False
            }
        },
        'results': comparison_results
    }
    
    with open('./results/final_comparison_sagemaker.json', 'w') as f:
        # JSON 직렬화 처리
        def serialize_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_for_json(item) for item in obj]
            else:
                return obj
        
        json.dump(serialize_for_json(final_comparison), f, indent=2)
    
    logger.info(" 최종 비교 결과 저장: ./results/final_comparison_sagemaker.json")
    
    return comparison_results


def main():
    """메인 실행 함수 - SageMaker 최적화"""
    parser = argparse.ArgumentParser(description='SAC 전기차 에너지 효율 최적화 - SageMaker 실험')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='데이터 디렉토리 경로')
    parser.add_argument('--timesteps_scratch', type=int, default=100000,
                       help='순수학습 스텝 수 (실험 설계서 기준)')
    parser.add_argument('--timesteps_transfer', type=int, default=50000,
                       help='전이학습 스텝 수 (실험 설계서 기준)')
    parser.add_argument('--mode', type=str, choices=['compare', 'train_scratch', 'train_transfer'], 
                       default='compare', help='실행 모드')
    parser.add_argument('--aws_instance', type=str, default='ml.m5.xlarge',
                       help='AWS SageMaker 인스턴스 타입')
    
    args = parser.parse_args()
    
    # SageMaker 환경 정보 로깅
    logger.info(" SageMaker 최적화 SAC 실험 시작")
    logger.info(f"인스턴스 타입: {args.aws_instance}")
    logger.info(f"데이터 디렉토리: {args.data_dir}")
    logger.info(f"실행 모드: {args.mode}")
    
    if args.mode == 'compare':
        # 전체 비교 실험 실행 (실험 설계서 전체)
        results = compare_models_and_baseline()
        
    elif args.mode == 'train_scratch':
        # 순수 학습만 실행
        model, results = train_sac_model(
            model_name="sac_from_scratch",
            is_transfer_learning=False,
            total_timesteps=args.timesteps_scratch,
            data_dir=args.data_dir
        )
        
    elif args.mode == 'train_transfer':
        # 전이 학습만 실행
        model, results = train_sac_model(
            model_name="sac_with_transfer", 
            is_transfer_learning=True,
            total_timesteps=args.timesteps_transfer,
            data_dir=args.data_dir
        )
    
    logger.info("SageMaker 실험 완료")


if __name__ == "__main__":
    main()
