# SageMaker 완전한 환경 설정 스크립트

import subprocess
import sys
import os

def install_all_dependencies():
    """SageMaker에서 필요한 모든 라이브러리 설치"""
    
    print("SageMaker 환경 설정 시작...")
    
    # 필수 라이브러리 목록
    libraries = [
        # 강화학습 핵심
        "stable-baselines3[extra]",
        "gymnasium",
        
        # 전이학습
        "huggingface-sb3",
        
        # 데이터 처리
        "pandas",
        "numpy", 
        "scipy",
        
        # 시각화
        "matplotlib",
        "seaborn",
        
        # AWS 관련
        "boto3",
        "sagemaker",
        
        # 기타 유틸
        "glob2"
    ]
    
    print(f"설치할 라이브러리: {len(libraries)}개")
    
    failed_installs = []
    
    for lib in libraries:
        try:
            print(f" {lib} 설치 중...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", lib, "--quiet"
            ])
            print(f"   {lib} 설치 완료")
            
        except Exception as e:
            print(f"   {lib} 설치 실패: {e}")
            failed_installs.append(lib)
    
    if failed_installs:
        print(f"\n 실패한 라이브러리: {failed_installs}")
        print(" 수동 설치 시도:")
        for lib in failed_installs:
            print(f"!pip install {lib}")
    else:
        print("\n 모든 라이브러리 설치 완료!")
    
    return len(failed_installs) == 0

def verify_imports():
    """필수 모듈 임포트 테스트"""
    
    print("\n🔍 모듈 임포트 검증 중...")
    
    import_tests = [
        ("stable_baselines3", "SAC"),
        ("gymnasium", "gym"),
        ("huggingface_sb3", "load_from_hub"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("matplotlib.pyplot", "plt"),
        ("seaborn", "sns"),
        ("scipy", "stats"),
        ("boto3", None),
        ("sagemaker", None)
    ]
    
    failed_imports = []
    
    for module, alias in import_tests:
        try:
            if alias:
                exec(f"import {module} as {alias}")
                print(f"   {module} as {alias}")
            else:
                exec(f"import {module}")
                print(f"   {module}")
                
        except ImportError as e:
            print(f"   {module} 임포트 실패: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n 임포트 실패: {failed_imports}")
        return False
    else:
        print("\n 모든 모듈 임포트 성공!")
        return True

def check_data_files():
    """데이터 파일 존재 확인"""
    
    print("\n📂 데이터 파일 확인 중...")
    
    required_files = [
        "./data/rush_separated_train_corrected_20250606_182210.csv",
        "./data/rush_separated_test_corrected_20250606_182210.csv", 
        "./data/rush_normalization_corrected_20250606_182210.json"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   {os.path.basename(file_path)}")
            
            # 파일 크기 확인
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"      크기: {size_mb:.2f} MB")
            
        else:
            print(f"   {os.path.basename(file_path)} (없음)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n 누락된 파일: {len(missing_files)}개")
        print(" 업로드 필요:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("\n 모든 데이터 파일 존재!")
        return True

def test_huggingface_connection():
    """허깅페이스 연결 및 모델 다운로드 테스트"""
    
    print("\n허깅페이스 모델 다운로드 테스트...")
    
    try:
        from huggingface_sb3 import load_from_hub
        from stable_baselines3 import SAC
        
        # 가장 작은 모델로 테스트
        print("  테스트 모델 다운로드 중...")
        checkpoint = load_from_hub(
            repo_id="sb3/sac-Pendulum-v1", 
            filename="sac-Pendulum-v1.zip"
        )
        
        # 모델 로드 테스트
        model = SAC.load(checkpoint)
        print("   허깅페이스 모델 다운로드 및 로드 성공")
        print(f"  테스트 모델 정보: {model.policy.__class__.__name__}")
        
        return True
        
    except Exception as e:
        print(f"   허깅페이스 연결 실패: {e}")
        print("   전이학습 없이 순수학습만 진행 가능")
        print("   S3 저장소 없이도 로컬 저장으로 완전한 실험 가능")
        return False

def check_s3_requirement():
    """S3 사용 필요성 안내"""
    
    print("\n저장 방식 안내:")
    print(" 로컬 저장: SageMaker 인스턴스 내부 (기본)")
    print("    - 장점: 추가 비용 없음, 설정 간단")
    print("    - 단점: 인스턴스 종료시 삭제됨")
    print("    - 권장: 실험 완료 후 결과 파일 다운로드")
    
    print("\n  S3 저장: AWS 클라우드 (선택사항)")  
    print("    - 장점: 영구 보관, 팀 공유 용이")
    print("    - 단점: 추가 비용 ($0.023/GB/월)")
    print("    - 권장: 장기 보관이나 팀 협업시")
    
    print("\n 현재 설정: S3 없이 로컬 저장만 사용")
    print("   실험 완료 후 중요 파일만 수동 다운로드하면 됩니다!")
    
    return True

def create_directory_structure():
    """필요한 디렉토리 구조 생성"""
    
    print("\n디렉토리 구조 생성 중...")
    
    directories = [
        "./data",
        "./models", 
        "./results",
        "./logs"
    ]
    
    for dir_path in directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"   {dir_path}")
        except Exception as e:
            print(f"   {dir_path} 생성 실패: {e}")

def generate_sagemaker_notebook():
    """SageMaker 실행용 노트북 코드 생성"""
    
    notebook_content = '''
# SageMaker SAC 전기차 에너지 효율 최적화 실험
# 실행 순서대로 셀을 실행하세요

# ============================================================================
# Cell 1: 환경 설정 (첫 번째 실행 필수)
# ============================================================================

# 환경 설정 스크립트 실행
exec(open('sagemaker_setup_complete.py').read())

# 완전한 환경 설정
success = install_all_dependencies()
if success:
    print(" 환경 설정 완료 - 다음 셀 실행 가능")
else:
    print(" 환경 설정 실패 - 수동 설치 필요")

# ============================================================================
# Cell 2: 모듈 검증 및 데이터 확인
# ============================================================================

# 모듈 임포트 검증
verify_imports()

# 데이터 파일 확인
check_data_files()

# 허깅페이스 연결 테스트
test_huggingface_connection()

# ============================================================================
# Cell 3: 훈련 코드 로드 및 실행
# ============================================================================

# 훈련 모듈 로드
exec(open('local_training_sagemaker.py').read())

print(" SAC 전기차 에너지 효율 최적화 실험 시작")

# 전체 실험 실행 (순수학습 100k + 전이학습 50k + 크루즈 모드)
results = compare_models_and_baseline()

print(" 훈련 완료 - 모델이 ./models/ 디렉토리에 저장됨")

# ============================================================================
# Cell 4: 결과 분석 및 테스트
# ============================================================================

# 테스트 모듈 로드
exec(open('test_sagemaker.py').read())

# 종합 성능 테스트 실행
tester = SageMakerExperimentTest(local_models_dir="./models")
final_results = tester.run_complete_sagemaker_test()

print(" 전체 실험 완료!")

# ============================================================================
# Cell 5: 결과 확인 및 다운로드
# ============================================================================

import os

# 생성된 결과 파일 목록
result_files = []
for root, dirs, files in os.walk('./'):
    for file in files:
        if any(keyword in file for keyword in ['results', 'models', 'report', '.png', '.json']):
            result_files.append(os.path.join(root, file))

print("생성된 결과 파일:")
for file in sorted(result_files):
    print(f"  - {file}")

print("\\n 중요 파일:")
print("  - ./models/sac_from_scratch.zip (순수학습 모델)")
print("  - ./models/sac_with_transfer.zip (전이학습 모델)")
print("  - ./results/final_comparison_sagemaker.json (최종 결과)")
print("  - ./sagemaker_results_*/sagemaker_experiment_report.md (보고서)")

# ============================================================================
# Cell 6: 실험 요약 및 가설 검증
# ============================================================================

if 'final_results' in locals() and final_results:
    print("실험 설계서 가설 검증 결과:")
    
    hypothesis = final_results.get('hypothesis_verification', {})
    
    print(f"  H1 (전이>순수>크루즈): {'' if hypothesis.get('H1_energy_efficiency_ranking') else ''}")
    print(f"  H2 (50% 스텝 단축): {'' if hypothesis.get('H2_training_efficiency') else ''}")  
    print(f"  H3 (20% 효율 개선): {'' if hypothesis.get('H3_20percent_improvement') else ''}")
    print(f"  H4 (비용 효율성): {'' if hypothesis.get('H4_cost_efficiency') else ''}")
    
    print("\\n실험 성공! 논문 작성에 필요한 모든 데이터 확보")
else:
    print(" 실험 결과를 찾을 수 없습니다. 이전 셀을 다시 실행하세요.")
'''

    with open("sagemaker_experiment_notebook.py", "w", encoding="utf-8") as f:
        f.write(notebook_content)
    
    print("SageMaker 실험 노트북 생성: sagemaker_experiment_notebook.py")

def main():
    """메인 설정 함수"""
    
    print("SageMaker SAC 전기차 실험 환경 설정")
    print("=" * 60)
    
    # 0. S3 사용 필요성 안내
    check_s3_requirement()
    
    # 1. 디렉토리 구조 생성
    create_directory_structure()
    
    # 2. 라이브러리 설치
    install_success = install_all_dependencies()
    
    # 3. 모듈 검증
    if install_success:
        import_success = verify_imports()
    else:
        import_success = False
    
    # 4. 데이터 파일 확인
    data_success = check_data_files()
    
    # 5. 허깅페이스 연결 테스트 (실패해도 실험 가능)
    if import_success:
        hf_success = test_huggingface_connection()
    else:
        hf_success = False
    
    # 6. 노트북 생성
    generate_sagemaker_notebook()
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("SageMaker 환경 설정 결과 (S3 없이 로컬 저장)")
    print("=" * 60)
    
    checks = [
        ("라이브러리 설치", install_success),
        ("모듈 임포트", import_success), 
        ("데이터 파일", data_success),
        ("허깅페이스 연결", hf_success, "선택사항")
    ]
    
    for item in checks:
        if len(item) == 3:  # 선택사항
            check_name, success, note = item
            status = "" if success else ""
            print(f"{status} {check_name} ({note})")
        else:
            check_name, success = item
            status = "" if success else ""
            print(f"{status} {check_name}")
    
    # 허깅페이스 연결은 필수가 아니므로 제외
    essential_success = all(success for check_name, success in checks[:3])
    
    if essential_success:
        print("\n핵심 설정 완료 S3 없이 실험 실행 가능")
        print("\n다음 단계:")
        print("1. 데이터 파일을 ./data/ 폴더에 업로드")
        print("2. sagemaker_experiment_notebook.py 코드를 노트북에서 순서대로 실행")
        print("3. 결과 분석 및 보고서 확인")
        print("4. 실험 완료 후 중요 파일 다운로드")
        
        if not hf_success:
            print("\n 허깅페이스 연결 실패시:")
            print("  - 전이학습 건너뛰고 순수학습만 실행")
            print("  - 또는 인터넷 연결 확인 후 재시도")
            
    else:
        print("\n 필수 설정 실패 - 수동 설정 필요")
        print("\n문제 해결:")
        if not install_success:
            print("  - 라이브러리 수동 설치: !pip install stable-baselines3[extra] huggingface-sb3")
        if not data_success:
            print("  - 데이터 파일 업로드: SageMaker 파일 브라우저 사용")

if __name__ == "__main__":
    main()