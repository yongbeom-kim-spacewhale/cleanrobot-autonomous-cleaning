# rokey_cobot3

디지털 트윈 기반 로봇 자동화 시뮬레이션 시스템 프로젝트 저장소.

## 목표

- Isaac Sim 기반 다중 로봇 작업 환경 구축
- ROS 2 기반 로봇 제어 및 시스템 통합
- AI 비전 인식 결과를 로봇 동작에 반영
- 제3자가 `git clone` 후 README 절차로 실행 가능한 결과물 제작

## 저장소 구조

```text
cobot3_ws/
├── src/                  # ROS 2 패키지
├── isaacpjt/
│   └── basic/            # Isaac Sim standalone 스크립트
├── docs/
│   └── issues/           # 문제·오류·실패 보고서
├── evidence/
│   └── videos/           # 재현·시연 영상 안내
├── build/                # colcon 생성물, Git 제외
├── install/              # colcon 생성물, Git 제외
└── log/                  # colcon 생성물, Git 제외
```

## 기본 작업 흐름

```bash
git pull
# 파일 수정
git add <파일>
git commit -m "변경 내용"
git push
```

## 문제 보고 규칙

문제·오류·실패 발생 시 화면 녹화로 재현 과정과 현상을 남긴다.
영상 없는 문제 보고는 미완료로 취급한다.

보고서에 다음 항목을 포함한다.

- 날짜
- 담당자
- 환경
- 재현 절차
- 기대 결과
- 실제 결과
- 관련 로그
- 영상 파일 경로
- 해결 상태

`docs/issues/ISSUE_TEMPLATE.md`를 복사해 사용한다.

## 주의

- `build/`, `install/`, `log/`는 커밋하지 않는다.
- 비밀번호, PAT, API 키, 개인 `.env` 파일을 커밋하지 않는다.
- 대용량 영상은 GitHub 용량 정책 확인 후 Git LFS 또는 별도 공유 저장소를 사용한다.
# rokey_cobot3
