================================================================================
OnStreamLLM v0.1 - GitHub Guide
Developer: Cho Miles (@chomiles) | https://github.com/chomiles
================================================================================


################################################################################
# 한국어 (Korean)
################################################################################

■ 이 저장소에 포함된 것
  - 소스 코드, 빌드 스크립트, 설정 예시
  - 모델·가상환경·배포본은 포함되지 않습니다.

■ Release(배포판)에서 받는 것
  - OnStreamLLM.exe 포터블 패키지 (zip)
  - Python 설치 없이 실행 가능
  - 모델·런타임 라이브러리는 앱에서 다운로드

--------------------------------------------------------------------------------
[일반 사용자] 배포판 사용 방법
--------------------------------------------------------------------------------

1. GitHub Releases에서 OnStreamLLM-v0.1.0.zip 을 다운로드합니다.
2. 원하는 폴더에 압축을 풉니다.
3. OnStreamLLM.exe 를 실행합니다.
4. 모델 탭에서 STT·번역 모델을 다운로드합니다.
5. 필요한 런타임 라이브러리(Torch, SenseVoice, llama 등)를 설치합니다.
6. 엔진 시작 후 마이크/스피커/화면 감지를 사용합니다.

OBS 브라우저 소스:
  - 설정 → 서버 / 원격 탭에서 주소 복사
  - 기본: http://127.0.0.1:17865/overlay
  - 같은 PC에서만 접근 가능

--------------------------------------------------------------------------------
[개발자] 소스에서 빌드·실행
--------------------------------------------------------------------------------

요구 사항:
  - Windows 10/11 64bit
  - 인터넷 연결 (최초 setup 시)
  - NVIDIA GPU 권장 (CUDA 가속)

개발 환경 설치:
  powershell -ExecutionPolicy Bypass -File setup.ps1

실행:
  run.bat
  또는: .\.venv\Scripts\python.exe -m live_translate.main

배포판 빌드:
  powershell -ExecutionPolicy Bypass -File build_portable.ps1
  (기본 프로필: download-runtime)

결과물:
  dist\OnStreamLLM\

Release zip 생성 예:
  Compress-Archive -Path dist\OnStreamLLM\* -DestinationPath release\OnStreamLLM-v0.1.0.zip

--------------------------------------------------------------------------------
주의사항
--------------------------------------------------------------------------------

• 모델 용량: STT·LLM 모델은 수 GB입니다. 앱에서 선택 다운로드하세요.
• 백신 오진: 개인 개발 앱으로 일부 백신이 오진할 수 있습니다. 공식 Release 확인 후 예외 등록하세요.
• 원격 연산 보안:
  - 호스트는 모든 인터페이스(0.0.0.0)에서 수신합니다.
  - 패스워드(12자 이상)와 클라이언트 IP 화이트리스트로 접근을 제한하세요.
  - LAN/VPN 외부 공개·포트포워딩은 권장하지 않습니다. TLS 미적용.
• OBS 오버레이·설정 API는 로컬 PC에서만 접근됩니다.
• 설정 파일: Config\settings.json (패스워드는 Windows 계정에 묶어 암호화)
• 로그: logs\live_translate.log

--------------------------------------------------------------------------------
GitHub 업로드 대상 (이 폴더 Github\ 내용)
--------------------------------------------------------------------------------

포함: src, tests, Config(예시만), pyproject.toml, setup.ps1, build_portable.ps1,
      live_translate.spec, run.bat, run.ps1, portable_main.py, icon.ico,
      info.txt, README_FIELD_TEST.md, .gitignore, github.txt

제외: models, .venv, dist, build, runtime, settings.json, runtime_libraries


################################################################################
# English
################################################################################

■ What this repository contains
  - Source code, build scripts, and sample configuration
  - Models, virtual environments, and release binaries are NOT included

■ What you get from Releases
  - Portable OnStreamLLM.exe package (zip)
  - No Python installation required
  - Models and runtime libraries are downloaded inside the app

--------------------------------------------------------------------------------
[End users] Using the release build
--------------------------------------------------------------------------------

1. Download OnStreamLLM-v0.1.0.zip from GitHub Releases.
2. Extract it to any folder.
3. Run OnStreamLLM.exe.
4. Download STT and translation models from the Models tab.
5. Install required runtime libraries (Torch, SenseVoice, llama, etc.).
6. Start the engine, then enable input/output/screen detection.

OBS browser source:
  - Copy the URL from Settings → Server / Remote
  - Default: http://127.0.0.1:17865/overlay
  - Local access only

--------------------------------------------------------------------------------
[Developers] Build and run from source
--------------------------------------------------------------------------------

Requirements:
  - Windows 10/11 64-bit
  - Internet connection (for initial setup)
  - NVIDIA GPU recommended (CUDA acceleration)

Setup:
  powershell -ExecutionPolicy Bypass -File setup.ps1

Run:
  run.bat
  or: .\.venv\Scripts\python.exe -m live_translate.main

Build release:
  powershell -ExecutionPolicy Bypass -File build_portable.ps1
  (default profile: download-runtime)

Output:
  dist\OnStreamLLM\

Example release zip:
  Compress-Archive -Path dist\OnStreamLLM\* -DestinationPath release\OnStreamLLM-v0.1.0.zip

--------------------------------------------------------------------------------
Important notes
--------------------------------------------------------------------------------

• Model size: STT/LLM models are several GB. Download only what you need in the app.
• Antivirus false positives: Some scanners may flag the app. Verify the official GitHub Release before adding an exception.
• Remote compute security:
  - The host listens on all interfaces (0.0.0.0).
  - Restrict access with a password (12+ chars) and client IP whitelist.
  - Do not expose via port forwarding. Traffic is not TLS-encrypted.
• OBS overlay and settings API are local-only.
• Settings file: Config\settings.json (password encrypted per Windows user)
• Logs: logs\live_translate.log

--------------------------------------------------------------------------------
GitHub upload contents (this Github\ folder)
--------------------------------------------------------------------------------

Include: src, tests, Config (examples only), pyproject.toml, setup.ps1,
         build_portable.ps1, live_translate.spec, run.bat, run.ps1,
         portable_main.py, icon.ico, info.txt, README_FIELD_TEST.md,
         .gitignore, github.txt

Exclude: models, .venv, dist, build, runtime, settings.json, runtime_libraries


################################################################################
# 日本語 (Japanese)
################################################################################

■ このリポジトリに含まれるもの
  - ソースコード、ビルドスクリプト、設定サンプル
  - モデル・仮想環境・配布バイナリは含まれません

■ Releases で入手するもの
  - ポータブル OnStreamLLM.exe パッケージ (zip)
  - Python のインストール不要
  - モデルとランタイムライブラリはアプリ内でダウンロード

--------------------------------------------------------------------------------
[一般ユーザー] 配布版の使い方
--------------------------------------------------------------------------------

1. GitHub Releases から OnStreamLLM-v0.1.0.zip をダウンロードします。
2. 任意のフォルダに展開します。
3. OnStreamLLM.exe を実行します。
4. モデルタブで STT・翻訳モデルをダウンロードします。
5. 必要なランタイムライブラリ (Torch, SenseVoice, llama 等) をインストールします。
6. エンジン開始後、入力/出力/画面検出を使用します。

OBS ブラウザソース:
  - 設定 → サーバー / リモート タブで URL をコピー
  - 既定: http://127.0.0.1:17865/overlay
  - 同一 PC からのみアクセス可能

--------------------------------------------------------------------------------
[開発者] ソースからのビルド・実行
--------------------------------------------------------------------------------

要件:
  - Windows 10/11 64bit
  - インターネット接続 (初回 setup 時)
  - NVIDIA GPU 推奨 (CUDA 加速)

開発環境セットアップ:
  powershell -ExecutionPolicy Bypass -File setup.ps1

実行:
  run.bat
  または: .\.venv\Scripts\python.exe -m live_translate.main

配布版ビルド:
  powershell -ExecutionPolicy Bypass -File build_portable.ps1
  (既定プロファイル: download-runtime)

出力:
  dist\OnStreamLLM\

Release zip 作成例:
  Compress-Archive -Path dist\OnStreamLLM\* -DestinationPath release\OnStreamLLM-v0.1.0.zip

--------------------------------------------------------------------------------
注意事項
--------------------------------------------------------------------------------

• モデル容量: STT・LLM モデルは数 GB です。アプリ内で必要なものだけダウンロードしてください。
• ウイルス対策の誤検知: 個人開発アプリのため誤検知される場合があります。公式 Release を確認してから例外登録してください。
• リモート演算のセキュリティ:
  - ホストはすべてのインターフェース (0.0.0.0) で受信します。
  - パスワード (12文字以上) とクライアント IP ホワイトリストでアクセスを制限してください。
  - LAN/VPN 外への公開・ポートフォワーディングは非推奨。TLS 未対応。
• OBS オーバーレイと設定 API はローカル PC のみアクセス可能です。
• 設定ファイル: Config\settings.json (パスワードは Windows ユーザーに紐づけて暗号化)
• ログ: logs\live_translate.log

--------------------------------------------------------------------------------
GitHub アップロード対象 (この Github\ フォルダの内容)
--------------------------------------------------------------------------------

含める: src, tests, Config (サンプルのみ), pyproject.toml, setup.ps1,
        build_portable.ps1, live_translate.spec, run.bat, run.ps1,
        portable_main.py, icon.ico, info.txt, README_FIELD_TEST.md,
        .gitignore, github.txt

含めない: models, .venv, dist, build, runtime, settings.json, runtime_libraries
