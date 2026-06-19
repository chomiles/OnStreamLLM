OnStreamLLM User Guide / 사용 설명서 / 使い方ガイド
=====================================================

한국어
------

1. 처음 실행
- 포터블 배포판은 `OnStreamLLM.exe`를 실행합니다.
- 소스 실행은 `run.bat`를 실행합니다. 처음 실행 시 전용 Python 환경이 자동 준비됩니다.
- 모델과 필수 라이브러리는 배포본에 포함하지 않고, 앱 안의 모델 관리 탭에서 내려받습니다.

2. 필수 라이브러리 설치
- `모델 관리` 탭 맨 위의 `필수 라이브러리 다운로드` 영역에서 설치합니다.
- 필수 설치 순서: Torch -> Qwen ASR -> llama -> SenseVoice.
- 필수 설치가 모두 끝나면 앱을 한 번 다시 실행해야 가장 안정적으로 시작됩니다.
- PaddleOCR은 필수가 아니며, 화면 번역을 사용할 때 별도로 설치합니다.
- 화면 번역을 처음 켜면 PaddleOCR 설치 여부를 묻습니다.
- SenseVoice는 GPU 충돌 방지를 위해 CPU 전용 라이브러리로 설치됩니다.

3. 권장 모델과 CPU/GPU 분담
- 게임이나 방송과 같이 실행할 때는 `프리셋 1 - 게임 권장: SenseVoice CPU / Hy-MT2 GPU`를 권장합니다.
- 이 조합은 STT를 CPU의 일부 코어에 맡기고, 번역 LLM은 GPU에 맡겨 게임 부하와 겹치는 정도를 줄입니다.
- `STT 연산 장치`는 CPU, `LLM 연산 장치`는 NVIDIA GPU로 설정합니다.
- `STT CPU 스레드`는 2개부터 시작하고, 여유가 있으면 3~4개까지 올립니다.
- SenseVoice 2024는 CPU 전용으로 사용합니다.
- Hy-MT2는 GPU 사용을 권장합니다.
- Qwen 4B는 무거우며 12GB 이상 VRAM에서, 게임을 실행하지 않을 때 사용을 권장합니다.
- Qwen 8B는 매우 무거우며 16GB 이상 VRAM에서, 게임을 실행하지 않을 때 사용을 권장합니다.

4. 모델 다운로드
- `모델 관리` 탭에서 사용할 STT 모델과 번역 모델을 다운로드합니다.
- SenseVoice는 `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` 모델을 사용합니다.
- 다운로드 후 모델을 선택하고 `설정 저장`을 누릅니다.
- Qwen 계열 모델은 로딩 시간이 길 수 있습니다. 하단 상태줄에 `모델 로드 완료`가 표시될 때까지 기다려 주세요.

5. 주요 메뉴
- `번역기`: 엔진 시작/중지, 입력 오디오 감지, 출력 오디오 감지, 화면 번역, 한줄 번역기를 사용합니다.
- `채널별 번역/장치 설정`: 입력 오디오, 출력 오디오, 화면 번역의 원문 언어와 번역 언어를 지정합니다. 입력/출력 오디오는 이곳에서 장치도 선택합니다.
- `모델 관리`: 필수 라이브러리 설치, 모델 다운로드, 프리셋 선택, CPU/GPU 연산 장치와 CPU 스레드를 설정합니다.
- `설정`: 자막 스타일, 팝업, 서버/원격 연결, OBS 오버레이 등을 설정합니다.
- `정보`: 앱 정보와 안내를 확인합니다.

6. 번역 시작
- `모델 관리`에서 모델과 연산 장치를 선택한 뒤 `설정 저장`을 누릅니다.
- `번역기` 탭에서 `엔진 시작`을 누릅니다.
- 모델 로드가 끝나면 필요한 감지 토글을 켭니다.
- 사용을 마치면 `엔진 중지`를 눌러 감지를 멈추고 모델 메모리 정리를 시도합니다.

7. 화면 번역
- 화면 번역을 켜기 전에 `화면 영역 지정`으로 자막이나 채팅 영역을 선택합니다.
- Korean, Japanese처럼 OCR 모델이 바뀌는 언어로 전환하면 엔진 재준비 시간이 필요합니다.
- 언어 변경이나 화면 번역 끄기 직후에는 전환 대기 시간이 표시될 수 있습니다. 카운트가 끝난 뒤 다시 켜면 안정적입니다.
- 줄 바꿈이 있는 화면 텍스트는 가능한 한 줄 바꿈을 유지해서 번역합니다.

8. 한줄 번역기
- `한줄 번역기 사용`을 켜면 입력 칸이 나타납니다.
- 입력 후 약 1.5초 동안 멈추면 자동 번역됩니다.
- 복사 버튼은 번역된 문장만 한 줄로 복사합니다.
- `교차검증`을 켜면 번역 결과를 다시 시작 언어로 번역해 확인할 수 있습니다.

9. 문제 해결
- GPU VRAM이 부족하면 Qwen 프리셋 대신 프리셋 1을 사용합니다.
- 화면 번역 오류가 반복되면 화면 영역을 다시 지정하고, 언어 변경 후 10초 정도 기다렸다가 켭니다.
- 오류 확인은 `logs/live_translate.log` 파일을 확인하거나 공유해 주세요.


English
-------

1. First launch
- For the portable build, run `OnStreamLLM.exe`.
- For source execution, run `run.bat`. A private Python environment is prepared on first launch.
- Models and required libraries are downloaded from inside the app, not bundled into the release.

2. Required libraries
- Open the `Model Management` tab and use the `Required Library Download` area at the top.
- Required install order: Torch -> Qwen ASR -> llama -> SenseVoice.
- After required libraries are installed, restart the app once for the most stable startup.
- PaddleOCR is optional and is installed separately when screen translation is used.
- When you enable screen translation for the first time, the app asks whether to install PaddleOCR.
- SenseVoice is installed as CPU-only to avoid GPU runtime conflicts.

3. Recommended CPU/GPU split
- For gaming or streaming, use `Preset 1 - Gaming: SenseVoice CPU / Hy-MT2 GPU`.
- This keeps STT on a small number of CPU threads while the translation LLM runs on the GPU.
- Set `STT device` to CPU and `LLM device` to your NVIDIA GPU.
- Start with 2 STT CPU threads. If the PC has headroom, use 3 or 4.
- SenseVoice 2024 should be used as CPU-only.
- Hy-MT2 is recommended on GPU.
- Qwen 4B is heavy and recommended for 12GB+ VRAM when not gaming.
- Qwen 8B is very heavy and recommended for 16GB+ VRAM when not gaming.

4. Model download
- Download the STT and translation models from `Model Management`.
- SenseVoice uses `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`.
- Select the downloaded models and press `Save Settings`.
- Qwen models can take time to load. Wait until the status bar says model loading is complete.

5. Main menus
- `Translator`: Start/stop the engine, enable input audio detection, output audio detection, screen translation, and the one-line translator.
- `Per-Channel Translation / Device Settings`: Choose source/target languages for input audio, output audio, and screen translation. Input/output audio devices are selected here.
- `Model Management`: Install required libraries, download models, choose presets, choose CPU/GPU devices, and assign CPU threads.
- `Settings`: Configure caption style, popup, server/remote connection, and OBS overlay.
- `Info`: Shows app information and guidance.

6. Start translation
- Select models and compute devices, then press `Save Settings`.
- Go to `Translator` and press `Start Engine`.
- After the models are ready, enable the detection toggles you need.
- Press `Stop Engine` when done to stop detection and release model memory as much as possible.

7. Screen translation
- Before enabling screen translation, select the subtitle or chat region.
- Switching OCR languages such as Korean and Japanese requires the OCR engine to reload.
- After changing languages or disabling screen translation, wait for the transition countdown before enabling it again.
- Line breaks in detected screen text are preserved as much as possible during translation.

8. One-line translator
- Enable `One-Line Translator` to show the input box.
- Translation runs automatically after you stop typing for about 1.5 seconds.
- The copy button copies only the translated sentence.
- Cross-check translates the result back to the source language.

9. Troubleshooting
- If VRAM is low, use Preset 1 instead of Qwen presets.
- If screen translation errors repeat, select the screen region again and wait about 10 seconds after changing languages.
- Check or share `logs/live_translate.log` for detailed errors.


日本語
------

1. 初回起動
- ポータブル版は `OnStreamLLM.exe` を実行します。
- ソースから実行する場合は `run.bat` を実行します。初回起動時に専用 Python 環境が準備されます。
- モデルと必須ライブラリは配布物に含めず、アプリ内のモデル管理タブからダウンロードします。

2. 必須ライブラリのインストール
- `モデル管理` タブ上部の `必須ライブラリダウンロード` からインストールします。
- 必須インストール順序: Torch -> Qwen ASR -> llama -> SenseVoice。
- 必須ライブラリのインストール後は、安定起動のためアプリを一度再起動してください。
- PaddleOCR は必須ではなく、画面翻訳を使う時に別途インストールします。
- 初めて画面翻訳をオンにすると、PaddleOCR をインストールするか確認されます。
- SenseVoice は GPU 衝突を避けるため CPU 専用ライブラリとしてインストールされます。

3. 推奨 CPU/GPU 分担
- ゲームや配信と同時に使う場合は `プリセット1 - ゲーム推奨: SenseVoice CPU / Hy-MT2 GPU` を推奨します。
- STT は少数の CPU スレッドに任せ、翻訳 LLM は GPU で処理します。
- `STT 演算デバイス` は CPU、`LLM 演算デバイス` は NVIDIA GPU に設定します。
- `STT CPU スレッド` は 2 から始め、余裕があれば 3~4 に増やします。
- SenseVoice 2024 は CPU 専用で使います。
- Hy-MT2 は GPU 使用を推奨します。
- Qwen 4B は重く、VRAM 12GB 以上でゲームを起動していない時の使用を推奨します。
- Qwen 8B は非常に重く、VRAM 16GB 以上でゲームを起動していない時の使用を推奨します。

4. モデルのダウンロード
- `モデル管理` タブで STT モデルと翻訳モデルをダウンロードします。
- SenseVoice は `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` を使用します。
- ダウンロード後、モデルを選択して `設定保存` を押します。
- Qwen 系モデルはロードに時間がかかることがあります。ステータスバーにロード完了が表示されるまで待ってください。

5. 主なメニュー
- `翻訳機`: エンジン開始/停止、入力音声検出、出力音声検出、画面翻訳、一行翻訳を使います。
- `チャンネル別翻訳/デバイス設定`: 入力音声、出力音声、画面翻訳の原文言語と翻訳先言語を設定します。入力/出力デバイスもここで選びます。
- `モデル管理`: 必須ライブラリ、モデル、プリセット、CPU/GPU、CPU スレッドを設定します。
- `設定`: 字幕スタイル、ポップアップ、サーバー/リモート接続、OBS オーバーレイを設定します。
- `情報`: アプリ情報と案内を確認します。

6. 翻訳開始
- モデルと演算デバイスを選び、`設定保存` を押します。
- `翻訳機` タブで `エンジン開始` を押します。
- モデル準備完了後、必要な検出トグルをオンにします。
- 終了時は `エンジン停止` を押して検出を止め、モデルメモリの解放を試みます。

7. 画面翻訳
- 画面翻訳をオンにする前に、字幕やチャットの範囲を選択します。
- Korean や Japanese のように OCR モデルが変わる言語へ切り替えると、再ロード時間が必要です。
- 言語変更や画面翻訳オフ直後は、カウントダウンが終わってから再度オンにしてください。
- 改行を含む画面テキストは、できるだけ改行を保ったまま翻訳します。

8. 一行翻訳
- `一行翻訳を使用` をオンにすると入力欄が表示されます。
- 入力後、約 1.5 秒止まると自動翻訳されます。
- コピーは翻訳された文だけをコピーします。
- `クロスチェック` をオンにすると、翻訳結果を開始言語へ戻して確認できます。

9. トラブルシューティング
- VRAM が足りない場合は Qwen プリセットではなくプリセット1を使います。
- 画面翻訳エラーが続く場合は、範囲を再選択し、言語変更後に約 10 秒待ってからオンにします。
- 詳細なエラーは `logs/live_translate.log` を確認または共有してください。
