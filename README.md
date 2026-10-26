# AI-Based-Mood-Tracking-and-Journaling-App
An AI-Based Mood Tracking and Journaling App that helps users track their daily emotions, maintain journal entries, and understand their emotional patterns over time. Using AI-powered sentiment analysis, the application analyzes journal text, identifies moods, and provides insights into emotional well-being through visualizations and mood tracking.

## Application

The working application is in [`backend/`](./backend) and [`mobile/`](./mobile). The backend is a local-first FastAPI service with SQLite persistence, VADER sentiment analysis, optional transformer emotion classification, correction storage, CSV export, and weekly/monthly analytics.

### Run the backend on Windows

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run the API tests from the repository root with `python -m pytest backend\tests -q`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The API is intentionally local-first. Requests use the `X-User-ID` header (default `local-user`) to keep one device's records separate from another; this is not multi-user production authentication. Journal text is processed by the local backend and is never logged. Export and delete-all endpoints are available under `/api/export` and `/api/data`.

### One-command laptop startup

Because Flutter/Android tooling is not installed in the current environment, the repository also includes a browser desktop target using the same FastAPI backend and APIs. It is the fastest way to run the application on this Windows laptop today:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_app.ps1
```

The script checks Python, starts FastAPI on `127.0.0.1:8000`, waits for `/api/health`, and opens the application in the default browser. It defaults to VADER plus the safe emotion fallback. To enable the transformer if its dependencies are installed:

```powershell
$env:MOOD_ENABLE_EMOTION_MODEL = "true"
.\run_app.ps1
```

Use `.\stop_app.ps1` to stop the local service. Choose another port with `.\run_app.ps1 -Port 8010`. The desktop target supports entry creation, AI comparison, history search/filtering, editing, deletion, correction, insights, CSV download, and delete-all confirmation.

### Mobile client

From `mobile/`, run `flutter pub get`, `flutter analyze`, `flutter test`, and `flutter run`. Android emulators can usually reach a host backend at `http://10.0.2.2:8000`; physical devices need the host LAN address. Flutter, Dart, and the Android SDK were not installed in the current environment, so an APK and mobile test results cannot be claimed here.

The client now includes onboarding, dashboard mood capture, immediate AI comparison, searchable/filterable history, prediction correction, weekly/monthly graphical distributions, loading/error/empty states, CSV export/share, and delete-all controls. The API URL is configured at build/run time, with an Android-emulator default:

```powershell
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000
# physical Android phone:
flutter run --dart-define=API_BASE_URL=http://192.168.1.50:8000
# desktop/web:
flutter run --dart-define=API_BASE_URL=http://127.0.0.1:8000
```

The value is read by [`mobile/lib/api_client.dart`](./mobile/lib/api_client.dart); no production server address is embedded.

Transformer support is optional. The required backend dependencies provide VADER and the documented fallback. To enable the verified Hugging Face model in a suitable Python environment, install `transformers` and `torch` separately, then set `MOOD_ENABLE_EMOTION_MODEL=true`.

### Windows Flutter/Android installation

1. Install Git for Windows and the Flutter Windows SDK from the official Flutter installation guide.
2. Add `<flutter>\bin` to the user `PATH`, reopen PowerShell, and run `flutter doctor`.
3. Install Android Studio, select Android SDK, SDK Platform Tools, Android SDK Command-line Tools, and an Android SDK platform in SDK Manager.
4. Accept licenses with `flutter doctor --android-licenses`.
5. Create an emulator in Android Studio Device Manager, or enable Developer Options and USB debugging on a physical phone.
6. From `mobile/`, run `flutter pub get`, `flutter analyze`, `flutter test`, and `flutter build apk --debug`.

The debug APK, after a successful build, will be at `mobile/build/app/outputs/flutter-apk/app-debug.apk`. Install it with `adb install -r mobile\build\app\outputs\flutter-apk\app-debug.apk`, or copy it to a phone and open it. This repository currently has no generated `android/` directory because Flutter was unavailable when the project was created; run `flutter create .` from `mobile/` after installing Flutter to generate the standard Android runner files, then rerun the checks.

### Research and limitations

The original GoEmotions assets are preserved under [`dataset 1/`](./dataset%201/). They are Reddit-derived and are not representative of private journals. Existing preprocessing/evaluation artifacts are not silently presented as new application metrics. Personalization stores corrections per user and reports measured agreement only after enough examples; it does not train across users or diagnose mental-health conditions.

To reproduce metrics from the existing held-out predictions without downloading data:

```powershell
cd "dataset 1"
python evaluate_existing_predictions.py
```

This writes `preprocessed/reproducible_evaluation.json` with macro precision, macro recall, macro-F1, per-label scores, and a confusion matrix. The report uses `test_ekman.tsv` and `hf_model_test_predictions.tsv`, keeps the held-out split separate, and explicitly documents its primary-label reduction and Reddit-to-journal limitation.

### API surface

- `GET /api/health`, `GET /api/model/info`
- `GET /api/auth/session`
- `POST /api/analyze`
- `POST/GET /api/entries`, `GET/PUT/DELETE /api/entries/{id}`
- `POST /api/entries/{id}/correction` (supports `helpful: true` for agreement, or `helpful: false` with `corrected_emotion`)
- `GET /api/personalization/status` (checks user adaptation status, evidence counts, active state, and explanation)
- `POST /api/personalization/toggle` (enables or disables personalization for the user)
- `POST /api/personalization/reset` (permanently resets user adaptive priors and correction records)
- `GET /api/insights/weekly`, `GET /api/insights/monthly` (includes daily trendlines, tone alignment/divergence, and calibration metrics)
- `GET /api/export` (downloads comprehensive 14-column CSV with AI analysis, confidence, feedback, and formula injection sanitization)
- `DELETE /api/data` (scoped deletion of all entries and feedback for the active user)

### Personalization: Empirical Bayes Prior Calibration

The application features a privacy-preserving, mathematically sound personalization mechanism:
- **Empirical Bayes with Bounded Shrinkage:** Replaces coarse heuristic tuning or unfeasible on-device neural fine-tuning. Base model posterior log-odds are adjusted using Laplace-smoothed user emotion priors $P_{\text{user}}(e)$ and confusion transitions $T(e \mid e_{\text{base}})$.
- **Minimum Evidence Gating ($N \ge 5$):** Personalization remains strictly inactive until the user provides at least 5 verified feedback events, preventing ungrounded alterations.
- **Confidence Bounding:** Bounded shrinkage ($\beta_{\max} = 0.65, K_{\text{half}} = 10$) mathematically ensures that unambiguous, high-confidence base predictions ($p_0 \ge 0.85$) are never distorted by noisy feedback.
- **Side-by-Side Storage:** Both original model prediction and personalized prediction are stored separately and displayed transparently in the UI.

To run the reproducible walk-forward evaluation benchmark:

```powershell
python "dataset 1/evaluate_personalization.py"
```

Evaluated on the held-out GoEmotions test split (5,428 samples) across distinct simulated user personas using a strict walk-forward (zero-leakage) protocol, this method achieved:
- **Pre-threshold ($N < 5$):** Exactly $0.00\%$ deviation from base model.
- **Post-threshold ($N \ge 5$):** +6.36% average accuracy improvement, +0.0321 macro-F1 gain, -0.0478 Brier score calibration reduction, and -6.36% reduction in manual user correction burden.

### Security and local-first boundary

The application is engineered strictly as a **local-first, on-device** architecture:
- Loopback binding (`127.0.0.1`) is enforced by default.
- Cross-user tenant access is rejected with `HTTP 403 Forbidden` if an unauthorized identifier is passed.
- All sentiment analysis, transformer emotion inference, and personalization calibration run 100% locally on SQLite; journal content is never transmitted to cloud LLMs or third parties.
- Do not bind to public addresses (`0.0.0.0`) without adding multi-user cryptographic authentication (e.g., OAuth2/JWT), TLS encryption, and SQLCipher database encryption.

### Upgraded features in v1.2.0

1. **Empirical Bayes Personalization:** Seamlessly adapts ambiguous emotion classifications to individual linguistic habits without catastrophic forgetting.
2. **Prediction Feedback & Agreement:** Dual buttons allow users to confirm prediction accuracy (`👍 Accurate`) or correct the emotion via a dropdown selector populated with supported emotion categories.
3. **In-Page Accessible Editor:** Multi-line journal entries can be edited directly in an accessible modal dialog preserving line breaks and paragraph spacing.
4. **Comprehensive CSV Export:** Downloads complete datasets including sentiment scores, emotion tags, model confidence, user feedback status, and formula injection sanitization.
5. **Non-Blocking Lifespan Model Initialization:** Model loading is managed through FastAPI lifespan context handlers, eliminating import freezes.

