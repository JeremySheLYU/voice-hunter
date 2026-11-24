# Round 1 workbench QA

The standard and headphone mix, full MOV export and bad-input recovery execute against an isolated HTTP server. The suite generates its own 9.6-second stereo video and opens a second isolated service without a private sample configured. Both services close after the run; no real-user upload, installed workbench session, or public service is used.

Run from the project root with the product's Python/NumPy environment, FFmpeg, ffprobe, Node.js 20+ and linkedom:

```
python3 qa_round1/workbench/run.py --out qa_round1/workbench/rerun.json --strict
```

The script searches for local linkedom and Playwright packages, including the installed OpenClaw dependency path on this Mac. Use `LINKEDOM_MODULE` and `PLAYWRIGHT_MODULE` for alternate installations. Browser tests try a fresh Chrome, Playwright Chromium or Playwright WebKit process. The program records unavailable engines in the result; a missing browser does not become a desktop/mobile pass. The Playwright suite runs at 1280×800 and 375×812 when an engine works. The DOM harness explicitly simulates media playback, requested seeks and canvas calls. It loads the actual app page/script from HTTP, submits raw uploads, calls the actual processing endpoints and downloads complete files.

**Saved round 1 result:** 18 of 20 DOM, markup and HTTP workflow cases passed. Two cases failed the same user-facing expectation. After a successful render and export, changing the separation interval or switching separation method left prior-target solo, prior-target downloads and rendering enabled without marking the results as belonging to previous settings. Re-running extraction correctly cleared outputs. Changing a mix slider also correctly marked the previous mix as stale. The existing model note still said the video was ready for download across the tested separation edits.

The downloaded WAV and MOV were decoded independently with FFmpeg and had the same number of samples. Against the intentionally adjusted WAV, MOV audio correlation was 0.99981; against the original, 0.86518. Maximum absolute error when summing the two downloaded float stems and comparing with the downloaded original was 2.98e-8. These are export consistency measurements, not perceived audio quality. Full measurements are in results.json.

A genuine browser launch was attempted. Chrome exited with SIGABRT, bundled Chromium reported macOS bootstrap permission denied, and bundled WebKit exited 134. Layout, actual media playback, tap sizes, file dialogs and physical iPhone behavior remain unverified. Source markup checks confirmed control labels and responsive rules. No hearing or listener judgments were made.
