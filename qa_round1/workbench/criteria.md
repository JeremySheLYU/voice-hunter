# Voice Hunter workbench acceptance criteria (round 1)
Scope: ordinary operation, with an isolated loopback instance or isolated in-process session when loopback unavailable. No writes to product code. No offensive or security testing.

1. At desktop and narrow viewport, critical controls should be visible and accessibly labeled. Real browser navigation/keyboard/pointer actions and actual layout can only pass when observed in a browser engine. Mark not tested where unavailable.
2. Open the instrument, duet and authorized optional iPhone demo through normal sample URLs. The right filename/mode/guide windows should be selected. An absent optional phone sample should have no visible button.
3. Import a stereo MOV with Unicode filename. Read a 2-channel timeline. Submit valid example or speech model inputs, and audition target, residual and source through their matching media URLs.
4. Render adjusted standard stereo and headphone mixes. Changing controls after render should warn that the downloadable mix is stale. Compare A/B at the same requested position. Re-render and verify a new matching render URL. Validate complete downloaded WAV and MOV bytes with ffprobe and decoded audio, not HTTP status alone.
5. Changing source, separation method, target option or example selection should not leave a previous target/export presented as though it were the new one. Explicit last-applied labels or stale notices are acceptable; invisibly old results fail.
6. Recover from invalid interval, overlapping examples, and unsupported mono input. Busy state must clear, source must remain usable after rejection, and subsequent valid processing must succeed.
7. Report real HTTP, actual OS browser and DOM-harness observations separately. Simulated playback verifies event wiring only; rendered/heard behavior needs a browser/listener. Use generated MOV/WAV inputs for reproducibility, no person/private clip needed.
