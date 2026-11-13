# Phase 6 results

## A1 - repair passes

### Stage 2 (3 policies)

| variant | success | percent | ci | mean_steps | median_steps | disturbed | repair_phases | pieces | categories |
|---|---|---|---|---|---|---|---|---|---|
| A1-0 | 168/500 | 33.6 | 29.6–37.9 | 1350 | 1284 | 287/500 (57 %) | 0.0 | 1.59 | passes_exhausted 292, letter_outside 34, high_contact 6 |
| A1-1 | 306/500 | 61.2 | 56.9–65.4 | 1678 | 1541 | 294/500 (59 %) | 0.77 | 1.59 | passes_exhausted 138, letter_outside 39, high_contact 17 |
| A1-2 | 385/500 | 77.0 | 73.1–80.5 | 1831 | 1584 | 295/500 (59 %) | 1.07 | 1.59 | passes_exhausted 59, letter_outside 39, high_contact 17 |
| A1-4 | 421/500 | 84.2 | 80.7–87.1 | 1965 | 1584 | 296/500 (59 %) | 1.26 | 1.59 | letter_outside 39, high_contact 18, passes_exhausted 18, total_limit 4 |

Paired differences on the same scenes (95 % CI by Newcombe, exact McNemar; discordant = scenes solved only by the first / only by the second):

| comparison | diff_pp | ci_pp | p | discordant |
|---|---|---|---|---|
| A1-1 − A1-0 | 27.6 | 23.6 … 31.4 | < 0.001 | 138 / 0 |
| A1-2 − A1-1 | 15.8 | 12.6 … 19.0 | < 0.001 | 79 / 0 |
| A1-4 − A1-2 | 7.2 | 4.9 … 9.6 | < 0.001 | 36 / 0 |
| A1-0 − A1-4 | -50.6 | -54.8 … -46.0 | < 0.001 | 0 / 253 |
| A1-1 − A1-4 | -23.0 | -26.7 … -19.3 | < 0.001 | 0 / 115 |
| A1-2 − A1-4 | -7.2 | -9.6 … -4.9 | < 0.001 | 0 / 36 |

Final errors, median / P90:

| variant | A | R | M |
|---|---|---|---|
| A1-0 | 1.57 / 8.66 cm, 5.3 / 65.0° | 1.54 / 8.97 cm, 2.9 / 57.4° | 1.18 / 5.30 cm, 2.3 / 11.0° |
| A1-1 | 1.26 / 3.98 cm, 3.1 / 22.2° | 1.29 / 3.78 cm, 1.9 / 17.8° | 1.31 / 10.56 cm, 2.7 / 40.7° |
| A1-2 | 1.23 / 1.89 cm, 2.8 / 7.9° | 1.29 / 2.88 cm, 1.8 / 9.0° | 1.19 / 6.79 cm, 2.5 / 21.5° |
| A1-4 | 1.23 / 1.88 cm, 2.8 / 7.8° | 1.25 / 1.94 cm, 1.7 / 7.6° | 1.17 / 1.99 cm, 2.3 / 9.1° |

### Stage 3 (shared policy)

| variant | success | percent | ci | mean_steps | median_steps | disturbed | repair_phases | pieces | categories |
|---|---|---|---|---|---|---|---|---|---|
| A1-0 | 176/500 | 35.2 | 31.1–39.5 | 1353 | 1298 | 285/500 (57 %) | 0.0 | 1.59 | passes_exhausted 287, letter_outside 31, high_contact 6 |
| A1-1 | 319/500 | 63.8 | 59.5–67.9 | 1687 | 1556 | 288/500 (58 %) | 0.76 | 1.59 | passes_exhausted 130, letter_outside 37, high_contact 14 |
| A1-2 | 388/500 | 77.6 | 73.7–81.0 | 1837 | 1586 | 291/500 (58 %) | 1.05 | 1.59 | passes_exhausted 59, letter_outside 38, high_contact 15 |
| A1-4 | 422/500 | 84.4 | 81.0–87.3 | 1966 | 1586 | 293/500 (59 %) | 1.26 | 1.59 | letter_outside 40, passes_exhausted 20, high_contact 15, total_limit 3 |

Paired differences on the same scenes (95 % CI by Newcombe, exact McNemar; discordant = scenes solved only by the first / only by the second):

| comparison | diff_pp | ci_pp | p | discordant |
|---|---|---|---|---|
| A1-1 − A1-0 | 28.6 | 24.5 … 32.5 | < 0.001 | 143 / 0 |
| A1-2 − A1-1 | 13.8 | 10.8 … 16.9 | < 0.001 | 69 / 0 |
| A1-4 − A1-2 | 6.8 | 4.6 … 9.2 | < 0.001 | 34 / 0 |
| A1-0 − A1-4 | -49.2 | -53.4 … -44.6 | < 0.001 | 0 / 246 |
| A1-1 − A1-4 | -20.6 | -24.2 … -17.0 | < 0.001 | 0 / 103 |
| A1-2 − A1-4 | -6.8 | -9.2 … -4.6 | < 0.001 | 0 / 34 |

Final errors, median / P90:

| variant | A | R | M |
|---|---|---|---|
| A1-0 | 1.54 / 8.54 cm, 4.7 / 59.4° | 1.51 / 8.93 cm, 3.5 / 48.1° | 1.29 / 7.49 cm, 2.2 / 20.0° |
| A1-1 | 1.20 / 4.53 cm, 3.2 / 16.3° | 1.25 / 2.84 cm, 2.4 / 12.9° | 1.34 / 8.66 cm, 2.5 / 45.2° |
| A1-2 | 1.16 / 1.89 cm, 2.8 / 8.0° | 1.23 / 2.06 cm, 2.2 / 9.6° | 1.28 / 6.92 cm, 2.3 / 26.5° |
| A1-4 | 1.17 / 1.89 cm, 2.8 / 7.9° | 1.21 / 1.89 cm, 2.1 / 8.3° | 1.28 / 1.96 cm, 2.1 / 8.1° |

## A2 - route segments (one budget of 1600 policy steps per placement attempt)

### Stage 2 (3 policies)

| variant | success | percent | ci | mean_steps | median_steps | disturbed | repair_phases | pieces | categories |
|---|---|---|---|---|---|---|---|---|---|
| A2-direct | 229/500 | 45.8 | 41.5–50.2 | 4259 | 2611 | 221/500 (44 %) | 1.95 | 1.0 | total_limit 142, letter_outside 106, high_contact 15, passes_exhausted 8 |
| A2-6 | 430/500 | 86.0 | 82.7–88.8 | 1876 | 1584 | 280/500 (56 %) | 1.07 | 2.63 | letter_outside 45, high_contact 14, passes_exhausted 6, total_limit 5 |
| A2-12 | 422/500 | 84.4 | 81.0–87.3 | 2016 | 1584 | 298/500 (60 %) | 1.25 | 1.59 | letter_outside 41, high_contact 17, total_limit 15, passes_exhausted 5 |
| A2-24 | 287/500 | 57.4 | 53.0–61.7 | 3671 | 2094 | 252/500 (50 %) | 1.85 | 1.07 | total_limit 108, letter_outside 80, high_contact 17, passes_exhausted 8 |

Paired differences on the same scenes (95 % CI by Newcombe, exact McNemar; discordant = scenes solved only by the first / only by the second):

| comparison | diff_pp | ci_pp | p | discordant |
|---|---|---|---|---|
| A2-direct − A2-12 | -38.6 | -43.1 … -33.9 | < 0.001 | 9 / 202 |
| A2-6 − A2-12 | 1.6 | -2.1 … 5.3 | 0.456 | 48 / 40 |
| A2-24 − A2-12 | -27.0 | -31.4 … -22.5 | < 0.001 | 14 / 149 |

Final errors, median / P90:

| variant | A | R | M |
|---|---|---|---|
| A2-direct | 1.38 / 25.61 cm, 3.7 / 99.4° | 1.35 / 20.81 cm, 2.7 / 113.8° | 1.44 / 22.76 cm, 3.0 / 126.9° |
| A2-6 | 1.16 / 1.88 cm, 3.2 / 8.0° | 1.25 / 1.93 cm, 1.4 / 7.4° | 1.28 / 5.47 cm, 2.4 / 22.0° |
| A2-12 | 1.23 / 1.88 cm, 2.8 / 7.9° | 1.25 / 1.93 cm, 1.9 / 7.8° | 1.18 / 1.99 cm, 2.4 / 9.3° |
| A2-24 | 1.26 / 13.88 cm, 3.0 / 40.3° | 1.34 / 20.46 cm, 2.7 / 113.8° | 1.35 / 15.88 cm, 2.6 / 119.1° |

### Stage 3 (shared policy)

| variant | success | percent | ci | mean_steps | median_steps | disturbed | repair_phases | pieces | categories |
|---|---|---|---|---|---|---|---|---|---|
| A2-direct | 234/500 | 46.8 | 42.5–51.2 | 4190 | 2251 | 206/500 (41 %) | 1.9 | 1.0 | total_limit 151, letter_outside 92, high_contact 17, passes_exhausted 4, top_contact 2 |
| A2-6 | 435/500 | 87.0 | 83.8–89.7 | 1890 | 1571 | 289/500 (58 %) | 1.12 | 2.63 | letter_outside 41, high_contact 13, total_limit 6, passes_exhausted 5 |
| A2-12 | 424/500 | 84.8 | 81.4–87.7 | 2030 | 1586 | 296/500 (59 %) | 1.26 | 1.59 | letter_outside 39, high_contact 15, total_limit 12, passes_exhausted 10 |
| A2-24 | 284/500 | 56.8 | 52.4–61.1 | 3664 | 1902 | 235/500 (47 %) | 1.87 | 1.08 | total_limit 112, letter_outside 81, high_contact 13, passes_exhausted 8, top_contact 2 |

Paired differences on the same scenes (95 % CI by Newcombe, exact McNemar; discordant = scenes solved only by the first / only by the second):

| comparison | diff_pp | ci_pp | p | discordant |
|---|---|---|---|---|
| A2-direct − A2-12 | -38.0 | -42.5 … -33.2 | < 0.001 | 12 / 202 |
| A2-6 − A2-12 | 2.2 | -1.8 … 6.2 | 0.320 | 56 / 45 |
| A2-24 − A2-12 | -28.0 | -32.5 … -23.3 | < 0.001 | 18 / 158 |

Final errors, median / P90:

| variant | A | R | M |
|---|---|---|---|
| A2-direct | 1.36 / 25.39 cm, 4.1 / 105.1° | 1.44 / 20.18 cm, 3.2 / 142.8° | 1.48 / 24.01 cm, 3.1 / 131.6° |
| A2-6 | 1.13 / 1.81 cm, 2.8 / 7.9° | 1.24 / 1.90 cm, 1.8 / 7.5° | 1.19 / 2.06 cm, 2.2 / 10.2° |
| A2-12 | 1.17 / 1.89 cm, 2.8 / 7.8° | 1.21 / 1.89 cm, 2.2 / 8.6° | 1.28 / 2.19 cm, 2.1 / 9.1° |
| A2-24 | 1.26 / 15.01 cm, 3.5 / 32.8° | 1.43 / 20.34 cm, 3.2 / 141.5° | 1.29 / 16.04 cm, 2.7 / 102.5° |

## A3 - training states (three letter policies, Stage 2)

Word ARM on the 500 final scenes; letters = single-letter tasks solved out of 300 (new sets 1301000 / 1311000 / 1321000); events = disturbance events per scene.

| variant | success | percent | ci | mean_steps | disturbed | events | repair_phases | letters | categories |
|---|---|---|---|---|---|---|---|---|---|
| A3-start seed - | 391/500 | 78.2 | 74.4–81.6 | 2500 | 366/500 (73 %) | 1.74 | 1.95 | 275 / 270 / 270 | letter_outside 43, passes_exhausted 39, high_contact 17, total_limit 9, top_contact 1 |
| A3-single seed 0 | 402/500 | 80.4 | 76.7–83.6 | 2022 | 323/500 (65 %) | 1.43 | 1.44 | 269 / 275 / 272 | letter_outside 47, high_contact 27, passes_exhausted 20, total_limit 4 |
| A3-single seed 10000 | 403/500 | 80.6 | 76.9–83.8 | 2049 | 325/500 (65 %) | 1.4 | 1.43 | 287 / 270 / 276 | letter_outside 48, passes_exhausted 23, high_contact 21, total_limit 5 |
| A3-single seed 20000 | 399/500 | 79.8 | 76.1–83.1 | 2185 | 355/500 (71 %) | 1.62 | 1.67 | 284 / 275 / 283 | letter_outside 52, passes_exhausted 23, high_contact 17, total_limit 7, top_contact 2 |
| **A3-single mean of 3 seeds** | - | 80.3 ± 0.4 | 77.3–83.1 (seeds+scenes) | 2085 | 67 % | 1.48 | 1.51 | 280 / 273 / 277 | - |
| A3-mixed seed 0 | 402/500 | 80.4 | 76.7–83.6 | 1783 | 307/500 (61 %) | 1.29 | 1.2 | 268 / 260 / 277 | letter_outside 59, high_contact 26, passes_exhausted 9, top_contact 2, total_limit 2 |
| A3-mixed seed 10000 | 426/500 | 85.2 | 81.8–88.0 | 2001 | 327/500 (65 %) | 1.35 | 1.42 | 278 / 270 / 268 | letter_outside 33, passes_exhausted 22, high_contact 17, total_limit 2 |
| A3-mixed seed 20000 | 416/500 | 83.2 | 79.7–86.2 | 2005 | 307/500 (61 %) | 1.27 | 1.37 | 278 / 262 / 274 | letter_outside 43, passes_exhausted 21, high_contact 17, total_limit 2, top_contact 1 |
| **A3-mixed mean of 3 seeds** | - | 82.9 ± 2.4 | 79.4–86.3 (seeds+scenes) | 1930 | 63 % | 1.3 | 1.33 | 275 / 264 / 273 | - |

Differences with a 95 % bootstrap interval that resamples training seeds and scenes; per seed = difference and exact McNemar p of each trained model against the one start model.

| comparison | metric | diff_pp | ci_pp | per_seed |
|---|---|---|---|---|
| A3-mixed − A3-single | success | 2.7 | -1.1 … 6.6 | - |
| A3-single − A3-start | success | 2.1 | -1.7 … 5.8 | +2.2 (p 0.359), +2.4 (p 0.303), +1.6 (p 0.512) |
| A3-mixed − A3-start | success | 4.7 | 0.4 … 9.0 | +2.2 (p 0.367), +7.0 (p 0.001), +5.0 (p 0.025) |
| A3-mixed − A3-single | disturbed | -4.1 | -9.5 … 0.9 | - |

## A4 - disturbance penalty (three letter policies, Stage 2)

Word ARM on the 500 final scenes; letters = single-letter tasks solved out of 300 (new sets 1301000 / 1311000 / 1321000); events = disturbance events per scene.

| variant | success | percent | ci | mean_steps | disturbed | events | repair_phases | letters | categories |
|---|---|---|---|---|---|---|---|---|---|
| A4-bez seed 0 | 427/500 | 85.4 | 82.0–88.2 | 1965 | 300/500 (60 %) | 1.28 | 1.34 | 281 / 262 / 272 | letter_outside 40, passes_exhausted 18, high_contact 13, total_limit 2 |
| A4-bez seed 10000 | 414/500 | 82.8 | 79.2–85.9 | 1914 | 317/500 (63 %) | 1.38 | 1.34 | 275 / 265 / 281 | letter_outside 42, high_contact 23, passes_exhausted 19, total_limit 2 |
| A4-bez seed 20000 | 430/500 | 86.0 | 82.7–88.8 | 1939 | 312/500 (62 %) | 1.28 | 1.32 | 283 / 251 / 274 | letter_outside 30, high_contact 25, passes_exhausted 9, total_limit 5, top_contact 1 |
| **A4-bez mean of 3 seeds** | - | 84.7 ± 1.7 | 81.7–87.7 (seeds+scenes) | 1939 | 62 % | 1.31 | 1.33 | 280 / 259 / 276 | - |
| A4-s (= A3-mixed) seed 0 | 402/500 | 80.4 | 76.7–83.6 | 1783 | 307/500 (61 %) | 1.29 | 1.2 | 268 / 260 / 277 | letter_outside 59, high_contact 26, passes_exhausted 9, top_contact 2, total_limit 2 |
| A4-s (= A3-mixed) seed 10000 | 426/500 | 85.2 | 81.8–88.0 | 2001 | 327/500 (65 %) | 1.35 | 1.42 | 278 / 270 / 268 | letter_outside 33, passes_exhausted 22, high_contact 17, total_limit 2 |
| A4-s (= A3-mixed) seed 20000 | 416/500 | 83.2 | 79.7–86.2 | 2005 | 307/500 (61 %) | 1.27 | 1.37 | 278 / 262 / 274 | letter_outside 43, passes_exhausted 21, high_contact 17, total_limit 2, top_contact 1 |
| **A4-s (= A3-mixed) mean of 3 seeds** | - | 82.9 ± 2.4 | 79.4–86.3 (seeds+scenes) | 1930 | 63 % | 1.3 | 1.33 | 275 / 264 / 273 | - |

Differences with a 95 % bootstrap interval that resamples training seeds and scenes; per seed = difference and exact McNemar p of each trained model against the one start model.

| comparison | metric | diff_pp | ci_pp | per_seed |
|---|---|---|---|---|
| A4-bez − A4-s (= A3-mixed) | disturbed | -0.8 | -5.3 … 3.4 | - |
| A4-bez − A4-s (= A3-mixed) | success | 1.8 | -2.3 … 5.9 | - |

## A5 - one-hot letter ID (shared policy, Stage 3)

Word ARM on the 500 final scenes; letters = single-letter tasks solved out of 300 (new sets 1301000 / 1311000 / 1321000); events = disturbance events per scene.

| variant | success | percent | ci | mean_steps | disturbed | events | repair_phases | letters | categories |
|---|---|---|---|---|---|---|---|---|---|
| A5-noid seed 0 | 411/500 | 82.2 | 78.6–85.3 | 2061 | 321/500 (64 %) | 1.26 | 1.38 | 268 / 264 / 266 | letter_outside 48, passes_exhausted 20, high_contact 18, total_limit 3 |
| A5-noid seed 10000 | 411/500 | 82.2 | 78.6–85.3 | 2041 | 321/500 (64 %) | 1.28 | 1.36 | 274 / 271 / 265 | letter_outside 49, high_contact 17, passes_exhausted 16, total_limit 7 |
| A5-noid seed 20000 | 415/500 | 83.0 | 79.5–86.0 | 1956 | 302/500 (60 %) | 1.21 | 1.27 | 275 / 277 / 253 | letter_outside 51, passes_exhausted 17, high_contact 14, total_limit 3 |
| **A5-noid mean of 3 seeds** | - | 82.5 ± 0.5 | 79.5–85.3 (seeds+scenes) | 2019 | 63 % | 1.25 | 1.34 | 272 / 271 / 261 | - |
| A5-id seed 0 | 409/500 | 81.8 | 78.2–84.9 | 1963 | 287/500 (57 %) | 1.09 | 1.21 | 282 / 261 / 269 | letter_outside 51, high_contact 23, passes_exhausted 14, total_limit 3 |
| A5-id seed 10000 | 407/500 | 81.4 | 77.8–84.6 | 1930 | 289/500 (58 %) | 1.12 | 1.2 | 279 / 257 / 270 | letter_outside 55, high_contact 22, passes_exhausted 15, total_limit 1 |
| A5-id seed 20000 | 412/500 | 82.4 | 78.8–85.5 | 1972 | 292/500 (58 %) | 1.14 | 1.25 | 278 / 263 / 264 | letter_outside 47, high_contact 20, passes_exhausted 18, total_limit 3 |
| **A5-id mean of 3 seeds** | - | 81.9 ± 0.5 | 78.8–84.8 (seeds+scenes) | 1955 | 58 % | 1.12 | 1.22 | 280 / 260 / 268 | - |

Differences with a 95 % bootstrap interval that resamples training seeds and scenes; per seed = difference and exact McNemar p of each trained model against the one start model.

| comparison | metric | diff_pp | ci_pp | per_seed |
|---|---|---|---|---|
| A5-id − A5-noid | success | -0.6 | -3.5 … 2.4 | - |
| A5-id − A5-noid | disturbed | -5.1 | -8.9 … -1.2 | - |
