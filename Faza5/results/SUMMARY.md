# Final test

300 tasks or scenes per line, seeds: block 950000, letters 961000/971000/981000, word 992000.

| Stage | Task | Model | Success |
| --- | --- | --- | ---: |
| 1 | block pose | one policy | 280/300 (93.3 %) |
| 1 | block pose | manager: rotation + pushing | 283/300 (94.3 %) |
| 2 | letter A | 3 letter policies | 283/300 (94.3 %) |
| 2 | letter R | 3 letter policies | 259/300 (86.3 %) |
| 2 | letter M | 3 letter policies | 261/300 (87.0 %) |
| 2 | **word ARM** | 3 letter policies + manager | **251/300 (83.7 %)** |
| 3 | letter A | 1 shared policy | 279/300 (93.0 %) |
| 3 | letter R | 1 shared policy | 263/300 (87.7 %) |
| 3 | letter M | 1 shared policy | 269/300 (89.7 %) |
| 3 | **word ARM** | 1 shared policy + manager | **244/300 (81.3 %)** |
