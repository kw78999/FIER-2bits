# LongBench FIER/2mean reproduction

Scores are percentages. Only actually completed samples are included. Original Llama-3 has an 8K configured context window; the paper-requested 32K setting extrapolates RoPE beyond that window.

## Task scores

| Method | Budget | narrativeqa | qasper | multifieldqa_en | hotpotqa | gov_report | triviaqa | Average |
|---|---|---|---|---|---|---|---|---|
| 2mean | 1024 | 6.909 | 42.254 | 20.546 | 6.433 | 15.540 | 42.105 | 22.298 |
| 2mean | 2048 | 6.966 | 46.435 | 21.440 | 5.556 | 16.115 | 41.053 | 22.927 |
| 2mean | 4096 | 6.154 | 45.427 | 21.258 | 5.817 | 15.499 | 41.053 | 22.535 |
| 2mean | 512 | 4.590 | 38.534 | 20.980 | 6.060 | 15.229 | 42.105 | 21.250 |
| fier | 1024 | 6.164 | 45.813 | 20.591 | 5.556 | 15.834 | 41.053 | 22.502 |
| fier | 2048 | 5.959 | 45.707 | 21.580 | 5.556 | 15.334 | 41.053 | 22.531 |
| fier | 4096 | 6.284 | 45.737 | 21.252 | 5.556 | 15.622 | 41.053 | 22.584 |
| fier | 512 | 5.802 | 42.783 | 20.673 | 5.564 | 14.757 | 42.105 | 21.947 |
| full |  | 6.118 | 47.013 | 20.907 | 5.556 | 15.619 | 41.053 | 22.711 |
| quest | 1024 | 6.294 | 47.419 | 20.900 | 6.040 | 18.394 | 35.789 | 22.473 |
| quest | 2048 | 6.511 | 47.030 | 21.408 | 6.074 | 17.331 | 35.789 | 22.357 |
| quest | 4096 | 5.943 | 46.853 | 20.363 | 6.194 | 17.492 | 41.053 | 22.983 |
| quest | 512 | 6.773 | 42.833 | 21.129 | 6.104 | 16.770 | 35.789 | 21.566 |

## Retention vs Full

| Method | Budget | Average | Retention % | Min task % |
|---|---|---|---|---|
| 2mean | 1024 | 22.298 | 103.156 | 89.877 |
| 2mean | 2048 | 22.927 | 103.059 | 98.770 |
| 2mean | 4096 | 22.535 | 100.473 | 96.627 |
| 2mean | 512 | 21.250 | 94.415 | 75.021 |
| fier | 1024 | 22.502 | 99.678 | 97.447 |
| fier | 2048 | 22.531 | 99.337 | 97.221 |
| fier | 4096 | 22.584 | 100.278 | 97.286 |
| fier | 512 | 21.947 | 96.986 | 91.003 |
| full |  | 22.711 | 100.000 | 100.000 |
| quest | 1024 | 22.473 | 102.895 | 87.179 |
| quest | 2048 | 22.357 | 102.724 | 87.179 |
| quest | 4096 | 22.983 | 102.948 | 97.144 |
| quest | 512 | 21.566 | 101.217 | 87.179 |

## Runtime

| Method | Budget | Dataset | Decode ms/tok | Search | Attention | Failed |
|---|---|---|---|---|---|---|
| 2mean | 1024 | gov_report | 110.074 | 3.145 | 4.554 | 0 |
| 2mean | 2048 | gov_report | 110.216 | 3.173 | 4.069 | 0 |
| 2mean | 4096 | gov_report | 117.522 | 3.120 | 5.561 | 0 |
| 2mean | 512 | gov_report | 110.633 | 3.127 | 4.538 | 0 |
| 2mean | 1024 | hotpotqa | 111.452 | 3.677 | 5.120 | 0 |
| 2mean | 2048 | hotpotqa | 111.339 | 3.551 | 4.616 | 0 |
| 2mean | 4096 | hotpotqa | 118.652 | 3.466 | 6.037 | 0 |
| 2mean | 512 | hotpotqa | 111.125 | 3.508 | 5.082 | 0 |
| 2mean | 1024 | multifieldqa_en | 109.066 | 2.802 | 4.028 | 0 |
| 2mean | 2048 | multifieldqa_en | 109.417 | 2.790 | 3.539 | 0 |
| 2mean | 4096 | multifieldqa_en | 117.096 | 2.780 | 4.789 | 0 |
| 2mean | 512 | multifieldqa_en | 109.703 | 2.734 | 4.022 | 0 |
| 2mean | 1024 | narrativeqa | 114.348 | 4.638 | 7.363 | 0 |
| 2mean | 2048 | narrativeqa | 114.830 | 4.664 | 6.913 | 0 |
| 2mean | 4096 | narrativeqa | 122.358 | 4.628 | 8.399 | 0 |
| 2mean | 512 | narrativeqa | 114.868 | 4.609 | 7.382 | 0 |
| 2mean | 1024 | qasper | 107.881 | 2.251 | 3.590 | 0 |
| 2mean | 2048 | qasper | 108.739 | 2.339 | 3.120 | 0 |
| 2mean | 4096 | qasper | 114.997 | 2.193 | 4.453 | 0 |
| 2mean | 512 | qasper | 109.067 | 2.265 | 3.594 | 0 |
| 2mean | 1024 | triviaqa | 109.172 | 3.034 | 4.531 | 0 |
| 2mean | 2048 | triviaqa | 110.124 | 3.049 | 4.058 | 0 |
| 2mean | 4096 | triviaqa | 116.732 | 2.990 | 5.471 | 0 |
| 2mean | 512 | triviaqa | 109.928 | 3.009 | 4.522 | 0 |
| fier | 1024 | gov_report | 114.972 | 7.663 | 4.588 | 0 |
| fier | 2048 | gov_report | 115.568 | 7.698 | 4.069 | 0 |
| fier | 4096 | gov_report | 119.015 | 5.539 | 5.563 | 0 |
| fier | 512 | gov_report | 116.216 | 7.765 | 4.574 | 0 |
| fier | 1024 | hotpotqa | 115.846 | 8.098 | 5.130 | 0 |
| fier | 2048 | hotpotqa | 116.524 | 8.135 | 4.609 | 0 |
| fier | 4096 | hotpotqa | 120.537 | 6.503 | 6.045 | 0 |
| fier | 512 | hotpotqa | 116.392 | 8.060 | 5.098 | 0 |
| fier | 1024 | multifieldqa_en | 113.397 | 6.911 | 4.055 | 0 |
| fier | 2048 | multifieldqa_en | 115.137 | 7.209 | 3.543 | 0 |
| fier | 4096 | multifieldqa_en | 117.855 | 5.441 | 4.787 | 0 |
| fier | 512 | multifieldqa_en | 115.123 | 7.084 | 4.048 | 0 |
| fier | 1024 | narrativeqa | 119.903 | 10.454 | 7.413 | 0 |
| fier | 2048 | narrativeqa | 119.790 | 10.354 | 6.910 | 0 |
| fier | 4096 | narrativeqa | 125.058 | 8.898 | 8.408 | 0 |
| fier | 512 | narrativeqa | 120.126 | 10.380 | 7.401 | 0 |
| fier | 1024 | qasper | 113.513 | 6.687 | 3.633 | 0 |
| fier | 2048 | qasper | 114.317 | 6.788 | 3.111 | 0 |
| fier | 4096 | qasper | 117.739 | 4.277 | 4.463 | 0 |
| fier | 512 | qasper | 114.725 | 6.757 | 3.610 | 0 |
| fier | 1024 | triviaqa | 115.160 | 7.670 | 4.581 | 0 |
| fier | 2048 | triviaqa | 115.278 | 7.627 | 4.057 | 0 |
| fier | 4096 | triviaqa | 118.695 | 5.556 | 5.478 | 0 |
| fier | 512 | triviaqa | 118.290 | 7.629 | 4.551 | 0 |
| full |  | gov_report | 69.435 | 0.000 | 36.029 | 0 |
| full |  | hotpotqa | 79.771 | 0.000 | 45.018 | 0 |
| full |  | multifieldqa_en | 59.071 | 0.000 | 27.093 | 0 |
| full |  | narrativeqa | 123.448 | 0.000 | 83.021 | 0 |
| full |  | qasper | 50.429 | 0.000 | 19.831 | 0 |
| full |  | triviaqa | 68.836 | 0.000 | 35.622 | 0 |
| quest | 1024 | gov_report | 118.310 | 24.376 | 5.948 | 0 |
| quest | 2048 | gov_report | 118.695 | 24.448 | 5.676 | 0 |
| quest | 4096 | gov_report | 128.686 | 24.599 | 8.619 | 0 |
| quest | 512 | gov_report | 118.070 | 23.895 | 5.271 | 1 |
| quest | 1024 | hotpotqa | 122.912 | 28.712 | 6.486 | 0 |
| quest | 2048 | hotpotqa | 124.554 | 28.841 | 6.241 | 0 |
| quest | 4096 | hotpotqa | 132.871 | 28.843 | 9.022 | 0 |
| quest | 512 | hotpotqa | 123.558 | 28.696 | 5.870 | 0 |
| quest | 1024 | multifieldqa_en | 114.437 | 20.246 | 5.455 | 0 |
| quest | 2048 | multifieldqa_en | 113.847 | 20.222 | 5.127 | 0 |
| quest | 4096 | multifieldqa_en | 122.966 | 20.428 | 7.593 | 0 |
| quest | 512 | multifieldqa_en | 114.701 | 20.234 | 4.835 | 0 |
| quest | 1024 | narrativeqa | 143.065 | 45.490 | 8.823 | 0 |
| quest | 2048 | narrativeqa | 142.989 | 45.491 | 8.519 | 0 |
| quest | 4096 | narrativeqa | 152.774 | 45.601 | 11.482 | 0 |
| quest | 512 | narrativeqa | 142.506 | 45.368 | 8.167 | 0 |
| quest | 1024 | qasper | 108.642 | 16.044 | 4.979 | 0 |
| quest | 2048 | qasper | 109.538 | 16.102 | 4.715 | 0 |
| quest | 4096 | qasper | 118.575 | 16.232 | 7.366 | 0 |
| quest | 512 | qasper | 108.530 | 15.932 | 4.344 | 0 |
| quest | 1024 | triviaqa | 118.347 | 24.019 | 5.949 | 0 |
| quest | 2048 | triviaqa | 118.618 | 24.056 | 5.661 | 0 |
| quest | 4096 | triviaqa | 127.633 | 24.144 | 8.450 | 0 |
| quest | 512 | triviaqa | 118.598 | 23.990 | 5.321 | 0 |
