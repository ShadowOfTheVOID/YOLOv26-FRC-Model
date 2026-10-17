# Bumper-number labels

Hand-read team numbers for robot crops, one `index team` per line, used by
`train/train_bumper_ocr.py`. A crop index is its position in the output of
`train_bumper_ocr.py extract` with the arguments below; the extraction is
deterministic (the released `fuel_withBotbest.pt`, imgsz 960, conf 0.35,
starting at 10 s), so the same video and step give the same indices.
Unreadable crops are simply left out.

| file | video (Drive, 1080p broadcast) | step | roster red / blue | use |
|---|---|---|---|---|
| e8_labels.txt | Einstein 8 | 2.0 s | 8613 6800 254 / 4065 4414 1323 | train |
| e4_labels.txt | Einstein 4 | 2.5 s | 4065 4414 1323 / 2337 4946 6324 | train |
| e5_labels.txt | Einstein 5 | 2.5 s | 9470 5940 7407 / 6329 2056 3506 | train, **blue only** |
| e1_labels.txt | Einstein 1 | 2.5 s | 5940 9470 7407 / 2046 2910 868 | **held out** |

Einstein 5's red alliance is Einstein 1's red alliance, so its red crops were
not labelled: training on them would leak the test set.
