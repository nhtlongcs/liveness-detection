# Liveness detection

## Table of Contents

## Problem statement

In verification services related to face recognition (such as eKYC and face access control), the key question is whether the input face video is real (from a live person present at the point of capture), or fake (from a spoof artifact or lifeless body). Liveness detection is the AI problem to answer that question.

In this challenge, participants will build a liveness detection model to classify if a given facial video is real or spoofed.

- Input: a video of selfie/portrait face with a length of 1-5 seconds (you can use any frames you like).

- Output: Liveness score in [0...1] (0 = Fake, 1 = Real).

Example Output: Predict.csv

| fname        | liveness_score |
|--------------|----------------|
| VideoID.mp4  | 0.10372        |
| ,,,          | ...            |
| ,,,          | ...            |
| ,,,          | ...            |

## Data preparation
