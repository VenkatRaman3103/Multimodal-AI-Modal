import os

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import cv2
import numpy as np
import torch
import subprocess


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)

        self.video_dir = video_dir

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "fear": 2,
            "joy": 3,
            "neutral": 4,
            "sadness": 5,
            "surpise": 6,
        }

        self.sentimap_map = {"negaitve": 0, "neutral": 1, "positive": 2}
    
    def _load_video_frames(self, video_path):
        FRAME_LIMIT = 30
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            while len(frames) < FRAME_LIMIT and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video file: {video_path}")

        # pad or truncate
        if len(frames) < FRAME_LIMIT:
            frames += [np.zeros_like(frames[0])] * (FRAME_LIMIT - len(frames))
        else:
            frames = frames[:FRAME_LIMIT]

        # (T, H, W, C) -> (T, C, H, W) 
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2) # (T, C, H, W)

    def __extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path,
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            raise ValueError(f"Error extracting audio from {video_path}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        path = os.path.join(self.video_dir, video_filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        text_input = self.tokenizer(
            row["Utterance"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # video_frames = self._load_video_frames(path)
        self.__extract_audio_features(path)

        # print(video_frames)

        # return {
        #     "input_ids": text_input["input_ids"].squeeze(0),
        #     "attention_mask": text_input["attention_mask"].squeeze(0),
        #     "video": video_frames,
        # }


if __name__ == "__main__":
    meld = MELDDataset(
        "dataset/dev/dev_sent_emo.csv",
        "dataset/dev/dev_splits_complete/"
    )

    sample = meld[0]
    print(sample)