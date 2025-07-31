import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

class SentimentDataset(Dataset):
    """감정 분석을 위한 데이터셋 클래스"""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentTrainer:
    """감정 분석 모델 학습 클래스"""
    def __init__(self, model_name="klue/roberta-small", num_labels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.max_len = 128
        
    def compute_metrics(self, pred):
        """평가 지표 계산"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1}
    
    def prepare_data(self, train_df, val_df):
        """데이터셋 준비"""
        train_dataset = SentimentDataset(
            texts=train_df['kor_sentence'].values,
            labels=train_df['labels'].values,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        
        val_dataset = SentimentDataset(
            texts=val_df['kor_sentence'].values,
            labels=val_df['labels'].values,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir='./results'):
        """모델 학습 실행"""
        
        # --- 발생한 오류들을 해결하기 위해 수정된 부분 ---
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=100,
            eval_steps=100,
            save_total_limit=2,
            
            # 'evaluation_strategy'가 아닌 'eval_strategy' 사용
            eval_strategy="steps",
            save_strategy="steps",
            
            # EarlyStoppingCallback과 연동하기 위한 설정
            load_best_model_at_end=True,
            metric_for_best_model="f1", # 'f1' 또는 'accuracy' 등 compute_metrics의 반환 키값
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("\n모델 학습을 시작합니다...")
        trainer.train()
        
        # 최적 모델 저장장
        best_model_path = os.path.join(output_dir, 'best_model')
        trainer.save_model(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        
        print(f"\n모델 학습이 완료되었습니다. 최적의 모델이 {best_model_path}에 저장되었습니다.")
        
        return trainer